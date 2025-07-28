import soundfile
import time
import logging
import resampy
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm
from pathlib import Path
from torch.multiprocessing import Pool, set_start_method
from itertools import repeat

from .ims_tts import ImsTTS
from utils import create_clean_dir, setup_logger

set_start_method('spawn', force=True)
logger = setup_logger(__name__)


class SpeechSynthesis:
    def __init__(self, devices, settings, model_dir=None, results_dir=None, save_output=True, force_compute=False):
        self.devices = devices
        self.output_sr = settings.get('output_sr', 16000)
        self.save_output = save_output
        self.force_compute = force_compute if force_compute else settings.get('force_compute_synthesis', False)

        synthesizer_type = settings.get('synthesizer', 'ims')
        if synthesizer_type == 'ims':
            hifigan_path = settings['hifigan_path']
            fastspeech_path = settings['fastspeech_path']
            embedding_path = settings.get('embeddings_path', None)

            self.tts_models = []
            for device in self.devices:
                self.tts_models.append(ImsTTS(hifigan_path=hifigan_path, fastspeech_path=fastspeech_path,
                                              embedding_path=embedding_path, device=device,
                                              output_sr=self.output_sr, lang=settings.get('lang', 'en')))

        if results_dir:
            self.results_dir = results_dir
        elif 'results_path' in settings:
            self.results_dir = settings['results_path']
        elif 'results_dir' in settings:
            self.results_dir = settings['results_dir']
        else:
            if self.save_output:
                raise ValueError('Results dir must be specified in parameters or settings!')

    def synthesize_speech(self, dataset_name, texts, speaker_embeddings, prosody=None, emb_level='spk'):
        # depending on whether we save the generated audios to disk or not, we either return a dict of paths to the
        # saved wavs (wav.scp) or the wavs themselves
        dataset_results_dir = self.results_dir / dataset_name if self.save_output else ''
        wavs = {}

        if dataset_results_dir.exists() and not self.force_compute:
            already_synthesized_utts = {wav_file.stem: str(wav_file.absolute())
                                        for wav_file in dataset_results_dir.glob('*.wav')
                                        if wav_file.stem in texts.utterances}

            if len(already_synthesized_utts):
                logger.info(f'No synthesis necessary for {len(already_synthesized_utts)} of {len(texts)} utterances...')
                texts.remove_instances(list(already_synthesized_utts.keys()))
                if self.save_output:
                    wavs = already_synthesized_utts
                else:
                    wavs = {}
                    for utt, wav_file in already_synthesized_utts.items():
                        wav, _ = soundfile.read(wav_file)
                        wavs[utt] = wav

        if texts:
            logger.info(f'Synthesize {len(texts)} utterances...')
            if self.force_compute or not dataset_results_dir.exists():
                create_clean_dir(dataset_results_dir)

            text_is_phones = texts.is_phones

            if len(self.tts_models) == 1:
                instances = []
                for text, utt, speaker in texts:
                    try:
                        if emb_level == 'spk':
                            speaker_embedding = speaker_embeddings.get_embedding_for_identifier(speaker)
                        else:
                            speaker_embedding = speaker_embeddings.get_embedding_for_identifier(utt)

                        if prosody:
                            utt_prosody_dict = prosody.get_instance(utt)
                        else:
                            utt_prosody_dict = {}
                        instances.append((text, utt, speaker_embedding, utt_prosody_dict))
                    except KeyError:
                        logger.warn(f'Key error at {utt}')
                        continue
                wavs.update(synthesis_job(instances=instances, tts_model=self.tts_models[0],
                                          out_dir=dataset_results_dir, sleep=0, text_is_phones=text_is_phones,
                                          save_output=self.save_output))

            else:
                num_processes = len(self.tts_models)
                sleeps = [10 * i for i in range(num_processes)]
                text_iterators = texts.get_iterators(n=num_processes)

                instances = []
                for iterator in text_iterators:
                    job_instances = []
                    for text, utt, speaker in iterator:
                        try:
                            if emb_level == 'spk':
                                speaker_embedding = speaker_embeddings.get_embedding_for_identifier(speaker)
                            else:
                                speaker_embedding = speaker_embeddings.get_embedding_for_identifier(utt)

                            if prosody:
                                utt_prosody_dict = prosody.get_instance(utt)
                            else:
                                utt_prosody_dict = {}
                            job_instances.append((text, utt, speaker_embedding, utt_prosody_dict))
                        except KeyError:
                            logger.warn(f'Key error at {utt}')
                            continue
                    instances.append(job_instances)

                # multiprocessing
                with Pool(processes=num_processes) as pool:
                    job_params = zip(instances, self.tts_models, repeat(dataset_results_dir), sleeps,
                                     repeat(text_is_phones), repeat(self.save_output))
                    new_wavs = pool.starmap(tqdm(synthesis_job), job_params)

                for new_wav_dict in new_wavs:
                    wavs.update(new_wav_dict)
        return wavs


def synthesis_job(instances, tts_model, out_dir, sleep, text_is_phones=False, save_output=False):
    time.sleep(sleep)

    wavs = {}
    for text, utt, speaker_embedding, utt_prosody_dict in tqdm(instances):
        wav = tts_model.read_text(text=text, speaker_embedding=speaker_embedding, text_is_phones=text_is_phones,
                                  **utt_prosody_dict)

        if save_output:
            out_file = str((out_dir / f'{utt}.wav').absolute())
            soundfile.write(file=out_file, data=wav, samplerate=tts_model.output_sr)
            wavs[utt] = out_file
        else:
            wavs[utt] = wav
    return wavs


@torch.no_grad()
def synthesize_speech(
        device,
        texts,
        prosody,
        speaker_embeddings,
        tts_model,
        tts_samplerate,
        pros_samplerate,
        out_samplerate,
        result_dir,
):
    result_dir = Path(result_dir).absolute()
    if not result_dir.exists():
        logging.info(f'Creating directory {result_dir}')
        result_dir.mkdir(parents=True, exist_ok=True)

    tts_model = tts_model.to(device)
    tts_model.eval()

    logging.info(f'Synthesize {len(texts)} utterances...')
    wav_scp = list()
    for text, utt, speaker in tqdm(texts):
        speaker_embedding = speaker_embeddings.get_embedding_for_identifier(utt)
        utt_prosody_dict = prosody.get_instance(utt)
        duration = utt_prosody_dict['duration']
        pitch = utt_prosody_dict['pitch']
        energy = utt_prosody_dict['energy']
        speaker_embedding = speaker_embedding.to(device)
        tts_model.default_utterance_embedding = speaker_embedding
        samples = tts_model(
            text=text,
            text_is_phonemes=texts.is_phones,
            durations=duration,
            pitch=pitch,
            energy=energy,
        )
        i = 0
        while samples.shape[0] < 24000:  # 0.5 s
            # sometimes, the speaker embedding is so off that it leads to a practically empty audio
            # then, we need to sample a new embedding
            if i > 0 and i % 10 == 0:
                mask = torch.zeros(speaker_embedding.shape[0]).float().random_(-40, 40).to(device)
            else:
                mask = torch.zeros(speaker_embedding.shape[0]).float().random_(-2, 2).to(device)
            speaker_embedding = speaker_embedding * mask
            tts_model.default_utterance_embedding = speaker_embedding.to(device)
            samples = tts_model(
                text=text,
                text_is_phonemes=texts.is_phones,
                durations=duration,
                pitch=pitch,
                energy=energy,
            )
            i += 1
            if i > 30:
                break
        if i > 0:
            logging.info(f'Synthesized utt in {i} takes')
        samples = samples.detach().cpu().numpy()
        start_silence = round(utt_prosody_dict['start_silence'] * tts_samplerate / pros_samplerate)
        end_silence = round(utt_prosody_dict['end_silence'] * tts_samplerate / pros_samplerate)
        if start_silence is not None:
            start_sil = np.zeros(start_silence, dtype=np.float32)
            samples = np.hstack((start_sil, samples))
        if end_silence is not None:
            end_sil = np.zeros(end_silence, dtype=np.float32)
            samples = np.hstack((samples, end_sil))
        if out_samplerate != tts_samplerate:
            samples = resampy.resample(samples, tts_samplerate, out_samplerate)
        path = result_dir / f'{utt}.wav'
        sf.write(file=path, data=samples, samplerate=out_samplerate)
        wav_scp.append(f'{utt} {path}')

    path = result_dir / 'wav.scp'
    logging.info(f'Writing file {path}')
    path.write_text('\n'.join(wav_scp) + '\n', encoding='utf-8')
