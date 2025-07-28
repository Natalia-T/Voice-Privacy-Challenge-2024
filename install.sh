#!/bin/bash

set -euo pipefail

uv venv || exit 1

uv pip install -U pip || exit 1

uv pip install \
  torch==2.4.1 \
  torchvision==0.19.1 \
  torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu124 || exit 1

uv pip install \
  matplotlib \
  noisereduce \
  speechbrain==0.5.16 || exit 1

uv sync --inexact || exit 1

echo Done
