#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
python -m pip install --upgrade-strategy only-if-needed \
  --constraint constraints/cuda121.txt -e ".[saebench]"
python -m pip check

python - <<'PY'
import torch

print(f"torch={torch.__version__}, built CUDA={torch.version.cuda}")
if torch.__version__ != "2.5.1+cu121" or torch.version.cuda != "12.1":
    raise RuntimeError("expected torch 2.5.1+cu121 after installation")
print("CUDA available:", torch.cuda.is_available())
PY
