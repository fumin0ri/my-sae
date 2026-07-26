#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-EleutherAI/pythia-70m-deduped}"
LAYER="${LAYER:-3}"
WINDOW_SIZE="${WINDOW_SIZE:-10}"
N_PROMPTS="${N_PROMPTS:-256}"
FIT_DEVICE="${FIT_DEVICE:-cuda}"

echo "[1/4] Generating ${N_PROMPTS} balanced prompts"
python scripts/make_quickstart_data.py \
  --output data/quickstart.jsonl \
  --n "$N_PROMPTS" \
  --seed 0

echo "[2/4] Extracting ${WINDOW_SIZE}-token residual windows from ${MODEL}"
sr-extract \
  --model "$MODEL" \
  --data data/quickstart.jsonl \
  --output runs/quickstart/activations.pt \
  --layer "$LAYER" \
  --hook-point post \
  --window-size "$WINDOW_SIZE" \
  --window-mode last \
  --batch-size 16 \
  --max-length 256 \
  --dtype float32

echo "[3/4] Fitting a held-out token-shared low-rank subspace"
sr-fit \
  --activations runs/quickstart/activations.pt \
  --output-dir runs/quickstart/shared \
  --rank 8 \
  --ridge 1e-3 \
  --train-fraction 0.8 \
  --permutations 100 \
  --label-key state \
  --device "$FIT_DEVICE" \
  --seed 0

echo "[4/4] Reporting held-out and control metrics"
python scripts/print_quickstart_report.py runs/quickstart/shared/report.json
