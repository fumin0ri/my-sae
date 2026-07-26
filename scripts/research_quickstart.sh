#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-EleutherAI/pythia-70m-deduped}"
LAYERS="${LAYERS:-0,1,2,3,4,5}"
MAX_WINDOW="${MAX_WINDOW:-16}"
FIT_DEVICE="${FIT_DEVICE:-cuda}"

echo "[1/6] Generating grouped finite-state benchmark"
python scripts/make_research_data.py \
  --prompts-output data/research/prompts.jsonl \
  --pairs-output data/research/pairs.jsonl \
  --problems 128 \
  --paraphrases 4 \
  --pairs 32 \
  --seed 0

echo "[2/6] Extracting all requested layers in one model pass"
sr-extract-grid \
  --model "$MODEL" \
  --data data/research/prompts.jsonl \
  --output-dir runs/research/activations \
  --layers "$LAYERS" \
  --hook-point post \
  --window-size "$MAX_WINDOW" \
  --batch-size 16 \
  --max-length 256 \
  --dtype float32

echo "[3/6] Nested grouped model selection and locked testing"
sr-research \
  --activations-dir runs/research/activations \
  --output-dir runs/research/analysis \
  --window-sizes 4,8,10,16 \
  --ranks 2,4,8,16 \
  --ridges 0.001 \
  --seeds 0,1,2 \
  --group-key group_id \
  --label-key state \
  --permutations 100 \
  --device "$FIT_DEVICE"

SELECTED_LAYER="$(python -c 'import json; print(json.load(open("runs/research/analysis/selection.json"))["selected"]["layer"])')"

echo "[4/6] Causal patching at selected layer ${SELECTED_LAYER}"
sr-intervene \
  --model "$MODEL" \
  --pairs data/research/pairs.jsonl \
  --subspace runs/research/analysis/final_subspace.pt \
  --output runs/research/analysis/intervention-patch.jsonl \
  --layer "$SELECTED_LAYER" \
  --hook-point post \
  --mode patch

echo "[5/6] Learned and norm-matched random ablations"
sr-intervene \
  --model "$MODEL" \
  --pairs data/research/pairs.jsonl \
  --subspace runs/research/analysis/final_subspace.pt \
  --output runs/research/analysis/intervention-ablate.jsonl \
  --layer "$SELECTED_LAYER" \
  --hook-point post \
  --mode ablate
sr-intervene \
  --model "$MODEL" \
  --pairs data/research/pairs.jsonl \
  --subspace runs/research/analysis/final_subspace.pt \
  --output runs/research/analysis/intervention-random.jsonl \
  --layer "$SELECTED_LAYER" \
  --hook-point post \
  --mode random_ablate

echo "[6/6] Building publication figures and an HTML research report"
sr-visualize \
  --research-dir runs/research/analysis \
  --output-dir runs/research/report

echo "Open runs/research/report/index.html"
