#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# One-command, leakage-aware confirmatory experiment. Every value can be
# overridden from the environment without editing this file.
MODEL="${MODEL:-EleutherAI/pythia-70m-deduped}"
LAYER="${LAYER:-3}"
WINDOW_SIZE="${WINDOW_SIZE:-48}"
CONTEXT_WIDTH="${CONTEXT_WIDTH:-24}"
TARGET_SIZES="${TARGET_SIZES:-2,4,8}"
GAPS="${GAPS:-2,4,8}"
D_SAE="${D_SAE:-2048}"
K="${K:-32}"
STEPS="${STEPS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
RUN_CAUSAL="${RUN_CAUSAL:-1}"
RUN_DIR="${RUN_DIR:-runs/predictive-research}"

echo "[1/8] Generate paraphrase-grouped state-tracking data and independent causal pairs"
python scripts/make_research_data.py \
  --prompts-output data/research/prompts.jsonl \
  --pairs-output data/research/pairs.jsonl \
  --problems 128 \
  --paraphrases 4 \
  --pairs 32 \
  --seed 0

echo "[2/8] Extract frozen residual windows at the prespecified layer"
sr-extract-grid \
  --model "$MODEL" \
  --data data/research/prompts.jsonl \
  --output-dir "$RUN_DIR/activations" \
  --layers "$LAYER" \
  --hook-point post \
  --window-size "$WINDOW_SIZE" \
  --batch-size 16 \
  --max-length 256 \
  --dtype float32

ACTIVATIONS="$RUN_DIR/activations/layer-$(printf '%03d' "$LAYER").pt"

echo "[3/8] Train the proposed JEPA-regularized SAE"
sr-train-predictive-sae \
  --activations "$ACTIVATIONS" \
  --output-dir "$RUN_DIR/joint" \
  --objective joint \
  --d-sae "$D_SAE" \
  --k "$K" \
  --context-width "$CONTEXT_WIDTH" \
  --target-sizes "$TARGET_SIZES" \
  --gaps "$GAPS" \
  --context-mode causal \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --device "$TRAIN_DEVICE" \
  --seed 0

echo "[4/8] Train the standard-SAE plus frozen post-hoc-predictor control"
sr-train-predictive-sae \
  --activations "$ACTIVATIONS" \
  --output-dir "$RUN_DIR/posthoc" \
  --objective posthoc \
  --d-sae "$D_SAE" \
  --k "$K" \
  --context-width "$CONTEXT_WIDTH" \
  --target-sizes "$TARGET_SIZES" \
  --gaps "$GAPS" \
  --context-mode causal \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --device "$TRAIN_DEVICE" \
  --seed 0

echo "[5/8] Open the locked problem-group test exactly once"
sr-evaluate-predictive-sae \
  --activations "$ACTIVATIONS" \
  --joint-checkpoint "$RUN_DIR/joint/predictive_sae.pt" \
  --baseline-checkpoint "$RUN_DIR/posthoc/predictive_sae.pt" \
  --output-dir "$RUN_DIR/analysis" \
  --label-key state \
  --group-key group_id \
  --batch-size 64 \
  --device "$TRAIN_DEVICE" \
  --seed 0

echo "[6/8] Fit the original low-rank random-effects method as a retained baseline"
sr-fit \
  --activations "$ACTIVATIONS" \
  --output-dir "$RUN_DIR/low-rank-baseline" \
  --rank 8 \
  --ridge 0.001 \
  --permutations 100 \
  --label-key state \
  --group-key group_id \
  --device "$TRAIN_DEVICE"

if [[ "$RUN_CAUSAL" == "1" ]]; then
  echo "[7/8] Patch, ablate, and norm-match random directions in the original LLM"
  sr-intervene-predictive-sae \
    --model "$MODEL" \
    --pairs data/research/pairs.jsonl \
    --checkpoint "$RUN_DIR/joint/predictive_sae.pt" \
    --output "$RUN_DIR/analysis/intervention-patch.jsonl" \
    --layer "$LAYER" \
    --hook-point post \
    --mode patch \
    --target-size 4 \
    --gap 4
  sr-intervene-predictive-sae \
    --model "$MODEL" \
    --pairs data/research/pairs.jsonl \
    --checkpoint "$RUN_DIR/joint/predictive_sae.pt" \
    --output "$RUN_DIR/analysis/intervention-ablate.jsonl" \
    --layer "$LAYER" \
    --hook-point post \
    --mode ablate \
    --target-size 4 \
    --gap 4
  sr-intervene-predictive-sae \
    --model "$MODEL" \
    --pairs data/research/pairs.jsonl \
    --checkpoint "$RUN_DIR/joint/predictive_sae.pt" \
    --output "$RUN_DIR/analysis/intervention-random.jsonl" \
    --layer "$LAYER" \
    --hook-point post \
    --mode random_ablate \
    --target-size 4 \
    --gap 4
else
  echo "[7/8] Causal interventions skipped (RUN_CAUSAL=$RUN_CAUSAL)"
fi

echo "[8/8] Build publication PNG/PDF figures and a self-contained HTML report"
sr-visualize-predictive-sae \
  --run-dir "$RUN_DIR"

echo
echo "Done. Open: $RUN_DIR/report/index.html"
