#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

COMPARISON_DIR="${RUN_DIR:-runs/predictor-auxk-comparison}"
SHARED_DIR="$COMPARISON_DIR/shared"
SOFTPLUS_DIR="$COMPARISON_DIR/softplus"
TOPK_DIR="$COMPARISON_DIR/relu_topk"
AUXK_DIR="$COMPARISON_DIR/relu_topk_auxk"
LAYER="${LAYER:-16}"
HORIZON_WEIGHTING="inverse_probability"

if [[ -n "${ACTIVATION_MANIFEST:-}" ]]; then
  SHARED_ACTIVATION_MANIFEST="$ACTIVATION_MANIFEST"
else
  SHARED_ACTIVATION_MANIFEST="$SHARED_DIR/pile-activations/manifest.json"
  if [[ ! -f "$SHARED_ACTIVATION_MANIFEST" ]]; then
    echo "[shared 1/2] Extract Pile residual sequences once"
    RUN_DIR="$SHARED_DIR" START_STAGE=1 END_STAGE=1 \
      HORIZON_WEIGHTING="$HORIZON_WEIGHTING" \
      PREDICTOR_OUTPUT=softplus PREDICTOR_AUXK_WEIGHT=0 \
      bash scripts/transition_jepa_quickstart.sh
  else
    echo "[shared 1/2] Reuse $SHARED_ACTIVATION_MANIFEST"
  fi
fi

SHARED_MMLU_PROMPTS="$SHARED_DIR/evaluation-data/mmlu-prompts.jsonl"
SHARED_CAUSAL_PAIRS="$SHARED_DIR/evaluation-data/mmlu-causal-pairs.jsonl"
SHARED_EVAL_ACTIVATIONS="$SHARED_DIR/activations/layer-$(printf '%03d' "$LAYER").pt"
SHARED_MMLU_RESULTS="$SHARED_DIR/analysis/mmlu_model_accuracy.json"

if [[ ! -f "$SHARED_MMLU_PROMPTS" || ! -f "$SHARED_CAUSAL_PAIRS" || \
      ! -f "$SHARED_EVAL_ACTIVATIONS" || ! -f "$SHARED_MMLU_RESULTS" ]]; then
  echo "[shared 2/2] Build and score the common locked MMLU evaluation once"
  RUN_DIR="$SHARED_DIR" START_STAGE=3 END_STAGE=5 \
    ACTIVATION_MANIFEST="$SHARED_ACTIVATION_MANIFEST" \
    HORIZON_WEIGHTING="$HORIZON_WEIGHTING" \
    PREDICTOR_OUTPUT=softplus PREDICTOR_AUXK_WEIGHT=0 \
    bash scripts/transition_jepa_quickstart.sh
else
  echo "[shared 2/2] Reuse common MMLU evaluation assets"
fi

run_condition() {
  local label="$1"
  local condition_dir="$2"
  local predictor_output="$3"
  local auxk_weight="$4"

  echo "[$label 1/2] Train: output=$predictor_output, AuxK=$auxk_weight"
  RUN_DIR="$condition_dir" START_STAGE=2 END_STAGE=2 \
    ACTIVATION_MANIFEST="$SHARED_ACTIVATION_MANIFEST" \
    HORIZON_WEIGHTING="$HORIZON_WEIGHTING" \
    PREDICTOR_OUTPUT="$predictor_output" \
    PREDICTOR_AUXK_WEIGHT="$auxk_weight" \
    bash scripts/transition_jepa_quickstart.sh

  echo "[$label 2/2] Evaluate and visualize with common locked data"
  RUN_DIR="$condition_dir" START_STAGE=6 END_STAGE=8 \
    ACTIVATION_MANIFEST="$SHARED_ACTIVATION_MANIFEST" \
    EVAL_ACTIVATIONS="$SHARED_EVAL_ACTIVATIONS" \
    MMLU_PROMPTS="$SHARED_MMLU_PROMPTS" \
    CAUSAL_PAIRS="$SHARED_CAUSAL_PAIRS" \
    MMLU_MODEL_RESULTS="$SHARED_MMLU_RESULTS" \
    HORIZON_WEIGHTING="$HORIZON_WEIGHTING" \
    PREDICTOR_OUTPUT="$predictor_output" \
    PREDICTOR_AUXK_WEIGHT="$auxk_weight" \
    bash scripts/transition_jepa_quickstart.sh
}

run_condition "condition 1/3" "$SOFTPLUS_DIR" softplus 0
run_condition "condition 2/3" "$TOPK_DIR" relu_topk 0
run_condition "condition 3/3" "$AUXK_DIR" relu_topk "${PREDICTOR_AUXK_WEIGHT:-0.03125}"

sr-compare-transition-predictors \
  --softplus-run "$SOFTPLUS_DIR" \
  --relu-topk-run "$TOPK_DIR" \
  --relu-topk-auxk-run "$AUXK_DIR" \
  --output-dir "$COMPARISON_DIR/comparison"

echo
echo "Done. Open: $COMPARISON_DIR/comparison/index.html"
