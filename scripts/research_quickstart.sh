#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# One-command RTX 4090 (24GB) profile. Pythia-6.9B is the largest suite member
# that leaves enough VRAM for the 8x predictive SAE during causal intervention.
MODEL="${MODEL:-EleutherAI/pythia-6.9b-deduped}"
default_safetensors_revision() {
  case "$1" in
    EleutherAI/pythia-70m-deduped)
      printf '%s' "c92722694b14c67e3f8ed3bf164bb2456449e720"
      ;;
    EleutherAI/pythia-1.4b-deduped)
      printf '%s' "dd0ec760c55304118fd0d0c98b3c6e3a4fa286af"
      ;;
    EleutherAI/pythia-2.8b-deduped)
      printf '%s' "04c6993bdebe728d5ad1dae3a916eaa766166783"
      ;;
    EleutherAI/pythia-6.9b-deduped)
      printf '%s' "d7e0e8080e3935fff58cb35d13fdaab0b2da9f30"
      ;;
  esac
}
REVISION="${REVISION:-$(default_safetensors_revision "$MODEL")}"
USE_SAFETENSORS="${USE_SAFETENSORS:-1}"
LAYER="${LAYER:-16}"
WINDOW_SIZE="${WINDOW_SIZE:-64}"
CONTEXT_WIDTH="${CONTEXT_WIDTH:-32}"
TARGET_SIZES="${TARGET_SIZES:-2,4,8,16}"
GAPS="${GAPS:-2,4,8}"
D_SAE="${D_SAE:-32768}"
K="${K:-64}"
PREDICTOR_WIDTH="${PREDICTOR_WIDTH:-256}"
PREDICTOR_HEADS="${PREDICTOR_HEADS:-8}"
PREDICTOR_LAYERS="${PREDICTOR_LAYERS:-3}"
STEPS="${STEPS:-12000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-2}"
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
PROBLEMS="${PROBLEMS:-256}"
PAIRS="${PAIRS:-64}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
RUN_CAUSAL="${RUN_CAUSAL:-1}"
RUN_DIR="${RUN_DIR:-runs/predictive-research}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export USE_SAFETENSORS

python -c 'import os, torch; from shared_residual.modeling import require_safe_torch_load; require_safe_torch_load(os.environ["USE_SAFETENSORS"] == "1"); assert torch.cuda.is_available(), "CUDA GPU is required"; assert torch.cuda.is_bf16_supported(), "BF16-capable GPU is required"; p=torch.cuda.get_device_properties(0); print(f"GPU: {p.name}, VRAM={p.total_memory/2**30:.1f} GiB, torch={torch.__version__}, CUDA={torch.version.cuda}")'
MODEL_LOAD_ARGS=(--model "$MODEL")
if [[ -n "$REVISION" ]]; then
  MODEL_LOAD_ARGS+=(--revision "$REVISION")
fi
if [[ "$USE_SAFETENSORS" == "1" ]]; then
  MODEL_LOAD_ARGS+=(--use-safetensors)
fi
mkdir -p "$RUN_DIR"
git rev-parse HEAD > "$RUN_DIR/code-commit.txt"
python -m pip freeze > "$RUN_DIR/python-environment.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  > "$RUN_DIR/gpu-environment.csv"

echo "[1/8] Generate paraphrase-grouped state-tracking data and independent causal pairs"
python scripts/make_research_data.py \
  --prompts-output data/research/prompts.jsonl \
  --pairs-output data/research/pairs.jsonl \
  --problems "$PROBLEMS" \
  --paraphrases 4 \
  --pairs "$PAIRS" \
  --seed 0

echo "[2/8] Extract frozen residual windows at the prespecified layer"
sr-extract-grid \
  "${MODEL_LOAD_ARGS[@]}" \
  --data data/research/prompts.jsonl \
  --output-dir "$RUN_DIR/activations" \
  --layers "$LAYER" \
  --hook-point post \
  --window-size "$WINDOW_SIZE" \
  --batch-size "$EXTRACT_BATCH_SIZE" \
  --max-length 384 \
  --dtype bfloat16 \
  --storage-dtype bfloat16

ACTIVATIONS="$RUN_DIR/activations/layer-$(printf '%03d' "$LAYER").pt"

echo "[3/8] Train the proposed JEPA-regularized SAE"
sr-train-predictive-sae \
  --activations "$ACTIVATIONS" \
  --output-dir "$RUN_DIR/joint" \
  --objective joint \
  --d-sae "$D_SAE" \
  --k "$K" \
  --d-model "$PREDICTOR_WIDTH" \
  --n-heads "$PREDICTOR_HEADS" \
  --n-layers "$PREDICTOR_LAYERS" \
  --context-width "$CONTEXT_WIDTH" \
  --target-sizes "$TARGET_SIZES" \
  --gaps "$GAPS" \
  --context-mode causal \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION" \
  --amp-dtype bfloat16 \
  --lr 0.0002 \
  --warmup-steps 500 \
  --num-workers 2 \
  --log-every 500 \
  --device "$TRAIN_DEVICE" \
  --seed 0

echo "[4/8] Train the standard-SAE plus frozen post-hoc-predictor control"
sr-train-predictive-sae \
  --activations "$ACTIVATIONS" \
  --output-dir "$RUN_DIR/posthoc" \
  --objective posthoc \
  --d-sae "$D_SAE" \
  --k "$K" \
  --d-model "$PREDICTOR_WIDTH" \
  --n-heads "$PREDICTOR_HEADS" \
  --n-layers "$PREDICTOR_LAYERS" \
  --context-width "$CONTEXT_WIDTH" \
  --target-sizes "$TARGET_SIZES" \
  --gaps "$GAPS" \
  --context-mode causal \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION" \
  --amp-dtype bfloat16 \
  --lr 0.0002 \
  --warmup-steps 500 \
  --num-workers 2 \
  --log-every 500 \
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
  --batch-size "$EVAL_BATCH_SIZE" \
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
    "${MODEL_LOAD_ARGS[@]}" \
    --pairs data/research/pairs.jsonl \
    --checkpoint "$RUN_DIR/joint/predictive_sae.pt" \
    --output "$RUN_DIR/analysis/intervention-patch.jsonl" \
    --layer "$LAYER" \
    --hook-point post \
    --mode patch \
    --target-size 4 \
    --gap 4
  sr-intervene-predictive-sae \
    "${MODEL_LOAD_ARGS[@]}" \
    --pairs data/research/pairs.jsonl \
    --checkpoint "$RUN_DIR/joint/predictive_sae.pt" \
    --output "$RUN_DIR/analysis/intervention-ablate.jsonl" \
    --layer "$LAYER" \
    --hook-point post \
    --mode ablate \
    --target-size 4 \
    --gap 4
  sr-intervene-predictive-sae \
    "${MODEL_LOAD_ARGS[@]}" \
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
