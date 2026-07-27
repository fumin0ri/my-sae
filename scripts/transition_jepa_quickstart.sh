#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Offset-conditioned JEPA-SAE on one RTX 4090 (24GB), CUDA 12.1 compatible.
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
WINDOW_SIZE=10
D_SAE="${D_SAE:-32768}"
K="${K:-64}"
PREDICTOR_WIDTH="${PREDICTOR_WIDTH:-512}"
STANDARD_STEPS="${STANDARD_STEPS:-12000}"
FORECAST_STEPS="${FORECAST_STEPS:-8000}"
PREDICTOR_WARMUP_STEPS="${PREDICTOR_WARMUP_STEPS:-1000}"
PREDICTION_RAMP_STEPS="${PREDICTION_RAMP_STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-2}"
PILE_EXTRACT_BATCH_SIZE="${PILE_EXTRACT_BATCH_SIZE:-8}"
PILE_SEQUENCE_LENGTH="${PILE_SEQUENCE_LENGTH:-320}"
PILE_TRAIN_WINDOWS="${PILE_TRAIN_WINDOWS:-524288}"
PILE_VALIDATION_WINDOWS="${PILE_VALIDATION_WINDOWS:-16384}"
PILE_SHARD_WINDOWS="${PILE_SHARD_WINDOWS:-4096}"
PILE_DATASET="${PILE_DATASET:-EleutherAI/the_pile_deduplicated}"
PILE_DATASET_CONFIG="${PILE_DATASET_CONFIG:-default}"
PILE_DATASET_REVISION="${PILE_DATASET_REVISION:-fcbfcfde4222cbb1acd1d33bad0be250ee14b1bb}"
PILE_DATASET_TRUST_REMOTE_CODE="${PILE_DATASET_TRUST_REMOTE_CODE:-0}"
PILE_REQUIRE_ALL_DOMAINS="${PILE_REQUIRE_ALL_DOMAINS:-0}"
EVAL_EXTRACT_BATCH_SIZE="${EVAL_EXTRACT_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
MMLU_MAX_QUESTIONS="${MMLU_MAX_QUESTIONS:-0}"
MMLU_DATASET="${MMLU_DATASET:-cais/mmlu}"
MMLU_DATASET_CONFIG="${MMLU_DATASET_CONFIG:-all}"
MMLU_DATASET_REVISION="${MMLU_DATASET_REVISION:-c30699e8356da336a370243923dbaf21066bb9fe}"
MMLU_MAX_LENGTH="${MMLU_MAX_LENGTH:-1536}"
PAIRS="${PAIRS:-128}"
SEED="${SEED:-0}"
SPLIT_SEED="${SPLIT_SEED:-0}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
RUN_CAUSAL="${RUN_CAUSAL:-1}"
RUN_DIR="${RUN_DIR:-runs/transition-jepa-pile}"
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

echo "[1/11] Stream the official Pile mixture and extract residual shards"
PILE_DATA_ARGS=(--dataset "$PILE_DATASET" --dataset-config "$PILE_DATASET_CONFIG")
if [[ -n "$PILE_DATASET_REVISION" ]]; then
  PILE_DATA_ARGS+=(--dataset-revision "$PILE_DATASET_REVISION")
fi
if [[ "$PILE_DATASET_TRUST_REMOTE_CODE" == "1" ]]; then
  PILE_DATA_ARGS+=(--dataset-trust-remote-code)
fi
if [[ "$PILE_REQUIRE_ALL_DOMAINS" == "1" ]]; then
  PILE_DATA_ARGS+=(--require-all-domains)
fi
sr-extract-pile \
  "${MODEL_LOAD_ARGS[@]}" \
  "${PILE_DATA_ARGS[@]}" \
  --output-dir "$RUN_DIR/pile-activations" \
  --layer "$LAYER" \
  --hook-point post \
  --window-size "$WINDOW_SIZE" \
  --sequence-length "$PILE_SEQUENCE_LENGTH" \
  --train-windows "$PILE_TRAIN_WINDOWS" \
  --validation-windows "$PILE_VALIDATION_WINDOWS" \
  --shard-windows "$PILE_SHARD_WINDOWS" \
  --batch-size "$PILE_EXTRACT_BATCH_SIZE" \
  --dtype bfloat16 \
  --seed "$SEED"

ACTIVATION_MANIFEST="$RUN_DIR/pile-activations/manifest.json"

echo "[2/11] Build the balanced MMLU locked-test benchmark"
sr-make-mmlu \
  --prompts-output data/transition-jepa/prompts.jsonl \
  --pairs-output data/transition-jepa/pairs.jsonl \
  --dataset "$MMLU_DATASET" \
  --dataset-config "$MMLU_DATASET_CONFIG" \
  --dataset-revision "$MMLU_DATASET_REVISION" \
  --max-questions "$MMLU_MAX_QUESTIONS" \
  --pairs "$PAIRS" \
  --seed "$SEED"

echo "[3/11] Extract MMLU residual trajectories for evaluation only"
sr-extract-grid \
  "${MODEL_LOAD_ARGS[@]}" \
  --data data/transition-jepa/prompts.jsonl \
  --output-dir "$RUN_DIR/activations" \
  --layers "$LAYER" \
  --hook-point post \
  --window-size "$WINDOW_SIZE" \
  --batch-size "$EVAL_EXTRACT_BATCH_SIZE" \
  --max-length "$MMLU_MAX_LENGTH" \
  --truncation-side left \
  --dtype bfloat16 \
  --storage-dtype bfloat16

EVAL_ACTIVATIONS="$RUN_DIR/activations/layer-$(printf '%03d' "$LAYER").pt"

echo "[4/11] Measure zero-shot base-LLM MMLU answer accuracy"
sr-score-mmlu \
  "${MODEL_LOAD_ARGS[@]}" \
  --data data/transition-jepa/prompts.jsonl \
  --output "$RUN_DIR/analysis/mmlu_model_accuracy.json" \
  --batch-size "$EVAL_EXTRACT_BATCH_SIZE" \
  --max-length "$MMLU_MAX_LENGTH" \
  --dtype bfloat16

echo "[5/11] Pretrain the common all-position standard SAE on The Pile"
sr-train-standard-sae \
  --activation-manifest "$ACTIVATION_MANIFEST" \
  --output-dir "$RUN_DIR/standard" \
  --d-sae "$D_SAE" \
  --k "$K" \
  --steps "$STANDARD_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION" \
  --amp-dtype bfloat16 \
  --lr 0.0002 \
  --warmup-steps 500 \
  --log-every 500 \
  --device "$TRAIN_DEVICE" \
  --seed "$SEED"

STANDARD_CHECKPOINT="$RUN_DIR/standard/standard_sae.pt"
COMMON_FORECAST_ARGS=(
  --activation-manifest "$ACTIVATION_MANIFEST"
  --init-checkpoint "$STANDARD_CHECKPOINT"
  --d-sae "$D_SAE"
  --k "$K"
  --predictor-width "$PREDICTOR_WIDTH"
  --steps "$FORECAST_STEPS"
  --predictor-warmup-steps "$PREDICTOR_WARMUP_STEPS"
  --prediction-ramp-steps "$PREDICTION_RAMP_STEPS"
  --batch-size "$BATCH_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION"
  --amp-dtype bfloat16
  --predictor-lr 0.0003
  --sae-lr 0.0001
  --warmup-steps 300
  --log-every 400
  --device "$TRAIN_DEVICE"
  --seed "$SEED"
)

echo "[6/11] Jointly tune the SAE dictionary and predictor on The Pile"
sr-train-transition-jepa-sae \
  "${COMMON_FORECAST_ARGS[@]}" \
  --output-dir "$RUN_DIR/joint" \
  --objective joint

echo "[7/11] Train a predictor on the frozen standard-SAE dictionary"
sr-train-transition-jepa-sae \
  "${COMMON_FORECAST_ARGS[@]}" \
  --output-dir "$RUN_DIR/fixed" \
  --objective fixed

echo "[8/11] Train the offset-only shortcut control with z0 removed"
sr-train-transition-jepa-sae \
  "${COMMON_FORECAST_ARGS[@]}" \
  --output-dir "$RUN_DIR/k_only" \
  --objective k_only

echo "[9/11] Open the question-grouped MMLU locked test"
sr-evaluate-transition-jepa-sae \
  --activations "$EVAL_ACTIVATIONS" \
  --joint-checkpoint "$RUN_DIR/joint/transition_jepa_sae.pt" \
  --fixed-checkpoint "$RUN_DIR/fixed/transition_jepa_sae.pt" \
  --k-only-checkpoint "$RUN_DIR/k_only/transition_jepa_sae.pt" \
  --mmlu-model-results "$RUN_DIR/analysis/mmlu_model_accuracy.json" \
  --output-dir "$RUN_DIR/analysis" \
  --group-key question_id \
  --batch-size "$EVAL_BATCH_SIZE" \
  --device "$TRAIN_DEVICE" \
  --seed "$SEED" \
  --split-seed "$SPLIT_SEED"

if [[ "$RUN_CAUSAL" == "1" ]]; then
  echo "[10/11] Patch, ablate, and norm-match MMLU forecastable features"
  sr-intervene-transition-jepa-sae \
    "${MODEL_LOAD_ARGS[@]}" \
    --pairs data/transition-jepa/pairs.jsonl \
    --checkpoint "$RUN_DIR/joint/transition_jepa_sae.pt" \
    --output "$RUN_DIR/analysis/intervention-patch.jsonl" \
    --layer "$LAYER" \
    --hook-point post \
    --mode patch \
    --seed "$SEED"
  sr-intervene-transition-jepa-sae \
    "${MODEL_LOAD_ARGS[@]}" \
    --pairs data/transition-jepa/pairs.jsonl \
    --checkpoint "$RUN_DIR/joint/transition_jepa_sae.pt" \
    --output "$RUN_DIR/analysis/intervention-ablate.jsonl" \
    --layer "$LAYER" \
    --hook-point post \
    --mode ablate \
    --seed "$SEED"
  sr-intervene-transition-jepa-sae \
    "${MODEL_LOAD_ARGS[@]}" \
    --pairs data/transition-jepa/pairs.jsonl \
    --checkpoint "$RUN_DIR/joint/transition_jepa_sae.pt" \
    --output "$RUN_DIR/analysis/intervention-random.jsonl" \
    --layer "$LAYER" \
    --hook-point post \
    --mode random_ablate \
    --seed "$SEED"
else
  echo "[10/11] Causal interventions skipped (RUN_CAUSAL=$RUN_CAUSAL)"
fi

echo "[11/11] Build PNG/PDF figures and a self-contained HTML report"
sr-visualize-transition-jepa-sae --run-dir "$RUN_DIR"

echo
echo "Done. Open: $RUN_DIR/report/index.html"
