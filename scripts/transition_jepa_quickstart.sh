#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# High/low endpoint JEPA-SAE on one RTX 4090, CUDA 12.1 / torch 2.5.1.
MODEL="${MODEL:-EleutherAI/pythia-6.9b-deduped}"
default_safetensors_revision() {
  case "$1" in
    EleutherAI/pythia-70m-deduped) printf '%s' "c92722694b14c67e3f8ed3bf164bb2456449e720" ;;
    EleutherAI/pythia-1.4b-deduped) printf '%s' "dd0ec760c55304118fd0d0c98b3c6e3a4fa286af" ;;
    EleutherAI/pythia-2.8b-deduped) printf '%s' "04c6993bdebe728d5ad1dae3a916eaa766166783" ;;
    EleutherAI/pythia-6.9b-deduped) printf '%s' "d7e0e8080e3935fff58cb35d13fdaab0b2da9f30" ;;
  esac
}
REVISION="${REVISION:-$(default_safetensors_revision "$MODEL")}"
USE_SAFETENSORS="${USE_SAFETENSORS:-1}"
LAYER="${LAYER:-16}"
# T-SAE evaluates 128-position activation sequences; retain configurability.
WINDOW_SIZE="${WINDOW_SIZE:-128}"
if (( WINDOW_SIZE < 2 )); then
  echo "WINDOW_SIZE must be at least 2" >&2
  exit 2
fi

D_SAE="${D_SAE:-32768}"
K="${K:-64}"
HIGH_FRACTION="${HIGH_FRACTION:-0.2}"
HIGH_RECONSTRUCTION_WEIGHT="${HIGH_RECONSTRUCTION_WEIGHT:-0.2}"
PREDICTOR_WIDTH="${PREDICTOR_WIDTH:-512}"
TRAIN_STEPS="${TRAIN_STEPS:-12000}"
SAE_WARMUP_STEPS="${SAE_WARMUP_STEPS:-4000}"
PREDICTION_RAMP_STEPS="${PREDICTION_RAMP_STEPS:-1000}"

DEFAULT_WINDOW_BATCH_SIZE=$((320 / WINDOW_SIZE))
if (( DEFAULT_WINDOW_BATCH_SIZE < 1 )); then DEFAULT_WINDOW_BATCH_SIZE=1; fi
BATCH_SIZE="${BATCH_SIZE:-$DEFAULT_WINDOW_BATCH_SIZE}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-2}"
PILE_EXTRACT_BATCH_SIZE="${PILE_EXTRACT_BATCH_SIZE:-8}"
DEFAULT_WINDOWS_PER_SEQUENCE=$((320 / WINDOW_SIZE))
if (( DEFAULT_WINDOWS_PER_SEQUENCE < 1 )); then DEFAULT_WINDOWS_PER_SEQUENCE=1; fi
PILE_SEQUENCE_LENGTH="${PILE_SEQUENCE_LENGTH:-$((WINDOW_SIZE * DEFAULT_WINDOWS_PER_SEQUENCE))}"
if (( PILE_SEQUENCE_LENGTH % WINDOW_SIZE != 0 )); then
  echo "PILE_SEQUENCE_LENGTH must be divisible by WINDOW_SIZE" >&2
  exit 2
fi

PILE_TRAIN_POSITIONS="${PILE_TRAIN_POSITIONS:-5242880}"
PILE_VALIDATION_POSITIONS="${PILE_VALIDATION_POSITIONS:-163840}"
PILE_SHARD_POSITIONS="${PILE_SHARD_POSITIONS:-40960}"
PILE_TRAIN_WINDOWS="${PILE_TRAIN_WINDOWS:-}"
PILE_VALIDATION_WINDOWS="${PILE_VALIDATION_WINDOWS:-}"
PILE_SHARD_WINDOWS="${PILE_SHARD_WINDOWS:-}"
PILE_DISK_RESERVE_GIB="${PILE_DISK_RESERVE_GIB:-5}"
PILE_SKIP_DISK_SPACE_CHECK="${PILE_SKIP_DISK_SPACE_CHECK:-0}"
PILE_DATASET="${PILE_DATASET:-EleutherAI/the_pile_deduplicated}"
PILE_DATASET_CONFIG="${PILE_DATASET_CONFIG:-default}"
PILE_DATASET_REVISION="${PILE_DATASET_REVISION:-fcbfcfde4222cbb1acd1d33bad0be250ee14b1bb}"
PILE_DATASET_TRUST_REMOTE_CODE="${PILE_DATASET_TRUST_REMOTE_CODE:-0}"
PILE_REQUIRE_ALL_DOMAINS="${PILE_REQUIRE_ALL_DOMAINS:-0}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$DEFAULT_WINDOW_BATCH_SIZE}"
EVAL_MAXIMUM_BATCHES="${EVAL_MAXIMUM_BATCHES:-0}"
EVAL_DATASET="${EVAL_DATASET:-monology/pile-uncopyrighted}"
EVAL_DATASET_CONFIG="${EVAL_DATASET_CONFIG:-}"
LOSS_RECOVERED_INPUTS="${LOSS_RECOVERED_INPUTS:-32}"
LOSS_RECOVERED_CONTEXT_LENGTH="${LOSS_RECOVERED_CONTEXT_LENGTH:-2048}"
RUN_LOSS_RECOVERED="${RUN_LOSS_RECOVERED:-1}"

SEED="${SEED:-0}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
RUN_DIR="${RUN_DIR:-runs/high-low-jepa-pile}"
START_STAGE="${START_STAGE:-1}"
END_STAGE="${END_STAGE:-4}"
if (( START_STAGE < 1 || START_STAGE > 4 || END_STAGE < 1 || END_STAGE > 4 )); then
  echo "START_STAGE and END_STAGE must lie in [1, 4]" >&2
  exit 2
fi
if (( START_STAGE > END_STAGE )); then
  echo "START_STAGE cannot exceed END_STAGE" >&2
  exit 2
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export USE_SAFETENSORS
python -c 'import os, torch; from shared_residual.modeling import require_safe_torch_load; require_safe_torch_load(os.environ["USE_SAFETENSORS"] == "1"); assert torch.cuda.is_available(), "CUDA GPU is required"; assert torch.cuda.is_bf16_supported(), "BF16-capable GPU is required"; p=torch.cuda.get_device_properties(0); print(f"GPU: {p.name}, VRAM={p.total_memory/2**30:.1f} GiB, torch={torch.__version__}, CUDA={torch.version.cuda}")'
echo "High/low config: W=$WINDOW_SIZE, D=$D_SAE, K=$K, high=$HIGH_FRACTION, high-reconstruction-weight=$HIGH_RECONSTRUCTION_WEIGHT"

MODEL_LOAD_ARGS=(--model "$MODEL")
if [[ -n "$REVISION" ]]; then MODEL_LOAD_ARGS+=(--revision "$REVISION"); fi
if [[ "$USE_SAFETENSORS" == "1" ]]; then MODEL_LOAD_ARGS+=(--use-safetensors); fi

mkdir -p "$RUN_DIR"
git rev-parse HEAD > "$RUN_DIR/code-commit.txt"
python -m pip freeze > "$RUN_DIR/python-environment.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$RUN_DIR/gpu-environment.csv"

ACTIVATION_MANIFEST="$RUN_DIR/pile-activations/manifest.json"
CHECKPOINT="$RUN_DIR/model/transition_jepa_sae.pt"

PILE_DATA_ARGS=(--dataset "$PILE_DATASET" --dataset-config "$PILE_DATASET_CONFIG")
if [[ -n "$PILE_DATASET_REVISION" ]]; then PILE_DATA_ARGS+=(--dataset-revision "$PILE_DATASET_REVISION"); fi
if [[ "$PILE_DATASET_TRUST_REMOTE_CODE" == "1" ]]; then PILE_DATA_ARGS+=(--dataset-trust-remote-code); fi
if [[ "$PILE_REQUIRE_ALL_DOMAINS" == "1" ]]; then PILE_DATA_ARGS+=(--require-all-domains); fi
PILE_BUDGET_ARGS=(
  --train-positions "$PILE_TRAIN_POSITIONS"
  --validation-positions "$PILE_VALIDATION_POSITIONS"
  --shard-positions "$PILE_SHARD_POSITIONS"
  --disk-reserve-gib "$PILE_DISK_RESERVE_GIB"
)
if [[ -n "$PILE_TRAIN_WINDOWS" ]]; then PILE_BUDGET_ARGS+=(--train-windows "$PILE_TRAIN_WINDOWS"); fi
if [[ -n "$PILE_VALIDATION_WINDOWS" ]]; then PILE_BUDGET_ARGS+=(--validation-windows "$PILE_VALIDATION_WINDOWS"); fi
if [[ -n "$PILE_SHARD_WINDOWS" ]]; then PILE_BUDGET_ARGS+=(--shard-windows "$PILE_SHARD_WINDOWS"); fi
if [[ "$PILE_SKIP_DISK_SPACE_CHECK" == "1" ]]; then PILE_BUDGET_ARGS+=(--skip-disk-space-check); fi

if (( START_STAGE <= 1 && END_STAGE >= 1 )); then
  echo "[1/4] Extract document-disjoint Pile residual windows"
  sr-extract-pile \
    "${MODEL_LOAD_ARGS[@]}" \
    "${PILE_DATA_ARGS[@]}" \
    --output-dir "$RUN_DIR/pile-activations" \
    --layer "$LAYER" \
    --hook-point post \
    --window-size "$WINDOW_SIZE" \
    --sequence-length "$PILE_SEQUENCE_LENGTH" \
    "${PILE_BUDGET_ARGS[@]}" \
    --batch-size "$PILE_EXTRACT_BATCH_SIZE" \
    --dtype bfloat16 \
    --seed "$SEED"
fi

if (( START_STAGE <= 2 && END_STAGE >= 2 )); then
  echo "[2/4] Train the only model: high/low full-EMA endpoint JEPA-SAE"
  sr-train-transition-jepa-sae \
    --activation-manifest "$ACTIVATION_MANIFEST" \
    --output-dir "$RUN_DIR/model" \
    --d-sae "$D_SAE" \
    --k "$K" \
    --high-fraction "$HIGH_FRACTION" \
    --high-reconstruction-weight "$HIGH_RECONSTRUCTION_WEIGHT" \
    --predictor-width "$PREDICTOR_WIDTH" \
    --steps "$TRAIN_STEPS" \
    --sae-warmup-steps "$SAE_WARMUP_STEPS" \
    --prediction-ramp-steps "$PREDICTION_RAMP_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION" \
    --amp-dtype bfloat16 \
    --predictor-lr 0.0003 \
    --sae-lr 0.0002 \
    --warmup-steps 500 \
    --log-every 400 \
    --device "$TRAIN_DEVICE" \
    --seed "$SEED"
fi

if (( START_STAGE <= 3 && END_STAGE >= 3 )); then
  echo "[3/4] Evaluate with AI4LIFE temporal-saes metrics"
  EVAL_ARGS=(
    --activation-manifest "$ACTIVATION_MANIFEST"
    --checkpoint "$CHECKPOINT"
    --output-dir "$RUN_DIR/analysis"
    --batch-size "$EVAL_BATCH_SIZE"
    --maximum-batches "$EVAL_MAXIMUM_BATCHES"
    --device "$TRAIN_DEVICE"
    --amp-dtype bfloat16
    "${MODEL_LOAD_ARGS[@]}"
    --layer "$LAYER"
    --hook-point post
    --eval-dataset "$EVAL_DATASET"
    --loss-recovered-inputs "$LOSS_RECOVERED_INPUTS"
    --loss-recovered-context-length "$LOSS_RECOVERED_CONTEXT_LENGTH"
    --dtype bfloat16
  )
  if [[ -n "$EVAL_DATASET_CONFIG" ]]; then EVAL_ARGS+=(--eval-dataset-config "$EVAL_DATASET_CONFIG"); fi
  if [[ "$RUN_LOSS_RECOVERED" != "1" ]]; then EVAL_ARGS+=(--skip-loss-recovered); fi
  sr-evaluate-transition-jepa-sae "${EVAL_ARGS[@]}"
fi

if (( START_STAGE <= 4 && END_STAGE >= 4 )); then
  echo "[4/4] Build T-SAE metric PNG/PDF figures and HTML report"
  sr-visualize-transition-jepa-sae --run-dir "$RUN_DIR"
fi

echo
if (( END_STAGE == 4 )); then
  echo "Done. Open: $RUN_DIR/report/index.html"
else
  echo "Done. Completed stages $START_STAGE through $END_STAGE."
fi
