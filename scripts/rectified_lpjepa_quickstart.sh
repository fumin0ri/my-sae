#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Predictor-free high/low Rectified LpJEPA-SAE on one RTX 4090,
# CUDA 12.1 / torch 2.5.1.
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
WINDOW_SIZE="${WINDOW_SIZE:-32}"
MIN_SPAN_LENGTH="${MIN_SPAN_LENGTH:-2}"
if (( WINDOW_SIZE < 2 || MIN_SPAN_LENGTH < 2 || MIN_SPAN_LENGTH > WINDOW_SIZE )); then
  echo "Require 2 <= MIN_SPAN_LENGTH <= WINDOW_SIZE" >&2
  exit 2
fi
BURN_IN_TOKENS="${BURN_IN_TOKENS:-$WINDOW_SIZE}"

D_SAE="${D_SAE:-32768}"
HIGH_K="${HIGH_K:-128}"
LOW_K="${LOW_K:-64}"
HIGH_FRACTION="${HIGH_FRACTION:-0.2}"
HIGH_RECONSTRUCTION_WEIGHT="${HIGH_RECONSTRUCTION_WEIGHT:-0.1}"
RGG_P="${RGG_P:-1}"
TARGET_ACTIVE_FRACTION="${TARGET_ACTIVE_FRACTION:-0.025}"
TARGET_SIGMA="${TARGET_SIGMA:-0}"
INVARIANCE_WEIGHT="${INVARIANCE_WEIGHT:-1}"
RDM_WEIGHT="${RDM_WEIGHT:-5}"
RDM_PROJECTIONS="${RDM_PROJECTIONS:-1024}"
RDM_PROJECTION_CHUNK_SIZE="${RDM_PROJECTION_CHUNK_SIZE:-128}"
AXIS_RDM_FEATURES="${AXIS_RDM_FEATURES:-512}"
AXIS_RDM_WEIGHT="${AXIS_RDM_WEIGHT:-1}"

TRAIN_STEPS="${TRAIN_STEPS:-12000}"
REGULARIZATION_RAMP_STEPS="${REGULARIZATION_RAMP_STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-160}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-2}"
PAIRS_PER_SEQUENCE="${PAIRS_PER_SEQUENCE:-8}"
MAX_PAIRS_PER_SEQUENCE_PER_BATCH="${MAX_PAIRS_PER_SEQUENCE_PER_BATCH:-2}"
PAIR_SHUFFLE_BUFFER_PAIRS="${PAIR_SHUFFLE_BUFFER_PAIRS:-4096}"
PILE_EXTRACT_BATCH_SIZE="${PILE_EXTRACT_BATCH_SIZE:-8}"
PILE_SEQUENCE_LENGTH="${PILE_SEQUENCE_LENGTH:-320}"
if (( PILE_SEQUENCE_LENGTH < BURN_IN_TOKENS + WINDOW_SIZE )); then
  echo "PILE_SEQUENCE_LENGTH must be at least BURN_IN_TOKENS + WINDOW_SIZE" >&2
  exit 2
fi

PILE_TRAIN_POSITIONS="${PILE_TRAIN_POSITIONS:-5242880}"
PILE_VALIDATION_POSITIONS="${PILE_VALIDATION_POSITIONS:-163840}"
PILE_SHARD_POSITIONS="${PILE_SHARD_POSITIONS:-40960}"
PILE_TRAIN_SEQUENCES="${PILE_TRAIN_SEQUENCES:-}"
PILE_VALIDATION_SEQUENCES="${PILE_VALIDATION_SEQUENCES:-}"
PILE_SHARD_SEQUENCES="${PILE_SHARD_SEQUENCES:-}"
PILE_DISK_RESERVE_GIB="${PILE_DISK_RESERVE_GIB:-5}"
PILE_SKIP_DISK_SPACE_CHECK="${PILE_SKIP_DISK_SPACE_CHECK:-0}"
PILE_DATASET="${PILE_DATASET:-EleutherAI/the_pile_deduplicated}"
PILE_DATASET_CONFIG="${PILE_DATASET_CONFIG:-default}"
PILE_DATASET_REVISION="${PILE_DATASET_REVISION:-fcbfcfde4222cbb1acd1d33bad0be250ee14b1bb}"
PILE_DATASET_TRUST_REMOTE_CODE="${PILE_DATASET_TRUST_REMOTE_CODE:-0}"
PILE_REQUIRE_ALL_DOMAINS="${PILE_REQUIRE_ALL_DOMAINS:-0}"

DEFAULT_EVAL_BATCH_SIZE=$((320 / WINDOW_SIZE))
if (( DEFAULT_EVAL_BATCH_SIZE < 2 )); then DEFAULT_EVAL_BATCH_SIZE=2; fi
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$DEFAULT_EVAL_BATCH_SIZE}"
EVAL_MAXIMUM_VALIDATION_BATCHES="${EVAL_MAXIMUM_VALIDATION_BATCHES:-0}"

LOSS_RECOVERED_INPUTS="${LOSS_RECOVERED_INPUTS:-32}"
LOSS_RECOVERED_CONTEXT_LENGTH="${LOSS_RECOVERED_CONTEXT_LENGTH:-2048}"
RUN_LOSS_RECOVERED="${RUN_LOSS_RECOVERED:-1}"
SAEBENCH_EVALS="${SAEBENCH_EVALS:-core}"
SAEBENCH_COMPONENTS="${SAEBENCH_COMPONENTS:-full}"
SAEBENCH_CONTEXT_SIZE="${SAEBENCH_CONTEXT_SIZE:-128}"
SAEBENCH_LLM_BATCH_SIZE="${SAEBENCH_LLM_BATCH_SIZE:-1}"
SAEBENCH_SAE_BATCH_SIZE="${SAEBENCH_SAE_BATCH_SIZE:-64}"
SAEBENCH_CORE_RECONSTRUCTION_BATCHES="${SAEBENCH_CORE_RECONSTRUCTION_BATCHES:-200}"
SAEBENCH_CORE_SPARSITY_BATCHES="${SAEBENCH_CORE_SPARSITY_BATCHES:-2000}"
SAEBENCH_CORE_DATASET="${SAEBENCH_CORE_DATASET:-Skylion007/openwebtext}"
SAEBENCH_COMPUTE_WEIGHT_METRICS="${SAEBENCH_COMPUTE_WEIGHT_METRICS:-0}"
SAEBENCH_SAVE_ACTIVATIONS="${SAEBENCH_SAVE_ACTIVATIONS:-0}"
SAEBENCH_FORCE_RERUN="${SAEBENCH_FORCE_RERUN:-0}"

SEED="${SEED:-0}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
RUN_DIR="${RUN_DIR:-runs/rectified-lpjepa-pile}"
START_STAGE="${START_STAGE:-1}"
END_STAGE="${END_STAGE:-5}"
if (( START_STAGE < 1 || START_STAGE > 5 || END_STAGE < 1 || END_STAGE > 5 || START_STAGE > END_STAGE )); then
  echo "Require 1 <= START_STAGE <= END_STAGE <= 5" >&2
  exit 2
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export USE_SAFETENSORS
python - <<'PY'
import os
import torch

from shared_residual.modeling import require_safe_torch_load

require_safe_torch_load(os.environ["USE_SAFETENSORS"] == "1")
print(f"PyTorch: {torch.__version__}, built CUDA={torch.version.cuda}")
if torch.__version__ != "2.5.1+cu121" or torch.version.cuda != "12.1":
    raise RuntimeError(
        "This pipeline requires torch 2.5.1+cu121. A dependency probably "
        "replaced it. Repair the environment with: "
        "bash scripts/install_cuda121.sh"
    )
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is unavailable even with the expected PyTorch build. Check "
        "nvidia-smi and the driver, then rerun scripts/install_cuda121.sh."
    )
if not torch.cuda.is_bf16_supported():
    raise RuntimeError("a BF16-capable CUDA GPU is required")
p = torch.cuda.get_device_properties(0)
print(f"GPU: {p.name}, VRAM={p.total_memory / 2**30:.1f} GiB")
PY
echo "Config: span=$MIN_SPAN_LENGTH..$WINDOW_SIZE, D=$D_SAE, high_k=$HIGH_K, low_k=$LOW_K, high=$HIGH_FRACTION, RGG=p$RGG_P active=$TARGET_ACTIVE_FRACTION, inv=$INVARIANCE_WEIGHT, rdm=$RDM_WEIGHT/$RDM_PROJECTIONS projections, axis=$AXIS_RDM_WEIGHT/$AXIS_RDM_FEATURES features, pairs/sequence=$PAIRS_PER_SEQUENCE, per-batch cap=$MAX_PAIRS_PER_SEQUENCE_PER_BATCH, pair buffer=$PAIR_SHUFFLE_BUFFER_PAIRS, SAEBench=$SAEBENCH_EVALS/$SAEBENCH_COMPONENTS"

MODEL_LOAD_ARGS=(--model "$MODEL")
if [[ -n "$REVISION" ]]; then MODEL_LOAD_ARGS+=(--revision "$REVISION"); fi
if [[ "$USE_SAFETENSORS" == "1" ]]; then MODEL_LOAD_ARGS+=(--use-safetensors); fi

mkdir -p "$RUN_DIR"
git rev-parse HEAD > "$RUN_DIR/code-commit.txt"
python -m pip freeze > "$RUN_DIR/python-environment.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$RUN_DIR/gpu-environment.csv"

ACTIVATION_MANIFEST="${ACTIVATION_MANIFEST:-$RUN_DIR/pile-activations/manifest.json}"
CHECKPOINT="$RUN_DIR/model/rectified_lpjepa_sae.pt"

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
if [[ -n "$PILE_TRAIN_SEQUENCES" ]]; then PILE_BUDGET_ARGS+=(--train-sequences "$PILE_TRAIN_SEQUENCES"); fi
if [[ -n "$PILE_VALIDATION_SEQUENCES" ]]; then PILE_BUDGET_ARGS+=(--validation-sequences "$PILE_VALIDATION_SEQUENCES"); fi
if [[ -n "$PILE_SHARD_SEQUENCES" ]]; then PILE_BUDGET_ARGS+=(--shard-sequences "$PILE_SHARD_SEQUENCES"); fi
if [[ "$PILE_SKIP_DISK_SPACE_CHECK" == "1" ]]; then PILE_BUDGET_ARGS+=(--skip-disk-space-check); fi

if (( START_STAGE <= 1 && END_STAGE >= 1 )); then
  echo "[1/5] Extract document-disjoint long Pile residual sequences"
  sr-extract-pile \
    "${MODEL_LOAD_ARGS[@]}" "${PILE_DATA_ARGS[@]}" \
    --output-dir "$RUN_DIR/pile-activations" \
    --layer "$LAYER" --hook-point post \
    --min-span-length "$MIN_SPAN_LENGTH" --max-span-length "$WINDOW_SIZE" \
    --sequence-length "$PILE_SEQUENCE_LENGTH" --burn-in-tokens "$BURN_IN_TOKENS" \
    "${PILE_BUDGET_ARGS[@]}" \
    --batch-size "$PILE_EXTRACT_BATCH_SIZE" --dtype bfloat16 --seed "$SEED"
fi

if (( START_STAGE <= 2 && END_STAGE >= 2 )); then
  echo "[2/5] Train the predictor-free high/low Rectified LpJEPA-SAE"
  sr-train-rectified-lpjepa-sae \
    --activation-manifest "$ACTIVATION_MANIFEST" --output-dir "$RUN_DIR/model" \
    --d-sae "$D_SAE" --high-k "$HIGH_K" --low-k "$LOW_K" \
    --high-fraction "$HIGH_FRACTION" \
    --high-reconstruction-weight "$HIGH_RECONSTRUCTION_WEIGHT" \
    --rgg-p "$RGG_P" --target-active-fraction "$TARGET_ACTIVE_FRACTION" \
    --target-sigma "$TARGET_SIGMA" \
    --invariance-weight "$INVARIANCE_WEIGHT" --rdm-weight "$RDM_WEIGHT" \
    --rdm-projections "$RDM_PROJECTIONS" \
    --rdm-projection-chunk-size "$RDM_PROJECTION_CHUNK_SIZE" \
    --axis-rdm-features "$AXIS_RDM_FEATURES" \
    --axis-rdm-weight "$AXIS_RDM_WEIGHT" \
    --steps "$TRAIN_STEPS" \
    --regularization-ramp-steps "$REGULARIZATION_RAMP_STEPS" \
    --batch-size "$BATCH_SIZE" --gradient-accumulation-steps "$GRADIENT_ACCUMULATION" \
    --pairs-per-sequence "$PAIRS_PER_SEQUENCE" \
    --max-pairs-per-sequence-per-batch "$MAX_PAIRS_PER_SEQUENCE_PER_BATCH" \
    --pair-shuffle-buffer-pairs "$PAIR_SHUFFLE_BUFFER_PAIRS" \
    --amp-dtype bfloat16 --sae-lr 0.0002 --warmup-steps 500 \
    --log-every 400 --device "$TRAIN_DEVICE" --seed "$SEED"
fi

if (( START_STAGE <= 3 && END_STAGE >= 3 )); then
  echo "[3/5] Evaluate method-specific SAE quality, RDMReg, invariance, and swaps"
  EVAL_ARGS=(
    --activation-manifest "$ACTIVATION_MANIFEST"
    --checkpoint "$CHECKPOINT"
    --output-dir "$RUN_DIR/analysis"
    --batch-size "$EVAL_BATCH_SIZE"
    --maximum-validation-batches "$EVAL_MAXIMUM_VALIDATION_BATCHES"
    --device "$TRAIN_DEVICE" --amp-dtype bfloat16
    --seed "$SEED"
    "${MODEL_LOAD_ARGS[@]}"
    --layer "$LAYER" --hook-point post
    --loss-recovered-inputs "$LOSS_RECOVERED_INPUTS"
    --loss-recovered-context-length "$LOSS_RECOVERED_CONTEXT_LENGTH"
    --dtype bfloat16
  )
  if [[ "$RUN_LOSS_RECOVERED" != "1" ]]; then EVAL_ARGS+=(--skip-loss-recovered); fi
  sr-evaluate-rectified-lpjepa-sae "${EVAL_ARGS[@]}"
fi

if (( START_STAGE <= 4 && END_STAGE >= 4 )); then
  echo "[4/5] Run SAEBench on the trained Top-K SAE"
  SAEBENCH_ARGS=(
    --checkpoint "$CHECKPOINT"
    --output-dir "$RUN_DIR/saebench"
    --evals "$SAEBENCH_EVALS"
    --components "$SAEBENCH_COMPONENTS"
    --device "$TRAIN_DEVICE" --dtype bfloat16
    --context-size "$SAEBENCH_CONTEXT_SIZE"
    --llm-batch-size "$SAEBENCH_LLM_BATCH_SIZE"
    --sae-batch-size "$SAEBENCH_SAE_BATCH_SIZE"
    --core-reconstruction-batches "$SAEBENCH_CORE_RECONSTRUCTION_BATCHES"
    --core-sparsity-batches "$SAEBENCH_CORE_SPARSITY_BATCHES"
    --core-dataset "$SAEBENCH_CORE_DATASET"
    --seed "$SEED"
  )
  if [[ -n "$REVISION" ]]; then SAEBENCH_ARGS+=(--model-revision "$REVISION"); fi
  if [[ "$SAEBENCH_COMPUTE_WEIGHT_METRICS" == "1" ]]; then SAEBENCH_ARGS+=(--compute-weight-metrics); fi
  if [[ "$SAEBENCH_SAVE_ACTIVATIONS" == "1" ]]; then SAEBENCH_ARGS+=(--save-activations); fi
  if [[ "$SAEBENCH_FORCE_RERUN" == "1" ]]; then SAEBENCH_ARGS+=(--force-rerun); fi
  sr-evaluate-saebench "${SAEBENCH_ARGS[@]}"
fi

if (( START_STAGE <= 5 && END_STAGE >= 5 )); then
  echo "[5/5] Build SAE, invariance, RDMReg, and SAEBench figures"
  sr-visualize-rectified-lpjepa-sae --run-dir "$RUN_DIR"
fi

echo
if (( END_STAGE == 5 )); then
  echo "Done. Open: $RUN_DIR/report/index.html"
else
  echo "Done. Completed stages $START_STAGE through $END_STAGE."
fi
