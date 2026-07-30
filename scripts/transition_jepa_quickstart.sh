#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Fixed-endpoint JEPA-SAE on one RTX 4090 (24GB), CUDA 12.1 compatible.
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
WINDOW_SIZE="${WINDOW_SIZE:-10}"
if (( WINDOW_SIZE < 2 )); then
  echo "WINDOW_SIZE must be at least 2" >&2
  exit 2
fi
D_SAE="${D_SAE:-32768}"
K="${K:-64}"
HIGH_FRACTION="${HIGH_FRACTION:-0.2}"
HIGH_RECONSTRUCTION_WEIGHT="${HIGH_RECONSTRUCTION_WEIGHT:-0.2}"
PREDICTOR_WIDTH="${PREDICTOR_WIDTH:-512}"
STANDARD_STEPS="${STANDARD_STEPS:-12000}"
FORECAST_STEPS="${FORECAST_STEPS:-8000}"
PREDICTOR_WARMUP_STEPS="${PREDICTOR_WARMUP_STEPS:-1000}"
PREDICTION_RAMP_STEPS="${PREDICTION_RAMP_STEPS:-1000}"
INTERVENTION_HORIZON="${INTERVENTION_HORIZON:-$((WINDOW_SIZE - 1))}"
if (( INTERVENTION_HORIZON < 1 || INTERVENTION_HORIZON >= WINDOW_SIZE )); then
  echo "INTERVENTION_HORIZON must lie in [1, WINDOW_SIZE-1]" >&2
  exit 2
fi
DEFAULT_WINDOW_BATCH_SIZE=$((320 / WINDOW_SIZE))
if (( DEFAULT_WINDOW_BATCH_SIZE < 1 )); then
  DEFAULT_WINDOW_BATCH_SIZE=1
fi
BATCH_SIZE="${BATCH_SIZE:-$DEFAULT_WINDOW_BATCH_SIZE}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-2}"
PILE_EXTRACT_BATCH_SIZE="${PILE_EXTRACT_BATCH_SIZE:-8}"
DEFAULT_WINDOWS_PER_SEQUENCE=$((320 / WINDOW_SIZE))
if (( DEFAULT_WINDOWS_PER_SEQUENCE < 1 )); then
  DEFAULT_WINDOWS_PER_SEQUENCE=1
fi
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
EVAL_EXTRACT_BATCH_SIZE="${EVAL_EXTRACT_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$DEFAULT_WINDOW_BATCH_SIZE}"
MMLU_MAX_QUESTIONS="${MMLU_MAX_QUESTIONS:-0}"
MMLU_DATASET="${MMLU_DATASET:-cais/mmlu}"
MMLU_DATASET_CONFIG="${MMLU_DATASET_CONFIG:-all}"
MMLU_DATASET_REVISION="${MMLU_DATASET_REVISION:-c30699e8356da336a370243923dbaf21066bb9fe}"
MMLU_MAX_LENGTH="${MMLU_MAX_LENGTH:-1536}"
PAIRS="${PAIRS:-128}"
PAIR_POOL_SIZE="${PAIR_POOL_SIZE:-}"
if [[ -z "$PAIR_POOL_SIZE" ]]; then
  PAIR_POOL_SIZE=$((PAIRS * 16))
  if (( MMLU_MAX_QUESTIONS > 0 && PAIR_POOL_SIZE > MMLU_MAX_QUESTIONS )); then
    PAIR_POOL_SIZE="$MMLU_MAX_QUESTIONS"
  fi
fi
if (( PAIR_POOL_SIZE < PAIRS )); then
  echo "PAIR_POOL_SIZE must be at least PAIRS" >&2
  exit 2
fi
SEED="${SEED:-0}"
SPLIT_SEED="${SPLIT_SEED:-0}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
RUN_CAUSAL="${RUN_CAUSAL:-1}"
RUN_DIR="${RUN_DIR:-runs/transition-jepa-pile}"
START_STAGE="${START_STAGE:-1}"
END_STAGE="${END_STAGE:-12}"
if (( START_STAGE < 1 || START_STAGE > 12 || END_STAGE < 1 || END_STAGE > 12 )); then
  echo "START_STAGE and END_STAGE must lie in [1, 12]" >&2
  exit 2
fi
if (( START_STAGE > END_STAGE )); then
  echo "START_STAGE cannot exceed END_STAGE" >&2
  exit 2
fi
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export USE_SAFETENSORS

python -c 'import os, torch; from shared_residual.modeling import require_safe_torch_load; require_safe_torch_load(os.environ["USE_SAFETENSORS"] == "1"); assert torch.cuda.is_available(), "CUDA GPU is required"; assert torch.cuda.is_bf16_supported(), "BF16-capable GPU is required"; p=torch.cuda.get_device_properties(0); print(f"GPU: {p.name}, VRAM={p.total_memory/2**30:.1f} GiB, torch={torch.__version__}, CUDA={torch.version.cuda}")'
echo "Window config: W=$WINDOW_SIZE, train batch=$BATCH_SIZE windows, evaluation batch=$EVAL_BATCH_SIZE windows"
echo "Hierarchical config: high fraction=$HIGH_FRACTION, high-only reconstruction weight=$HIGH_RECONSTRUCTION_WEIGHT (total D_SAE=$D_SAE, total K=$K)"
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

ACTIVATION_MANIFEST="$RUN_DIR/pile-activations/manifest.json"
EVAL_ACTIVATIONS="$RUN_DIR/activations/layer-$(printf '%03d' "$LAYER").pt"
STANDARD_CHECKPOINT="$RUN_DIR/standard/standard_sae.pt"

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
PILE_BUDGET_ARGS=(
  --train-positions "$PILE_TRAIN_POSITIONS"
  --validation-positions "$PILE_VALIDATION_POSITIONS"
  --shard-positions "$PILE_SHARD_POSITIONS"
  --disk-reserve-gib "$PILE_DISK_RESERVE_GIB"
)
if [[ -n "$PILE_TRAIN_WINDOWS" ]]; then
  PILE_BUDGET_ARGS+=(--train-windows "$PILE_TRAIN_WINDOWS")
fi
if [[ -n "$PILE_VALIDATION_WINDOWS" ]]; then
  PILE_BUDGET_ARGS+=(--validation-windows "$PILE_VALIDATION_WINDOWS")
fi
if [[ -n "$PILE_SHARD_WINDOWS" ]]; then
  PILE_BUDGET_ARGS+=(--shard-windows "$PILE_SHARD_WINDOWS")
fi
if [[ "$PILE_SKIP_DISK_SPACE_CHECK" == "1" ]]; then
  PILE_BUDGET_ARGS+=(--skip-disk-space-check)
fi
if (( START_STAGE <= 1 && END_STAGE >= 1 )); then
  echo "[1/12] Stream the official Pile mixture and extract residual shards"
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
  echo "[2/12] Build the balanced MMLU locked-test benchmark"
  sr-make-mmlu \
    --prompts-output data/transition-jepa/prompts.jsonl \
    --pairs-output data/transition-jepa/pairs.jsonl \
    --dataset "$MMLU_DATASET" \
    --dataset-config "$MMLU_DATASET_CONFIG" \
    --dataset-revision "$MMLU_DATASET_REVISION" \
    --max-questions "$MMLU_MAX_QUESTIONS" \
    --pairs "$PAIR_POOL_SIZE" \
    --seed "$SEED"
fi

if (( START_STAGE <= 3 && END_STAGE >= 3 )); then
  echo "[3/12] Extract MMLU residual trajectories for evaluation only"
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
fi

if (( START_STAGE <= 4 && END_STAGE >= 4 )); then
  echo "[4/12] Measure zero-shot base-LLM MMLU answer accuracy"
  sr-score-mmlu \
    "${MODEL_LOAD_ARGS[@]}" \
    --data data/transition-jepa/prompts.jsonl \
    --output "$RUN_DIR/analysis/mmlu_model_accuracy.json" \
    --batch-size "$EVAL_EXTRACT_BATCH_SIZE" \
    --max-length "$MMLU_MAX_LENGTH" \
    --minimum-tokens "$WINDOW_SIZE" \
    --dtype bfloat16
fi

if (( START_STAGE <= 5 && END_STAGE >= 5 )); then
  echo "[5/12] Pretrain the common all-position standard SAE on The Pile"
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
fi

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

if (( START_STAGE <= 6 && END_STAGE >= 6 )); then
  echo "[6/12] Train the existing unsplit JEPA-SAE baseline"
  sr-train-transition-jepa-sae \
    "${COMMON_FORECAST_ARGS[@]}" \
    --output-dir "$RUN_DIR/joint" \
    --objective joint
fi

if (( START_STAGE <= 7 && END_STAGE >= 7 )); then
  echo "[7/12] Train the T-SAE-inspired high/low JEPA-SAE"
  sr-train-transition-jepa-sae \
    "${COMMON_FORECAST_ARGS[@]}" \
    --output-dir "$RUN_DIR/hierarchical" \
    --objective joint \
    --architecture hierarchical \
    --high-fraction "$HIGH_FRACTION" \
    --high-reconstruction-weight "$HIGH_RECONSTRUCTION_WEIGHT"
fi

if (( START_STAGE <= 8 && END_STAGE >= 8 )); then
  echo "[8/12] Train the endpoint predictor on the frozen standard SAE"
  sr-train-transition-jepa-sae \
    "${COMMON_FORECAST_ARGS[@]}" \
    --output-dir "$RUN_DIR/fixed" \
    --objective fixed
fi

if (( START_STAGE <= 9 && END_STAGE >= 9 )); then
  echo "[9/12] Train the position-only shortcut control"
  sr-train-transition-jepa-sae \
    "${COMMON_FORECAST_ARGS[@]}" \
    --output-dir "$RUN_DIR/k_only" \
    --objective k_only
fi

if (( START_STAGE <= 10 && END_STAGE >= 10 )); then
  echo "[10/12] Compare unsplit, high/low, fixed, and position-only models"
  sr-evaluate-transition-jepa-sae \
    --activations "$EVAL_ACTIVATIONS" \
    --joint-checkpoint "$RUN_DIR/joint/transition_jepa_sae.pt" \
    --hierarchical-checkpoint "$RUN_DIR/hierarchical/transition_jepa_sae.pt" \
    --fixed-checkpoint "$RUN_DIR/fixed/transition_jepa_sae.pt" \
    --k-only-checkpoint "$RUN_DIR/k_only/transition_jepa_sae.pt" \
    --mmlu-model-results "$RUN_DIR/analysis/mmlu_model_accuracy.json" \
    --output-dir "$RUN_DIR/analysis" \
    --group-key question_id \
    --batch-size "$EVAL_BATCH_SIZE" \
    --device "$TRAIN_DEVICE" \
    --seed "$SEED" \
    --split-seed "$SPLIT_SEED"
fi

if (( START_STAGE <= 11 && END_STAGE >= 11 )); then
  if [[ "$RUN_CAUSAL" == "1" ]]; then
    echo "[11/12] Compare causal edits through unsplit and high-only EMA decoders"
    for METHOD in joint hierarchical; do
      for MODE in patch ablate random_ablate; do
        OUTPUT_MODE="$MODE"
        if [[ "$MODE" == "random_ablate" ]]; then
          OUTPUT_MODE="random"
        fi
        sr-intervene-transition-jepa-sae \
          "${MODEL_LOAD_ARGS[@]}" \
          --pairs data/transition-jepa/pairs.jsonl \
          --checkpoint "$RUN_DIR/$METHOD/transition_jepa_sae.pt" \
          --output "$RUN_DIR/analysis/intervention-$METHOD-$OUTPUT_MODE.jsonl" \
          --layer "$LAYER" \
          --hook-point post \
          --mode "$MODE" \
          --horizon "$INTERVENTION_HORIZON" \
          --max-pairs "$PAIRS" \
          --minimum-pairs "$PAIRS" \
          --seed "$SEED"
      done
    done
  else
    echo "[11/12] Causal interventions skipped (RUN_CAUSAL=$RUN_CAUSAL)"
  fi
fi

if (( START_STAGE <= 12 && END_STAGE >= 12 )); then
  echo "[12/12] Build PNG/PDF figures and a self-contained HTML report"
  sr-visualize-transition-jepa-sae --run-dir "$RUN_DIR"
fi

echo
if (( END_STAGE == 12 )); then
  echo "Done. Open: $RUN_DIR/report/index.html"
else
  echo "Done. Completed stages $START_STAGE through $END_STAGE."
fi
