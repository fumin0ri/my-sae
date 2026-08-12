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

EVAL_EXTRACT_BATCH_SIZE="${EVAL_EXTRACT_BATCH_SIZE:-8}"
DEFAULT_EVAL_BATCH_SIZE=$((320 / WINDOW_SIZE))
if (( DEFAULT_EVAL_BATCH_SIZE < 2 )); then DEFAULT_EVAL_BATCH_SIZE=2; fi
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$DEFAULT_EVAL_BATCH_SIZE}"
EVAL_MAXIMUM_VALIDATION_BATCHES="${EVAL_MAXIMUM_VALIDATION_BATCHES:-0}"
PROBE_MAX_DIM="${PROBE_MAX_DIM:-1024}"
MMLU_MAX_QUESTIONS="${MMLU_MAX_QUESTIONS:-0}"
MMLU_DATASET="${MMLU_DATASET:-cais/mmlu}"
MMLU_DATASET_CONFIG="${MMLU_DATASET_CONFIG:-all}"
MMLU_DATASET_REVISION="${MMLU_DATASET_REVISION:-c30699e8356da336a370243923dbaf21066bb9fe}"
MMLU_MAX_LENGTH="${MMLU_MAX_LENGTH:-1536}"

LOSS_RECOVERED_INPUTS="${LOSS_RECOVERED_INPUTS:-32}"
LOSS_RECOVERED_CONTEXT_LENGTH="${LOSS_RECOVERED_CONTEXT_LENGTH:-2048}"
RUN_LOSS_RECOVERED="${RUN_LOSS_RECOVERED:-1}"
RUN_CAUSAL="${RUN_CAUSAL:-1}"
PAIRS="${PAIRS:-128}"
PAIR_POOL_SIZE="${PAIR_POOL_SIZE:-$((PAIRS * 16))}"
if (( MMLU_MAX_QUESTIONS > 0 && PAIR_POOL_SIZE > MMLU_MAX_QUESTIONS )); then
  PAIR_POOL_SIZE="$MMLU_MAX_QUESTIONS"
fi
if (( PAIR_POOL_SIZE < PAIRS )); then
  echo "PAIR_POOL_SIZE must be at least PAIRS" >&2
  exit 2
fi

SEED="${SEED:-0}"
SPLIT_SEED="${SPLIT_SEED:-0}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
RUN_DIR="${RUN_DIR:-runs/rectified-lpjepa-pile}"
START_STAGE="${START_STAGE:-1}"
END_STAGE="${END_STAGE:-8}"
if (( START_STAGE < 1 || START_STAGE > 8 || END_STAGE < 1 || END_STAGE > 8 || START_STAGE > END_STAGE )); then
  echo "Require 1 <= START_STAGE <= END_STAGE <= 8" >&2
  exit 2
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export USE_SAFETENSORS
python -c 'import os, torch; from shared_residual.modeling import require_safe_torch_load; require_safe_torch_load(os.environ["USE_SAFETENSORS"] == "1"); assert torch.cuda.is_available(), "CUDA GPU is required"; assert torch.cuda.is_bf16_supported(), "BF16-capable GPU is required"; p=torch.cuda.get_device_properties(0); print(f"GPU: {p.name}, VRAM={p.total_memory/2**30:.1f} GiB, torch={torch.__version__}, CUDA={torch.version.cuda}")'
echo "Config: span=$MIN_SPAN_LENGTH..$WINDOW_SIZE, D=$D_SAE, high_k=$HIGH_K, low_k=$LOW_K, high=$HIGH_FRACTION, RGG=p$RGG_P active=$TARGET_ACTIVE_FRACTION, inv=$INVARIANCE_WEIGHT, rdm=$RDM_WEIGHT/$RDM_PROJECTIONS projections, axis=$AXIS_RDM_WEIGHT/$AXIS_RDM_FEATURES features, pairs/sequence=$PAIRS_PER_SEQUENCE, per-batch cap=$MAX_PAIRS_PER_SEQUENCE_PER_BATCH, pair buffer=$PAIR_SHUFFLE_BUFFER_PAIRS, MMLU=$MMLU_MAX_QUESTIONS"

MODEL_LOAD_ARGS=(--model "$MODEL")
if [[ -n "$REVISION" ]]; then MODEL_LOAD_ARGS+=(--revision "$REVISION"); fi
if [[ "$USE_SAFETENSORS" == "1" ]]; then MODEL_LOAD_ARGS+=(--use-safetensors); fi

mkdir -p "$RUN_DIR"
git rev-parse HEAD > "$RUN_DIR/code-commit.txt"
python -m pip freeze > "$RUN_DIR/python-environment.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$RUN_DIR/gpu-environment.csv"

ACTIVATION_MANIFEST="${ACTIVATION_MANIFEST:-$RUN_DIR/pile-activations/manifest.json}"
CHECKPOINT="$RUN_DIR/model/rectified_lpjepa_sae.pt"
EVAL_ACTIVATIONS="${EVAL_ACTIVATIONS:-$RUN_DIR/activations/layer-$(printf '%03d' "$LAYER").pt}"
EVAL_ACTIVATION_DIR="${EVAL_ACTIVATION_DIR:-$(dirname "$EVAL_ACTIVATIONS")}"
MMLU_PROMPTS="${MMLU_PROMPTS:-$RUN_DIR/evaluation-data/mmlu-prompts.jsonl}"
CAUSAL_PAIRS="${CAUSAL_PAIRS:-$RUN_DIR/evaluation-data/mmlu-causal-pairs.jsonl}"
MMLU_MODEL_RESULTS="${MMLU_MODEL_RESULTS:-$RUN_DIR/analysis/mmlu_model_accuracy.json}"

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
  echo "[1/8] Extract document-disjoint long Pile residual sequences"
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
  echo "[2/8] Train the predictor-free high/low Rectified LpJEPA-SAE"
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
  echo "[3/8] Build balanced MMLU probes and causal pairs"
  mkdir -p "$(dirname "$MMLU_PROMPTS")" "$(dirname "$CAUSAL_PAIRS")"
  sr-make-mmlu \
    --prompts-output "$MMLU_PROMPTS" --pairs-output "$CAUSAL_PAIRS" \
    --dataset "$MMLU_DATASET" --dataset-config "$MMLU_DATASET_CONFIG" \
    --dataset-revision "$MMLU_DATASET_REVISION" \
    --max-questions "$MMLU_MAX_QUESTIONS" --pairs "$PAIR_POOL_SIZE" --seed "$SEED"
fi

if (( START_STAGE <= 4 && END_STAGE >= 4 )); then
  echo "[4/8] Extract MMLU residual trajectories"
  sr-extract-grid \
    "${MODEL_LOAD_ARGS[@]}" --data "$MMLU_PROMPTS" \
    --output-dir "$EVAL_ACTIVATION_DIR" --layers "$LAYER" \
    --hook-point post --window-size "$WINDOW_SIZE" \
    --batch-size "$EVAL_EXTRACT_BATCH_SIZE" --max-length "$MMLU_MAX_LENGTH" \
    --truncation-side left --dtype bfloat16 --storage-dtype bfloat16
fi

if (( START_STAGE <= 5 && END_STAGE >= 5 )); then
  echo "[5/8] Measure zero-shot base-model MMLU accuracy"
  mkdir -p "$(dirname "$MMLU_MODEL_RESULTS")"
  sr-score-mmlu \
    "${MODEL_LOAD_ARGS[@]}" --data "$MMLU_PROMPTS" --output "$MMLU_MODEL_RESULTS" \
    --batch-size "$EVAL_EXTRACT_BATCH_SIZE" --max-length "$MMLU_MAX_LENGTH" \
    --minimum-tokens "$WINDOW_SIZE" --dtype bfloat16
fi

if (( START_STAGE <= 6 && END_STAGE >= 6 )); then
  echo "[6/8] Evaluate SAE quality, RDMReg, view invariance, swaps, and MMLU probes"
  EVAL_ARGS=(
    --activation-manifest "$ACTIVATION_MANIFEST"
    --activations "$EVAL_ACTIVATIONS"
    --checkpoint "$CHECKPOINT"
    --mmlu-model-results "$MMLU_MODEL_RESULTS"
    --output-dir "$RUN_DIR/analysis"
    --group-key question_id
    --probe-max-dim "$PROBE_MAX_DIM"
    --batch-size "$EVAL_BATCH_SIZE"
    --maximum-validation-batches "$EVAL_MAXIMUM_VALIDATION_BATCHES"
    --device "$TRAIN_DEVICE" --amp-dtype bfloat16
    --seed "$SEED" --split-seed "$SPLIT_SEED"
    "${MODEL_LOAD_ARGS[@]}"
    --layer "$LAYER" --hook-point post
    --loss-recovered-inputs "$LOSS_RECOVERED_INPUTS"
    --loss-recovered-context-length "$LOSS_RECOVERED_CONTEXT_LENGTH"
    --dtype bfloat16
  )
  if [[ "$RUN_LOSS_RECOVERED" != "1" ]]; then EVAL_ARGS+=(--skip-loss-recovered); fi
  sr-evaluate-rectified-lpjepa-sae "${EVAL_ARGS[@]}"
fi

if (( START_STAGE <= 7 && END_STAGE >= 7 )); then
  if [[ "$RUN_CAUSAL" == "1" ]]; then
    echo "[7/8] Patch, ablate, and norm-match high features"
    for MODE in patch ablate random_ablate; do
      OUTPUT_MODE="$MODE"
      if [[ "$MODE" == "random_ablate" ]]; then OUTPUT_MODE="random"; fi
      sr-intervene-rectified-lpjepa-sae \
        "${MODEL_LOAD_ARGS[@]}" --pairs "$CAUSAL_PAIRS" \
        --checkpoint "$CHECKPOINT" \
        --output "$RUN_DIR/analysis/intervention-$OUTPUT_MODE.jsonl" \
        --layer "$LAYER" --hook-point post --mode "$MODE" \
        --max-pairs "$PAIRS" --minimum-pairs "$PAIRS" --seed "$SEED"
    done
  else
    echo "[7/8] Causal interventions skipped (RUN_CAUSAL=$RUN_CAUSAL)"
  fi
fi

if (( START_STAGE <= 8 && END_STAGE >= 8 )); then
  echo "[8/8] Build SAE, invariance, RDMReg, probe, and causal figures"
  sr-visualize-rectified-lpjepa-sae --run-dir "$RUN_DIR"
fi

echo
if (( END_STAGE == 8 )); then
  echo "Done. Open: $RUN_DIR/report/index.html"
else
  echo "Done. Completed stages $START_STAGE through $END_STAGE."
fi
