#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Format: model@safetensors-revision|comma-separated-layers;...
MODEL_SPECS="${MODEL_SPECS:-EleutherAI/pythia-1.4b-deduped@dd0ec760c55304118fd0d0c98b3c6e3a4fa286af|5,11,17;EleutherAI/pythia-2.8b-deduped@04c6993bdebe728d5ad1dae3a916eaa766166783|7,15,23;EleutherAI/pythia-6.9b-deduped@d7e0e8080e3935fff58cb35d13fdaab0b2da9f30|7,15,23}"
TASK_FAMILIES="${TASK_FAMILIES:-fsm,arithmetic,logic}"
SEEDS="${SEEDS:-0,1,2}"
STEPS="${STEPS:-12000}"
PROBLEMS="${PROBLEMS:-300}"
WINDOW_SIZE="${WINDOW_SIZE:-64}"
CONTEXT_WIDTH="${CONTEXT_WIDTH:-32}"
TARGET_SIZES="${TARGET_SIZES:-2,4,8,16}"
GAPS="${GAPS:-2,4,8}"
EXPANSION="${EXPANSION:-8}"
K="${K:-64}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-2}"
PREDICTOR_WIDTH="${PREDICTOR_WIDTH:-256}"
PREDICTOR_HEADS="${PREDICTOR_HEADS:-8}"
PREDICTOR_LAYERS="${PREDICTOR_LAYERS:-3}"
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
RUN_CAUSAL="${RUN_CAUSAL:-0}"
STUDY_ROOT="${STUDY_ROOT:-runs/predictive-replication}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

python -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU is required"; assert torch.cuda.is_bf16_supported(), "BF16-capable GPU is required"; p=torch.cuda.get_device_properties(0); print(f"GPU: {p.name}, VRAM={p.total_memory/2**30:.1f} GiB, torch={torch.__version__}, CUDA={torch.version.cuda}")'
mkdir -p "$STUDY_ROOT"
git rev-parse HEAD > "$STUDY_ROOT/code-commit.txt"
python -m pip freeze > "$STUDY_ROOT/python-environment.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  > "$STUDY_ROOT/gpu-environment.csv"

IFS=';' read -r -a specifications <<< "$MODEL_SPECS"
IFS=',' read -r -a feature_seeds <<< "$SEEDS"
IFS=',' read -r -a task_families <<< "$TASK_FAMILIES"

for task_family in "${task_families[@]}"; do
  task_data_dir="data/replication/$task_family"
  python scripts/make_research_data.py \
    --prompts-output "$task_data_dir/prompts.jsonl" \
    --pairs-output "$task_data_dir/pairs.jsonl" \
    --problems "$PROBLEMS" \
    --paraphrases 4 \
    --pairs 64 \
    --task-family "$task_family" \
    --seed 0

  for specification in "${specifications[@]}"; do
    model_revision="${specification%%|*}"
    layers="${specification#*|}"
    model="${model_revision%@*}"
    revision="${model_revision##*@}"
    model_key="$(printf '%s' "$model" | tr '/:' '__')"
    activation_dir="$STUDY_ROOT/activations/$task_family/$model_key"

    echo "Extracting task=$task_family model=$model layers=$layers"
    sr-extract-grid \
      --model "$model" \
      --revision "$revision" \
      --use-safetensors \
      --data "$task_data_dir/prompts.jsonl" \
      --output-dir "$activation_dir" \
      --layers "$layers" \
      --hook-point post \
      --window-size "$WINDOW_SIZE" \
      --batch-size "$EXTRACT_BATCH_SIZE" \
      --max-length 384 \
      --dtype bfloat16 \
      --storage-dtype bfloat16

    IFS=',' read -r -a layer_indices <<< "$layers"
    for layer in "${layer_indices[@]}"; do
      activations="$activation_dir/layer-$(printf '%03d' "$layer").pt"
      d_in="$(python -c "import torch; print(torch.load('$activations', map_location='cpu', weights_only=False)['activations'].shape[-1])")"
      d_sae="$((d_in * EXPANSION))"

      for seed in "${feature_seeds[@]}"; do
        run_dir="$STUDY_ROOT/runs/$task_family/$model_key/layer-$(printf '%03d' "$layer")/seed-$seed"
        echo "Training task=$task_family model=$model layer=$layer seed=$seed (d_sae=$d_sae)"
        common=(
          --activations "$activations"
          --d-sae "$d_sae"
          --k "$K"
          --d-model "$PREDICTOR_WIDTH"
          --n-heads "$PREDICTOR_HEADS"
          --n-layers "$PREDICTOR_LAYERS"
          --context-width "$CONTEXT_WIDTH"
          --target-sizes "$TARGET_SIZES"
          --gaps "$GAPS"
          --context-mode causal
          --steps "$STEPS"
          --batch-size "$BATCH_SIZE"
          --gradient-accumulation-steps "$GRADIENT_ACCUMULATION"
          --amp-dtype bfloat16
          --lr 0.0002
          --warmup-steps 500
          --num-workers 2
          --log-every 500
          --device "$TRAIN_DEVICE"
          --seed "$seed"
          --split-seed 0
        )
        sr-train-predictive-sae \
          "${common[@]}" \
          --objective joint \
          --output-dir "$run_dir/joint"
        sr-train-predictive-sae \
          "${common[@]}" \
          --objective posthoc \
          --output-dir "$run_dir/posthoc"
        sr-evaluate-predictive-sae \
          --activations "$activations" \
          --joint-checkpoint "$run_dir/joint/predictive_sae.pt" \
          --baseline-checkpoint "$run_dir/posthoc/predictive_sae.pt" \
          --output-dir "$run_dir/analysis" \
          --batch-size "$EVAL_BATCH_SIZE" \
          --device "$TRAIN_DEVICE" \
          --seed "$seed"

        if [[ "$seed" == "${feature_seeds[0]}" ]]; then
          sr-fit \
            --activations "$activations" \
            --output-dir "$run_dir/low-rank-baseline" \
            --rank 8 \
            --ridge 0.001 \
            --permutations 200 \
            --label-key state \
            --group-key group_id \
            --device "$TRAIN_DEVICE"
        fi

        if [[ "$RUN_CAUSAL" == "1" ]]; then
          for mode in patch ablate random_ablate; do
            output_mode="$mode"
            if [[ "$mode" == "random_ablate" ]]; then
              output_mode="random"
            fi
            sr-intervene-predictive-sae \
              --model "$model" \
              --revision "$revision" \
              --use-safetensors \
              --pairs "$task_data_dir/pairs.jsonl" \
              --checkpoint "$run_dir/joint/predictive_sae.pt" \
              --output "$run_dir/analysis/intervention-$output_mode.jsonl" \
              --layer "$layer" \
              --hook-point post \
              --mode "$mode" \
              --target-size 4 \
              --gap 4
          done
        fi
        sr-visualize-predictive-sae --run-dir "$run_dir"
      done
    done
  done
done

sr-aggregate-predictive-study \
  --runs-root "$STUDY_ROOT/runs" \
  --output-dir "$STUDY_ROOT/summary"

echo "Replication report: $STUDY_ROOT/summary/index.html"
