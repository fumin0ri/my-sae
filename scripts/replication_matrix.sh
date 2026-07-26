#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Format: model|comma-separated-layers;model|comma-separated-layers
MODEL_SPECS="${MODEL_SPECS:-EleutherAI/pythia-70m-deduped|1,3,5;EleutherAI/pythia-160m-deduped|3,7,11;EleutherAI/pythia-410m-deduped|5,11,17}"
TASK_FAMILIES="${TASK_FAMILIES:-fsm,arithmetic,logic}"
SEEDS="${SEEDS:-0,1,2}"
STEPS="${STEPS:-10000}"
PROBLEMS="${PROBLEMS:-300}"
WINDOW_SIZE="${WINDOW_SIZE:-64}"
CONTEXT_WIDTH="${CONTEXT_WIDTH:-32}"
TARGET_SIZES="${TARGET_SIZES:-2,4,8}"
GAPS="${GAPS:-2,4,8}"
EXPANSION="${EXPANSION:-8}"
K="${K:-64}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
RUN_CAUSAL="${RUN_CAUSAL:-0}"
STUDY_ROOT="${STUDY_ROOT:-runs/predictive-replication}"

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
    model="${specification%%|*}"
    layers="${specification#*|}"
    model_key="$(printf '%s' "$model" | tr '/:' '__')"
    activation_dir="$STUDY_ROOT/activations/$task_family/$model_key"

    echo "Extracting task=$task_family model=$model layers=$layers"
    sr-extract-grid \
      --model "$model" \
      --data "$task_data_dir/prompts.jsonl" \
      --output-dir "$activation_dir" \
      --layers "$layers" \
      --hook-point post \
      --window-size "$WINDOW_SIZE" \
      --batch-size 8 \
      --max-length 384 \
      --dtype float32

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
          --context-width "$CONTEXT_WIDTH"
          --target-sizes "$TARGET_SIZES"
          --gaps "$GAPS"
          --context-mode causal
          --steps "$STEPS"
          --batch-size "$BATCH_SIZE"
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
