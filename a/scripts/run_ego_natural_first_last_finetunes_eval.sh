#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/chenyuanwan/anaconda3/envs/cav-carla/bin/python}"
CONFIG="${CONFIG:-configs/carla_0915.yaml}"
FINETUNE_ROOT="${FINETUNE_ROOT:-results/ego_enhanced_poet_finetune}"
TRAINING_FINAL_EGO="${TRAINING_FINAL_EGO:-results/joint_Jul_01_14_59_59/models/ego/checkpoint-67210.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-results/ego_natural_seed_eval/ego67210_plus_6finetunes_first_last_tm_hdv3_random10}"

mapfile -t FINETUNE_DIRS < <(find "$FINETUNE_ROOT" -maxdepth 1 -mindepth 1 -type d | sort)
if [[ "${#FINETUNE_DIRS[@]}" -ne 6 ]]; then
  echo "Expected 6 finetune directories under $FINETUNE_ROOT, found ${#FINETUNE_DIRS[@]}." >&2
  printf '%s\n' "${FINETUNE_DIRS[@]}" >&2
  exit 1
fi

first_ego_dir="${FINETUNE_DIRS[0]}/models/ego"
if [[ ! -d "$first_ego_dir" ]]; then
  echo "Missing ego checkpoint directory: $first_ego_dir" >&2
  exit 1
fi

extra_checkpoints=("$TRAINING_FINAL_EGO")
for finetune_dir in "${FINETUNE_DIRS[@]:1}"; do
  ego_dir="$finetune_dir/models/ego"
  if [[ ! -d "$ego_dir" ]]; then
    echo "Missing ego checkpoint directory: $ego_dir" >&2
    exit 1
  fi
  mapfile -t ckpts < <(find "$ego_dir" -maxdepth 1 -name 'checkpoint-*.pt' -type f | sort -V)
  if [[ "${#ckpts[@]}" -eq 0 ]]; then
    echo "No checkpoint-*.pt found under $ego_dir" >&2
    exit 1
  fi
  extra_checkpoints+=("${ckpts[0]}")
  if [[ "${ckpts[-1]}" != "${ckpts[0]}" ]]; then
    extra_checkpoints+=("${ckpts[-1]}")
  fi
done

cmd=(
  "$PYTHON_BIN" scripts/run_ego_natural_seed_eval.py
  --backend carla
  --config "$CONFIG"
  --ego-dir "$first_ego_dir"
  --checkpoint-interval 999999999
  --include-final
  --label-with-source
  --seed-count 10
  --eval-seed-base 101
  --num-natural 3
  --max-steps 200
  --hdv-control-mode traffic_manager
  --traffic-manager-port 8000
  --hdv-tm-speed-difference 0.0
  --hdv-tm-distance-to-leading-vehicle 8.0
  --randomize-scenarios
  --relative-offset-jitter 5.0
  --randomize-lane-offsets
  --randomize-lane-offset-min 0
  --randomize-lane-offset-max 2
  --carla-rpc-timeout 300
  --output-dir "$OUTPUT_DIR"
  --rerun-existing
)

for checkpoint in "${extra_checkpoints[@]}"; do
  cmd+=(--ego-checkpoint "$checkpoint")
done

printf 'Running command:\n'
printf '%q ' "${cmd[@]}"
printf '\n\n'
exec "${cmd[@]}"
