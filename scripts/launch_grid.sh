#!/usr/bin/env bash
# Launch HP grid search on HF Jobs.
#
# Usage:
#   bash scripts/launch_grid.sh
#
# Each run gets a unique name and logs to the same Trackio Space.
# Compare results at: https://huggingface.co/spaces/kaya-go/moku-training
# Then push the best config with --push-to-hub.
set -euo pipefail

export HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1

FLAVOR="a10g-small"
TIMEOUT="3h"
EPOCHS=50

# Define grid
LRS=(5e-5 1e-4 2e-4)
BATCH_SIZES=(4 8)
WEIGHT_DECAYS=(1e-4 1e-3)

for lr in "${LRS[@]}"; do
  for bs in "${BATCH_SIZES[@]}"; do
    for wd in "${WEIGHT_DECAYS[@]}"; do
      run_name="grid_lr${lr}_bs${bs}_wd${wd}"
      echo "=== Launching: ${run_name} ==="

      hf jobs uv run \
        --detach \
        --flavor "$FLAVOR" \
        --timeout "$TIMEOUT" \
        --secrets HF_TOKEN \
        scripts/train.py \
          --run-name "$run_name" \
          --lr "$lr" \
          --batch-size "$bs" \
          --weight-decay "$wd" \
          --num-epochs "$EPOCHS"

      # Delay to avoid /whoami rate limiting
      sleep 10
    done
  done
done

echo ""
echo "All jobs launched! Monitor at: https://huggingface.co/spaces/kaya-go/moku-training"
echo "List jobs: hf jobs ps"
