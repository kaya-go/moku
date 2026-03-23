#!/usr/bin/env bash
# Round 9 hyperparameter search (9 runs).
#
# First round with the __getitems__ fix: DataLoader now processes real batches
# instead of silently dropping all but the first image per batch.
# ALL previous rounds (r5-r8) trained with effective batch_size=1.
#
# Strategy: single-stage v3 (382 real + 1000 generated) with real batching.
# Two axes explored:
#   A) LR × scheduler (baseline, no oversampling)
#   B) Oversampling real images to rebalance real:generated ratio
#
# Batch size: 16 (verified on A10G 24GB with actual batching).
# LR: re-exploration needed since effective batch size changed from 1 → 16.
#
# Data composition at each oversample level (v3 = 382 real + 1000 gen):
#   os=1 (default):  382 real + 1000 gen = 1382 total  (ratio 0.38:1)
#   os=3:           1146 real + 1000 gen = 2146 total  (ratio 1.15:1)
#   os=5:           1910 real + 1000 gen = 2910 total  (ratio 1.91:1)
#
# Usage:
#   bash scripts/launch_grid_r9.sh
set -euo pipefail

export HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FLAVOR="a10g-large"
ROUND="r9"
SLEEP=10

launch() {
	local run_name="$1"
	shift
	echo "  Launching: ${run_name}"
	hf jobs uv run \
		--detach \
		--flavor "$FLAVOR" \
		--timeout "4h" \
		--secrets HF_TOKEN \
		--secrets-file "$SCRIPT_DIR/../.env" \
		scripts/train.py \
		--stage 1 \
		--round "$ROUND" \
		--run-name "$run_name" \
		--batch-size 16 \
		--num-epochs 500 \
		"$@"
	sleep "$SLEEP"
}

echo "=== Round 9: Single-stage v3, bs=16 with __getitems__ fix (9 runs) ==="

# ── A) LR × scheduler baseline (no oversampling) ────────────────────────

launch "r9_lr1e-4_linear500" \
	--lr 1e-4 \
	--lr-scheduler linear

launch "r9_lr3e-4_linear500" \
	--lr 3e-4 \
	--lr-scheduler linear

launch "r9_lr5e-4_linear500" \
	--lr 5e-4 \
	--lr-scheduler linear

launch "r9_lr1e-4_cosmin500" \
	--lr 1e-4 \
	--lr-scheduler cosine_with_min_lr \
	--lr-scheduler-kwargs '{"min_lr": 1e-5}'

launch "r9_lr3e-4_cosmin500" \
	--lr 3e-4 \
	--lr-scheduler cosine_with_min_lr \
	--lr-scheduler-kwargs '{"min_lr": 3e-5}'

# ── B) Oversampling real images (best LR from above range) ──────────────

launch "r9_os3_lr3e-4_cosmin500" \
	--lr 3e-4 \
	--lr-scheduler cosine_with_min_lr \
	--lr-scheduler-kwargs '{"min_lr": 3e-5}' \
	--oversample-real 3

launch "r9_os5_lr3e-4_cosmin500" \
	--lr 3e-4 \
	--lr-scheduler cosine_with_min_lr \
	--lr-scheduler-kwargs '{"min_lr": 3e-5}' \
	--oversample-real 5

launch "r9_os3_lr5e-4_cosmin500" \
	--lr 5e-4 \
	--lr-scheduler cosine_with_min_lr \
	--lr-scheduler-kwargs '{"min_lr": 5e-5}' \
	--oversample-real 3

launch "r9_os5_lr5e-4_cosmin500" \
	--lr 5e-4 \
	--lr-scheduler cosine_with_min_lr \
	--lr-scheduler-kwargs '{"min_lr": 5e-5}' \
	--oversample-real 5

echo ""
echo "All r9 runs launched. Monitor at https://wandb.ai/hadim/moku"
