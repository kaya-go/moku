#!/usr/bin/env bash
# Round 6 hyperparameter search (3 runs).
#
# Strategy based on round 5 analysis:
#   - Best run: lr=4e-4, linear, 500ep → mAP@50:95=0.6075 at epoch 413
#   - Key findings from r5:
#     1. Linear schedule >> cosine on long runs (cosine kills LR too early)
#     2. lr=4e-4 is the sweet spot (3e-4 also good, 5e-4 overshoots)
#     3. Augmentation (fixed in r5) made a real difference vs r4
#     4. Best run still improving at epoch 400-500 (+0.025 mean mAP)
#        → NOT plateaued yet, longer training should help
#     5. High eval variance (small val set) — best artifact saving essential
#
# Round 6 goals:
#   - Double training length (1000 epochs) to find the true plateau
#   - Focus on linear schedule (proven winner) + cosine_with_min_lr variant
#   - Narrow LR range: 3e-4 and 4e-4 only
#
# Runs:
#   1. lr=4e-4, linear, 1000ep — extend the best r5 config
#   2. lr=3e-4, linear, 1000ep — slightly lower LR backup
#   3. lr=4e-4, cosine_with_min_lr (min_lr=5e-5), 1000ep — cosine that
#      doesn't kill LR to zero
#   4. lr=3e-4, cosine_with_min_lr (min_lr=5e-5), 1000ep — best smoothed
#      r5 LR + hybrid scheduler
#
# Timing: 500ep ≈ 29min on a10g-large → 1000ep ≈ 58min. Timeout 2h is safe.
#
# Usage:
#   bash scripts/launch_grid_r6.sh
set -euo pipefail

export HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FLAVOR="a10g-large"
ROUND="r6"
SLEEP=10

launch() {
	local run_name="$1"
	shift
	echo "  Launching: ${run_name}"
	hf jobs uv run \
		--detach \
		--flavor "$FLAVOR" \
		--timeout "2h" \
		--secrets HF_TOKEN \
		--secrets-file "$SCRIPT_DIR/../.env" \
		scripts/train.py \
		--stage 2 \
		--round "$ROUND" \
		--run-name "$run_name" \
		--batch-size 32 \
		"$@"
	sleep "$SLEEP"
}

echo "=== Round 6: Long training (4 runs) ==="

launch "r6_lr4e-4_linear1000" \
	--lr 4e-4 \
	--lr-scheduler linear \
	--num-epochs 1000

launch "r6_lr3e-4_linear1000" \
	--lr 3e-4 \
	--lr-scheduler linear \
	--num-epochs 1000

launch "r6_lr4e-4_cosmin1000" \
	--lr 4e-4 \
	--lr-scheduler cosine_with_min_lr \
	--lr-scheduler-kwargs '{"min_lr": 5e-5}' \
	--num-epochs 1000

launch "r6_lr3e-4_cosmin1000" \
	--lr 3e-4 \
	--lr-scheduler cosine_with_min_lr \
	--lr-scheduler-kwargs '{"min_lr": 5e-5}' \
	--num-epochs 1000

echo "Round 6 launched (4 runs)."
