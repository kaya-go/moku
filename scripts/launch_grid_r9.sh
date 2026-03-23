#!/usr/bin/env bash
# Round 9 hyperparameter search (5 runs).
#
# Diagnosis from r7/r8: v3 single-stage training underperforms v2 two-stage (r5/r6).
# Root cause: 1000 generated images (~77% of train) dilute the real image signal.
# Val/test sets are 100% real images, so the model overfits to the generated domain.
#
# Strategy: **oversample real images** to rebalance real:generated ratio.
# With heavy augmentation, repeated real images look different each time.
#
# Data composition at each oversample level (v3 = 306 real + 1000 gen):
#   os=1 (current):  306 real + 1000 gen = 1306 total  (ratio 0.31:1)
#   os=3:            918 real + 1000 gen = 1918 total  (ratio 0.92:1)
#   os=5:           1530 real + 1000 gen = 2530 total  (ratio 1.53:1)
#
# Run 5 is a **real-only baseline** on v2 real (no generated data at all).
# This tells us whether generated data helps or hurts after rebalancing.
# Uses --oversample-real 10 to inflate 306 → 3060 images for bs=128 viability.
#
# All runs:
#   - bs=128 (proven faster convergence in r8)
#   - cosine_with_min_lr (min_lr=5e-5, proven in r6)
#   - 500 epochs, a10g-large, 4h timeout
#
# Targets to beat (best from r5/r6 on test set):
#   mAP@50 = 0.807  |  corner_R4 = 0.893  |  stone_cdAP@2% = 0.911
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
		--batch-size 128 \
		--lr-scheduler cosine_with_min_lr \
		--lr-scheduler-kwargs '{"min_lr": 5e-5}' \
		--num-epochs 500 \
		"$@"
	sleep "$SLEEP"
}

echo "=== Round 9: Oversample real + real-only baseline (5 runs) ==="

# ── Oversample 3× (near-balanced real:gen ≈ 1:1) ────────────────────────

launch "r9_os3_lr2e-4_cosmin500" \
	--lr 2e-4 \
	--oversample-real 3

launch "r9_os3_lr3e-4_cosmin500" \
	--lr 3e-4 \
	--oversample-real 3

# ── Oversample 5× (real-dominant ≈ 1.5:1) ───────────────────────────────

launch "r9_os5_lr2e-4_cosmin500" \
	--lr 2e-4 \
	--oversample-real 5

launch "r9_os5_lr3e-4_cosmin500" \
	--lr 3e-4 \
	--oversample-real 5

# ── Real-only baseline (no generated data) ───────────────────────────────
# Uses v2 "real" config (306 train images, same val/test as v3).
# oversample-real=10 inflates to 3060 images for bs=128 viability (~24 steps/ep).

launch "r9_real_lr3e-4_cosmin500" \
	--lr 3e-4 \
	--dataset kaya-go/moku-v2 \
	--dataset-config real \
	--oversample-real 10

echo ""
echo "All r9 runs launched. Monitor at https://wandb.ai/hadim/moku"
