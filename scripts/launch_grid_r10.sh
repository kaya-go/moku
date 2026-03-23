#!/usr/bin/env bash
# Round 10 hyperparameter search (5 runs).
#
# Fixes scheduler calibration from r9:
#   - r9 used num_epochs=500 but only reached ~80-106 epochs in 4h
#   - warmup_ratio=0.1 meant 50 epochs of warmup → 60% of training was warmup
#   - cosine decay barely started (8-13% of cycle)
#   - result: LR still climbing when best mAP already reached → regression
#
# r10 changes:
#   - num_epochs matched to 4h budget: 80 (OS5), 100 (OS3)
#   - warmup_ratio=0.05 → ~4-5 epochs warmup (sufficient for fine-tune)
#   - LR range narrowed to 1e-4–3e-4 (sweet spot from r9)
#   - Focus on OS5 (best r9 config) + 2 OS3 for comparison
#   - cosine_with_min_lr confirmed as best scheduler
#
# Data composition:
#   OS5: 1910 real + 1000 gen = 2910 total → ~120 steps/ep → 80 ep ≈ 3.8h
#   OS3: 1146 real + 1000 gen = 2146 total → ~134 steps/ep → 100 ep ≈ 3.8h
#
# Usage:
#   bash scripts/launch_grid_r10.sh
set -euo pipefail

export HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FLAVOR="a10g-large"
ROUND="r10"
SLEEP=10

launch() {
	local run_name="$1"
	local num_epochs="$2"
	shift 2
	echo "  Launching: ${run_name} (${num_epochs} epochs)"
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
		--num-epochs "$num_epochs" \
		--warmup-ratio 0.05 \
		"$@"
	sleep "$SLEEP"
}

echo "=== Round 10: Calibrated scheduler, OS5/OS3 focus (5 runs) ==="

# ── A) OS5 × LR (80 epochs — fits 4h on A10G) ──────────────────────────

launch "r10_os5_lr1e-4_cosmin80" 80 \
	--lr 1e-4 \
	--lr-scheduler cosine_with_min_lr \
	--lr-scheduler-kwargs '{"min_lr": 1e-5}' \
	--oversample-real 5

launch "r10_os5_lr2e-4_cosmin80" 80 \
	--lr 2e-4 \
	--lr-scheduler cosine_with_min_lr \
	--lr-scheduler-kwargs '{"min_lr": 2e-5}' \
	--oversample-real 5

launch "r10_os5_lr3e-4_cosmin80" 80 \
	--lr 3e-4 \
	--lr-scheduler cosine_with_min_lr \
	--lr-scheduler-kwargs '{"min_lr": 3e-5}' \
	--oversample-real 5

# ── B) OS3 × LR (100 epochs — fits 4h on A10G) ─────────────────────────

launch "r10_os3_lr2e-4_cosmin100" 100 \
	--lr 2e-4 \
	--lr-scheduler cosine_with_min_lr \
	--lr-scheduler-kwargs '{"min_lr": 2e-5}' \
	--oversample-real 3

launch "r10_os3_lr3e-4_cosmin100" 100 \
	--lr 3e-4 \
	--lr-scheduler cosine_with_min_lr \
	--lr-scheduler-kwargs '{"min_lr": 3e-5}' \
	--oversample-real 3

echo ""
echo "All r10 runs launched. Monitor at https://wandb.ai/hadim/moku"
