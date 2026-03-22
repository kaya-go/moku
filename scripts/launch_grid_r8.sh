#!/usr/bin/env bash
# Round 8 hyperparameter search (3 runs).
#
# Follow-up to r7 (single-stage v3). Key changes:
#   - Batch size 128 (vs 32 in r7) for 4x faster epoch throughput
#   - r7 runs timed out at ~275/500 epochs; r8 should complete well within 2h
#   - Focus on the two best LR regimes from r7: lr=1e-4 and lr=3e-4
#   - With batch_size x4, effective LR scales up; LR adjusted accordingly
#
# Strategy:
#   - lr=1e-4 linear: best mAP@50 in r7, conservative and stable
#   - lr=2e-4 cosine_with_min_lr: between r7's 1e-4 and 3e-4, with cosine decay
#   - lr=3e-4 cosine_with_min_lr: r7's 2nd best, retain for comparison at bs=128
#
# Timing: bs=128 on v3 → ~4x faster epochs → ~500ep in ~25-30min. Timeout 2h safe.
#
# Usage:
#   bash scripts/launch_grid_r8.sh
set -euo pipefail

export HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FLAVOR="a10g-large"
ROUND="r8"
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
		"$@"
	sleep "$SLEEP"
}

echo "=== Round 8: Single-stage v3, batch_size=128 (3 runs) ==="

launch "r8_lr1e-4_linear500" \
	--lr 1e-4 \
	--lr-scheduler linear \
	--num-epochs 500

launch "r8_lr2e-4_cosmin500" \
	--lr 2e-4 \
	--lr-scheduler cosine_with_min_lr \
	--lr-scheduler-kwargs '{"min_lr": 3e-5}' \
	--num-epochs 500

launch "r8_lr3e-4_cosmin500" \
	--lr 3e-4 \
	--lr-scheduler cosine_with_min_lr \
	--lr-scheduler-kwargs '{"min_lr": 5e-5}' \
	--num-epochs 500

echo ""
echo "All r8 runs launched. Monitor at https://wandb.ai/hadim/moku"
