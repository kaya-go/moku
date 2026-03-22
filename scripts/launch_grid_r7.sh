#!/usr/bin/env bash
# Round 7 hyperparameter search (4 runs).
#
# First round on v3 dataset (real + generated, no synthetic).
# Single-stage training: COCO pretrained → fine-tune on v3 directly.
#
# Strategy:
#   - v3 has more training data than v2 real (real + Gemini-generated images)
#   - Single-stage: fresh detection head from COCO pretrained backbone
#   - Broad LR sweep: 1e-4 → 5e-4 to find the right regime for single-stage
#   - Linear scheduler (proven winner from r5/r6) + one cosine_with_min_lr backup
#   - 500 epochs to see trends, extend in r8 if not plateaued
#
# Runs:
#   1. lr=1e-4, linear, 500ep — conservative LR
#   2. lr=3e-4, linear, 500ep — mid-range LR
#   3. lr=5e-4, linear, 500ep — aggressive LR
#   4. lr=3e-4, cosine_with_min_lr (min_lr=5e-5), 500ep — cosine variant
#
# Timing: 500ep on v3 (larger dataset) ≈ 40-50min on a10g-large. Timeout 2h safe.
#
# Usage:
#   bash scripts/launch_grid_r7.sh
set -euo pipefail

export HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FLAVOR="a10g-large"
ROUND="r7"
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
		--stage 1 \
		--round "$ROUND" \
		--run-name "$run_name" \
		--batch-size 32 \
		"$@"
	sleep "$SLEEP"
}

echo "=== Round 7: Single-stage v3 (4 runs) ==="

launch "r7_lr1e-4_linear500" \
	--lr 1e-4 \
	--lr-scheduler linear \
	--num-epochs 500

launch "r7_lr3e-4_linear500" \
	--lr 3e-4 \
	--lr-scheduler linear \
	--num-epochs 500

launch "r7_lr5e-4_linear500" \
	--lr 5e-4 \
	--lr-scheduler linear \
	--num-epochs 500

launch "r7_lr3e-4_cosmin500" \
	--lr 3e-4 \
	--lr-scheduler cosine_with_min_lr \
	--lr-scheduler-kwargs '{"min_lr": 5e-5}' \
	--num-epochs 500

echo ""
echo "All r7 runs launched. Monitor at https://wandb.ai/hadim/moku"
