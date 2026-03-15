#!/usr/bin/env bash
# Two-stage training launcher for HF Jobs.
#
# Usage:
#   bash scripts/launch_grid.sh           # Launch stage 1 + stage 2 LR sweep
#   bash scripts/launch_grid.sh stage1    # Stage 1 only (synthetic pre-train)
#   bash scripts/launch_grid.sh stage2    # Stage 2 only (real fine-tune LR sweep)
#
# Logs to W&B for comparison.
# Compare results at: https://wandb.ai/kaya-go/moku
set -euo pipefail

export HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1

FLAVOR="a10g-large"
STAGE="${1:-both}"

# ── Stage 1: Synthetic pre-training ──────────────────────────────────────
launch_stage1() {
	echo "=== Stage 1: Synthetic pre-training ==="
	hf jobs uv run \
		--detach \
		--flavor "$FLAVOR" \
		--timeout "3h" \
		--secrets HF_TOKEN \
		scripts/train.py \
		--stage 1 \
		--run-name "stage1_synthetic" \
		--lr 1e-4 \
		--num-epochs 30 \
		--batch-size 16
	echo "Stage 1 launched. Wait for completion before launching stage 2."
}

# ── Stage 2: Real fine-tuning LR sweep ───────────────────────────────────
launch_stage2() {
	echo "=== Stage 2: Real fine-tuning LR sweep ==="
	LRS=(2e-4 5e-4 1e-3)
	for lr in "${LRS[@]}"; do
		run_id="run_3"
		run_name="stage2_lr${lr}_${run_id}"
		hub_rev="stage2-lr${lr}_${run_id}"
		echo "  Launching: ${run_name} → branch ${hub_rev}"
		hf jobs uv run \
			--detach \
			--flavor "$FLAVOR" \
			--timeout "3h" \
			--secrets HF_TOKEN \
			scripts/train.py \
			--stage 2 \
			--run-name "$run_name" \
			--lr "$lr" \
			--num-epochs 500 \
			--batch-size 16
		sleep 3
	done
	echo "Stage 2 sweep launched."
}

# ── Main ─────────────────────────────────────────────────────────────────
case "$STAGE" in
stage1) launch_stage1 ;;
stage2) launch_stage2 ;;
both)
	launch_stage1
	echo ""
	echo "⚠ Stage 1 must finish before launching stage 2."
	echo "  Run 'hf jobs ps' to monitor, then:"
	echo "  bash scripts/launch_grid.sh stage2"
	;;
*)
	echo "Usage: $0 [stage1|stage2|both]"
	exit 1
	;;
esac

echo ""
echo "Monitor at: https://huggingface.co/spaces/kaya-go/moku-experiments"
echo "List jobs: hf jobs ps"
