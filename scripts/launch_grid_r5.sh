#!/usr/bin/env bash
# Round 5 hyperparameter search (6 runs).
#
# Strategy based on round 4 analysis:
#   - Best run: lr=5e-4, cosine, 200ep → mAP@50:95=0.598 at epoch 70
#   - Key findings from r4:
#     1. lr=5e-4 clearly beats lr=1e-3 (0.598 vs 0.597/0.588/0.584)
#     2. lr=2e-3 is catastrophic (0.067 final)
#     3. Weight decay hurts at all levels tested (1e-3, 5e-3)
#     4. Cosine + linear schedulers are both good
#     5. mAP curves are noisy — evaluation variance is high
#     6. mAP curves still trending up at epoch 200 — longer training may help
#
# Round 5 goals:
#   - Refine LR in [2e-4, 5e-4] range
#   - Longer training (500 epochs) — best checkpoint saved via W&B artifacts
#   - No early stopping — let it run and rely on artifact saving
#   - All runs save best W&B artifacts automatically
#
# Groups:
#   A (runs 1-4): LR grid in [2e-4, 5e-4] with cosine schedule, 500 epochs
#   B (runs 5-6): Linear schedule at best LR range, 500 epochs
#
# Usage:
#   bash scripts/launch_grid_r5.sh           # Launch all 6 runs
#   bash scripts/launch_grid_r5.sh A         # Launch group A only
#   bash scripts/launch_grid_r5.sh B         # Launch group B only
set -euo pipefail

export HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FLAVOR="a10g-large"
GROUP="${1:-all}"
ROUND="r5"
SLEEP=10

launch() {
	local run_name="$1"
	shift
	echo "  Launching: ${run_name}"
	hf jobs uv run \
		--detach \
		--flavor "$FLAVOR" \
		--timeout "8h" \
		--secrets HF_TOKEN \
		--secrets-file "$SCRIPT_DIR/../.env" \
		scripts/train.py \
		--stage 2 \
		--round "$ROUND" \
		--run-name "$run_name" \
		--batch-size 16 \
		"$@"
	sleep "$SLEEP"
}

# ── Group A: LR grid in [2e-4, 5e-4] with cosine ────────────────────────
launch_group_a() {
	echo "=== Group A: LR refinement [2e-4 .. 5e-4] (4 runs) ==="

	launch "r5_lr2e-4_cos500" \
		--lr 2e-4 \
		--lr-scheduler cosine \
		--num-epochs 500

	launch "r5_lr3e-4_cos500" \
		--lr 3e-4 \
		--lr-scheduler cosine \
		--num-epochs 500

	launch "r5_lr4e-4_cos500" \
		--lr 4e-4 \
		--lr-scheduler cosine \
		--num-epochs 500

	launch "r5_lr5e-4_cos500" \
		--lr 5e-4 \
		--lr-scheduler cosine \
		--num-epochs 500

	echo "Group A launched."
}

# ── Group B: Linear schedule comparison ──────────────────────────────────
launch_group_b() {
	echo "=== Group B: Linear schedule (2 runs) ==="

	launch "r5_lr4e-4_linear500" \
		--lr 4e-4 \
		--lr-scheduler linear \
		--num-epochs 500

	launch "r5_lr5e-4_linear500" \
		--lr 5e-4 \
		--lr-scheduler linear \
		--num-epochs 500

	echo "Group B launched."
}

# ── Main ─────────────────────────────────────────────────────────────────
case "$GROUP" in
A | a) launch_group_a ;;
B | b) launch_group_b ;;
all)
	launch_group_a
	echo ""
	launch_group_b
	;;
*)
	echo "Usage: $0 [A|B|all]"
	exit 1
	;;
esac
