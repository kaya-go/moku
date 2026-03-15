#!/usr/bin/env bash
# Round 4 hyperparameter search (10 runs).
#
# Strategy based on analysis of runs 2 & 3:
#   - Best run so far: lr=1e-3 run_3 → mAP@50:95=0.40, mAP@50=0.61
#   - Key observations:
#     1. Higher LR (1e-3) outperforms lower LR (2e-4) in mAP despite higher train loss
#     2. Cosine schedule decays LR to 0, causing model collapse after ~250 epochs
#     3. Training overfits after ~200-250 epochs on small dataset
#     4. mAP is noisy epoch-to-epoch, need stable training regime
#
# Groups:
#   A (runs 1-3): LR refinement around best (1e-3)
#   B (runs 4-6): Scheduler alternatives (avoid LR→0 collapse)
#   C (runs 7-8): Regularization via weight decay
#   D (runs 9-10): Combined best ideas
#
# Usage:
#   bash scripts/launch_grid_r4.sh           # Launch all 10 runs
#   bash scripts/launch_grid_r4.sh A         # Launch group A only (runs 1-3)
#   bash scripts/launch_grid_r4.sh B         # Launch group B only (runs 4-6)
#   bash scripts/launch_grid_r4.sh C         # Launch group C only (runs 7-8)
#   bash scripts/launch_grid_r4.sh D         # Launch group D only (runs 9-10)
set -euo pipefail

export HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1

# Load .env if present (local dev)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
[[ -f "$SCRIPT_DIR/../.env" ]] && set -a && source "$SCRIPT_DIR/../.env" && set +a

FLAVOR="a10g-large"
GROUP="${1:-all}"
ROUND="r4"
SLEEP=10

launch() {
	local run_name="$1"
	shift
	echo "  Launching: ${run_name}"
	hf jobs uv run \
		--detach \
		--flavor "$FLAVOR" \
		--timeout "3h" \
		--secrets HF_TOKEN \
		--secrets WANDB_API_KEY \
		scripts/train.py \
		--stage 2 \
		--round "$ROUND" \
		--run-name "$run_name" \
		--batch-size 16 \
		"$@"
	sleep "$SLEEP"
}

# ── Group A: LR refinement around 1e-3 ──────────────────────────────────
launch_group_a() {
	echo "=== Group A: LR refinement (3 runs) ==="

	# A1: Below optimum — confirm 1e-3 is the sweet spot
	launch "r4_lr5e-4_cos200" \
		--lr 5e-4 \
		--lr-scheduler cosine \
		--num-epochs 200

	# A2: Repeat best LR, shorter schedule (200 instead of 500)
	launch "r4_lr1e-3_cos200" \
		--lr 1e-3 \
		--lr-scheduler cosine \
		--num-epochs 200

	# A3: Above optimum — test higher end
	launch "r4_lr2e-3_cos200" \
		--lr 2e-3 \
		--lr-scheduler cosine \
		--num-epochs 200

	echo "Group A launched."
}

# ── Group B: Scheduler alternatives ──────────────────────────────────────
launch_group_b() {
	echo "=== Group B: Scheduler alternatives (3 runs) ==="

	# B1: Linear decay — gentler than cosine (LR decays linearly, stays higher longer)
	launch "r4_lr1e-3_linear200" \
		--lr 1e-3 \
		--lr-scheduler linear \
		--num-epochs 200

	# B2: Cosine with restarts — periodic warm restarts prevent collapse
	launch "r4_lr1e-3_cosrestart200" \
		--lr 1e-3 \
		--lr-scheduler cosine_with_restarts \
		--lr-scheduler-kwargs '{"num_cycles": 4}' \
		--num-epochs 200

	# B3: Cosine with min LR — floor at 10% of peak LR, never decays to 0
	launch "r4_lr1e-3_cosminlr200" \
		--lr 1e-3 \
		--lr-scheduler cosine_with_min_lr \
		--lr-scheduler-kwargs '{"min_lr_rate": 0.1}' \
		--num-epochs 200

	echo "Group B launched."
}

# ── Group C: Regularization via weight decay ─────────────────────────────
launch_group_c() {
	echo "=== Group C: Regularization (2 runs) ==="

	# C1: 10x weight decay (1e-3 vs default 1e-4)
	launch "r4_lr1e-3_cos200_wd1e-3" \
		--lr 1e-3 \
		--lr-scheduler cosine \
		--num-epochs 200 \
		--weight-decay 1e-3

	# C2: 50x weight decay (5e-3) + relaxed grad clipping
	launch "r4_lr1e-3_cos200_wd5e-3" \
		--lr 1e-3 \
		--lr-scheduler cosine \
		--num-epochs 200 \
		--weight-decay 5e-3 \
		--max-grad-norm 0.5

	echo "Group C launched."
}

# ── Group D: Combined best ideas ─────────────────────────────────────────
launch_group_d() {
	echo "=== Group D: Combined strategies (2 runs) ==="

	# D1: Restarts + weight decay + longer training
	launch "r4_lr1e-3_cosrestart300_wd1e-3" \
		--lr 1e-3 \
		--lr-scheduler cosine_with_restarts \
		--lr-scheduler-kwargs '{"num_cycles": 5}' \
		--num-epochs 300 \
		--weight-decay 1e-3

	# D2: Higher LR + linear decay + weight decay + longer warmup
	launch "r4_lr1.5e-3_linear200_wd1e-3" \
		--lr 1.5e-3 \
		--lr-scheduler linear \
		--num-epochs 200 \
		--weight-decay 1e-3 \
		--warmup-ratio 0.15

	echo "Group D launched."
}

# ── Main ─────────────────────────────────────────────────────────────────
case "$GROUP" in
A | a) launch_group_a ;;
B | b) launch_group_b ;;
C | c) launch_group_c ;;
D | d) launch_group_d ;;
all)
	launch_group_a
	echo ""
	launch_group_b
	echo ""
	launch_group_c
	echo ""
	launch_group_d
	;;
*)
	echo "Usage: $0 [A|B|C|D|all]"
	exit 1
	;;
esac

echo ""
echo "Monitor at: https://wandb.ai/hadim/moku/overview"
echo "List jobs: hf jobs ps"
