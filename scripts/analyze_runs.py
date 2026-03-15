"""Analyze W&B training runs for a given round.

Usage:
    pixi run python scripts/analyze_runs.py r6
    pixi run python scripts/analyze_runs.py r5
    pixi run python scripts/analyze_runs.py r6 --top 3
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 30)

from moku.runs import fetch_wandb_histories, fetch_wandb_runs


def _ema(values: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Exponential moving average (W&B smoothing=0.8 equivalent)."""
    return pd.Series(values).ewm(alpha=alpha).mean().values


def print_section(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print("=" * 80)


def analyze(group: str, top_n: int | None = None) -> None:
    # ── 1. Summary table ──────────────────────────────────────────────────
    print_section(f"RUN SUMMARIES — {group}")

    runs_df = fetch_wandb_runs(group=group)
    if runs_df.empty:
        print("  No runs found.")
        return

    cols = ["name", "state", "eval/map", "eval/map_50", "eval/map_75", "eval/mar_400"]
    available = [c for c in cols if c in runs_df.columns]
    print(runs_df[available].sort_values("eval/map", ascending=False).reset_index(drop=True).to_string())

    # ── 2. Fetch history ──────────────────────────────────────────────────
    print_section("FETCHING HISTORY...")

    history = fetch_wandb_histories(group=group)
    history = history.sort_values(["run", "_step"])
    history["epoch"] = history.groupby("run")["train/epoch"].ffill()

    map_col = "train/eval/map" if "train/eval/map" in history.columns else "eval/map"
    print(f"  map_col={map_col}, rows={len(history)}, runs={history['run'].nunique()}")

    eval_hist = history.dropna(subset=[map_col])

    # ── 3. Best epoch per run ─────────────────────────────────────────────
    print_section("BEST EPOCH PER RUN")

    sorted_runs = sorted(eval_hist.groupby("run"), key=lambda x: x[1][map_col].max(), reverse=True)
    if top_n:
        sorted_runs = sorted_runs[:top_n]

    for run_name, grp in sorted_runs:
        best_idx = grp[map_col].idxmax()
        row = grp.loc[best_idx]
        last_epoch = int(grp["epoch"].max())
        last_map = grp.loc[grp["epoch"].idxmax(), map_col]
        best_epoch = int(row["epoch"])
        print(
            f"  {run_name:35s}  best_ep={best_epoch:5d}  best_mAP={row[map_col]:.4f}"
            f"  last_ep={last_epoch:5d}  last_mAP={last_map:.4f}"
            f"  drop={row[map_col] - last_map:+.4f}"
        )

    # ── 4. Smoothed comparison ────────────────────────────────────────────
    print_section("SMOOTHED COMPARISON (EMA α=0.2 ≈ W&B smoothing 0.8)")

    rows = []
    for run_name, grp in sorted_runs:
        maps = grp[map_col].values
        ema = _ema(maps)
        last100 = maps[-100:].mean() if len(maps) >= 100 else maps.mean()

        rows.append(
            {
                "run": run_name,
                "raw_max": f"{maps.max():.4f}",
                "ema_peak": f"{ema.max():.4f}",
                "ema_last": f"{ema[-1]:.4f}",
                "last100_mean": f"{last100:.4f}",
                "raw_mean": f"{maps.mean():.4f}",
            }
        )

    print(pd.DataFrame(rows).to_string(index=False))

    # ── 5. Plateau analysis ───────────────────────────────────────────────
    print_section("PLATEAU ANALYSIS (100-epoch windows)")

    for run_name, grp in sorted_runs:
        maps = grp[map_col].values
        epochs = grp["epoch"].values
        max_ep = int(epochs.max())

        print(f"\n  {run_name}:")
        step = 100
        for start in range(0, max_ep, step):
            end = start + step
            mask = (epochs >= start) & (epochs < end)
            if mask.sum() == 0:
                continue
            w = maps[mask]
            print(f"    ep {start:4d}-{end:4d}: n={mask.sum():3d}  mean={w.mean():.4f}  max={w.max():.4f}")

        # Trend: last 100 vs previous 100
        if max_ep >= 200:
            mask_last = (epochs >= max_ep - 100) & (epochs <= max_ep)
            mask_prev = (epochs >= max_ep - 200) & (epochs < max_ep - 100)
            if mask_last.sum() > 0 and mask_prev.sum() > 0:
                delta = maps[mask_last].mean() - maps[mask_prev].mean()
                print(f"    → last 100 vs prev 100: {delta:+.4f}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze W&B runs for a round")
    parser.add_argument("group", help="W&B group name (e.g. r5, r6)")
    parser.add_argument("--top", type=int, default=None, help="Only show top N runs")
    args = parser.parse_args()

    analyze(args.group, top_n=args.top)


if __name__ == "__main__":
    main()
