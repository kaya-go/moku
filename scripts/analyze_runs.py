"""Analyze W&B training runs for a given round.

Focuses on the 3 key metrics: mAP@50, corner_R4, stone_cdAP.

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

# Key metrics aligned with evaluation pipeline
KEY_SUMMARY_COLS = ["eval/map_50", "eval/corner_R4", "eval/stone_cdAP"]
# History column mapping: Trainer prefixes with train/ during training
_HIST_COLS = {
    "map_50": ("train/eval/map_50", "eval/map_50"),
    "corner_R4": ("train/eval/corner_R4", "eval/corner_R4"),
    "stone_cdAP": ("train/eval/stone_cdAP", "eval/stone_cdAP"),
    "map": ("train/eval/map", "eval/map"),
}


def _resolve_col(history: pd.DataFrame, key: str) -> str | None:
    """Return the first available column name for a metric key."""
    primary, fallback = _HIST_COLS[key]
    if primary in history.columns:
        return primary
    if fallback in history.columns:
        return fallback
    return None


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

    cols = ["name", "state"] + KEY_SUMMARY_COLS + ["eval/map"]
    available = [c for c in cols if c in runs_df.columns]
    sort_col = "eval/map_50" if "eval/map_50" in runs_df.columns else "eval/map"
    print(runs_df[available].sort_values(sort_col, ascending=False).reset_index(drop=True).to_string())

    # ── 2. Fetch history ──────────────────────────────────────────────────
    print_section("FETCHING HISTORY...")

    history = fetch_wandb_histories(group=group)
    history = history.sort_values(["run", "_step"])
    history["epoch"] = history.groupby("run")["train/epoch"].ffill()

    # Resolve metric columns
    cols_resolved = {}
    for key in ("map_50", "corner_R4", "stone_cdAP", "map"):
        col = _resolve_col(history, key)
        if col:
            cols_resolved[key] = col
    print(f"  resolved columns: {cols_resolved}")
    print(f"  rows={len(history)}, runs={history['run'].nunique()}")

    map50_col = cols_resolved.get("map_50")
    if not map50_col:
        print("  ⚠ No mAP@50 column found in history. Aborting.")
        return

    eval_hist = history.dropna(subset=[map50_col])

    # ── 3. Best epoch per run (by mAP@50) ─────────────────────────────────
    print_section("BEST EPOCH PER RUN (by mAP@50)")

    sorted_runs = sorted(eval_hist.groupby("run"), key=lambda x: x[1][map50_col].max(), reverse=True)
    if top_n:
        sorted_runs = sorted_runs[:top_n]

    for run_name, grp in sorted_runs:
        best_idx = grp[map50_col].idxmax()
        row = grp.loc[best_idx]
        best_epoch = int(row["epoch"])

        parts = [f"  {run_name:35s}  best_ep={best_epoch:4d}  mAP@50={row[map50_col]:.4f}"]
        for key in ("corner_R4", "stone_cdAP"):
            col = cols_resolved.get(key)
            if col and col in grp.columns and pd.notna(row.get(col)):
                parts.append(f"  {key}={row[col]:.4f}")
        print("".join(parts))

    # ── 4. Smoothed comparison ────────────────────────────────────────────
    print_section("SMOOTHED COMPARISON (EMA α=0.2)")

    rows = []
    for run_name, grp in sorted_runs:
        entry: dict = {"run": run_name}
        for key in ("map_50", "corner_R4", "stone_cdAP"):
            col = cols_resolved.get(key)
            if not col or col not in grp.columns:
                continue
            vals = grp[col].dropna().values
            if len(vals) == 0:
                continue
            ema = _ema(vals)
            entry[f"{key}_max"] = f"{vals.max():.4f}"
            entry[f"{key}_ema_peak"] = f"{ema.max():.4f}"
            entry[f"{key}_ema_last"] = f"{ema[-1]:.4f}"
        rows.append(entry)

    print(pd.DataFrame(rows).to_string(index=False))

    # ── 5. Plateau analysis (mAP@50) ─────────────────────────────────────
    print_section("PLATEAU ANALYSIS — mAP@50 (100-epoch windows)")

    for run_name, grp in sorted_runs:
        vals = grp[map50_col].values
        epochs = grp["epoch"].values
        max_ep = int(epochs.max())

        print(f"\n  {run_name}:")
        step = 100
        for start in range(0, max_ep, step):
            end = start + step
            mask = (epochs >= start) & (epochs < end)
            if mask.sum() == 0:
                continue
            w = vals[mask]
            print(f"    ep {start:4d}-{end:4d}: n={mask.sum():3d}  mean={w.mean():.4f}  max={w.max():.4f}")

        if max_ep >= 200:
            mask_last = (epochs >= max_ep - 100) & (epochs <= max_ep)
            mask_prev = (epochs >= max_ep - 200) & (epochs < max_ep - 100)
            if mask_last.sum() > 0 and mask_prev.sum() > 0:
                delta = vals[mask_last].mean() - vals[mask_prev].mean()
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
