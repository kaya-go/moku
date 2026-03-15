#!/usr/bin/env python3
"""Delete specific runs from the Trackio experiment logs on HF Hub.

Removes matching entries from both moku.parquet (logs) and moku_configs.parquet
(configs) in the kaya-go/moku-experiment-logs dataset repo.

Usage:
    # List all runs (dry-run):
    pixi run python scripts/delete_runs.py

    # Delete specific runs by name:
    pixi run python scripts/delete_runs.py stage2_lr2e-4_run_3 stage2_lr5e-4_run_3

    # Delete runs matching a pattern:
    pixi run python scripts/delete_runs.py --pattern "run_3"

    # Delete ALL runs (reset):
    pixi run python scripts/delete_runs.py --all
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "kaya-go/moku-experiment-logs"
LOGS_FILE = "moku.parquet"
CONFIGS_FILE = "moku_configs.parquet"


def download_parquets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download current logs and configs parquets from HF."""
    path_logs = hf_hub_download(REPO_ID, LOGS_FILE, repo_type="dataset", force_download=True)
    path_configs = hf_hub_download(REPO_ID, CONFIGS_FILE, repo_type="dataset", force_download=True)
    return pd.read_parquet(path_logs), pd.read_parquet(path_configs)


def get_all_run_names(df_logs: pd.DataFrame, df_configs: pd.DataFrame) -> set[str]:
    """Collect all unique run names from logs and configs."""
    names: set[str] = set()
    if "run_name" in df_logs.columns:
        names.update(df_logs["run_name"].dropna().unique())
    if "output_dir" in df_configs.columns:
        names.update(df_configs["output_dir"].dropna().str.replace("runs/", "", n=1).unique())
    return names


def list_runs(df_logs: pd.DataFrame, df_configs: pd.DataFrame) -> None:
    """Print all runs with their log counts."""
    print("=== Experiment Runs ===\n")
    if "run_name" in df_logs.columns:
        counts = df_logs["run_name"].value_counts()
        for name, count in counts.items():
            print(f"  {name:40s}  {count:5d} log entries")
    print(f"\n  Total: {len(df_logs)} log entries, {len(df_configs)} config entries")


def filter_runs(
    df_logs: pd.DataFrame,
    df_configs: pd.DataFrame,
    run_names: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """Remove entries matching run_names. Returns cleaned DFs and removal counts."""
    log_mask = df_logs["run_name"].isin(run_names)
    config_mask = df_configs["output_dir"].apply(
        lambda x: any(name in str(x) for name in run_names) if pd.notna(x) else False
    )
    return (
        df_logs[~log_mask],
        df_configs[~config_mask],
        int(log_mask.sum()),
        int(config_mask.sum()),
    )


def upload_parquets(df_logs: pd.DataFrame, df_configs: pd.DataFrame, commit_msg: str) -> None:
    """Upload cleaned parquets back to HF."""
    api = HfApi()
    with tempfile.TemporaryDirectory() as tmp:
        logs_path = Path(tmp) / LOGS_FILE
        configs_path = Path(tmp) / CONFIGS_FILE
        df_logs.to_parquet(logs_path, index=False)
        df_configs.to_parquet(configs_path, index=False)

        api.upload_file(
            path_or_fileobj=str(logs_path),
            path_in_repo=LOGS_FILE,
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message=commit_msg,
        )
        api.upload_file(
            path_or_fileobj=str(configs_path),
            path_in_repo=CONFIGS_FILE,
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message=commit_msg,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Delete runs from Trackio experiment logs")
    p.add_argument("run_names", nargs="*", help="Run names to delete (e.g. stage2_lr2e-4_run_3)")
    p.add_argument("--pattern", type=str, help="Delete runs matching this regex pattern")
    p.add_argument("--all", action="store_true", help="Delete ALL runs (reset)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading experiment logs from {REPO_ID}...")
    try:
        df_logs, df_configs = download_parquets()
    except Exception as e:
        print(f"Error downloading parquets: {e}")
        print("The experiment logs repo may be empty.")
        sys.exit(1)

    all_names = get_all_run_names(df_logs, df_configs)

    # No args → list mode
    if not args.run_names and not args.pattern and not args.all:
        list_runs(df_logs, df_configs)
        print("\nTo delete runs, pass run names or --pattern. See --help.")
        return

    # Determine which runs to delete
    if args.all:
        to_delete = all_names
    elif args.pattern:
        to_delete = {name for name in all_names if re.search(args.pattern, name)}
    else:
        to_delete = set(args.run_names)

    if not to_delete:
        print("No matching runs found.")
        return

    print(f"\nRuns to delete ({len(to_delete)}):")
    for name in sorted(to_delete):
        print(f"  - {name}")

    # Filter
    df_logs_clean, df_configs_clean, n_logs, n_configs = filter_runs(df_logs, df_configs, to_delete)

    print(f"\nWill remove: {n_logs} log entries, {n_configs} config entries")
    print(f"Remaining:   {len(df_logs_clean)} log entries, {len(df_configs_clean)} config entries")

    if n_logs == 0 and n_configs == 0:
        print("Nothing to remove.")
        return

    # Upload
    names_str = ", ".join(sorted(to_delete))
    commit_msg = f"chore: delete runs: {names_str}"
    if len(commit_msg) > 120:
        commit_msg = f"chore: delete {len(to_delete)} runs"

    print("\nUploading cleaned parquets...")
    upload_parquets(df_logs_clean, df_configs_clean, commit_msg)
    print("Done!")


if __name__ == "__main__":
    main()
