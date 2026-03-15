"""Utilities for loading and summarizing training run logs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_training_runs(runs_dir: str | Path) -> pd.DataFrame:
    """Load trainer log history from all runs in a directory.

    Expects each sub-directory to contain a ``trainer_state.json``.
    Returns a DataFrame with a ``run`` column identifying each run.
    """
    runs_dir = Path(runs_dir)
    rows: list[dict] = []
    for state_file in sorted(runs_dir.glob("*/trainer_state.json")):
        run_name = state_file.parent.name
        with open(state_file) as f:
            state = json.load(f)
        for entry in state.get("log_history", []):
            rows.append({"run": run_name, **entry})
    return pd.DataFrame(rows)


def summarize_runs(runs_dir: str | Path) -> pd.DataFrame:
    """Summarize final eval metrics for each run in a directory.

    Returns one row per run with the last recorded eval_loss and training config.
    """
    runs_dir = Path(runs_dir)
    summaries: list[dict] = []
    for state_file in sorted(runs_dir.glob("*/trainer_state.json")):
        run_name = state_file.parent.name
        with open(state_file) as f:
            state = json.load(f)
        eval_entries = [e for e in state.get("log_history", []) if "eval_loss" in e]
        last_eval = eval_entries[-1] if eval_entries else {}
        config: dict = {"run": run_name}
        config_file = state_file.parent / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                cfg = json.load(f)
            config.update({k: cfg[k] for k in ["num_labels"] if k in cfg})
        config["eval_loss"] = last_eval.get("eval_loss")
        config["epoch"] = last_eval.get("epoch")
        config["step"] = last_eval.get("step")
        summaries.append(config)
    return pd.DataFrame(summaries)
