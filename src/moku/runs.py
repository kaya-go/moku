"""Training runs on HF Jobs: launch them and read what they write to the runs bucket.

A run launched by :func:`launch` writes ``<run>/`` in the bucket (see
:mod:`moku.training.engine` for the layout). Nothing else tracks runs: metrics,
logs and checkpoints are all read back from the bucket.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from pathlib import Path

import pandas as pd

RUNS_BUCKET = "hadim/moku-runs"
LOCAL_RUNS = Path("runs")
BUCKET_PREFIX = "hf://buckets/"


def git_state() -> str:
    """``<sha>`` of HEAD, with ``-dirty`` when the working tree has changes."""
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "src", "scripts"], capture_output=True, text=True).stdout
    return sha + ("-dirty" if dirty.strip() else "")


PIXI_IMAGE = "ghcr.io/prefix-dev/pixi:0.81.0"
JOB_FILES = ("pixi.toml", "pixi.lock", "pyproject.toml", "src", "scripts")
JOB_STAGING = LOCAL_RUNS / "_job"


def stage_project(dest: Path = JOB_STAGING) -> Path:
    """Copy what a job needs (locked pixi project, package, scripts) into ``dest``, caches excluded."""
    import shutil

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.egg-info", ".DS_Store")
    for name in JOB_FILES:
        if Path(name).is_dir():
            shutil.copytree(name, dest / name, ignore=ignore)
        else:
            shutil.copy2(name, dest / name)
    return dest


def launch_command(
    run_name: str,
    train_args: list[str],
    flavor: str = "a100-large",
    timeout: str = "3h",
    bucket: str = RUNS_BUCKET,
) -> list[str]:
    """``hf jobs run`` in the pixi image: the project is installed from ``pixi.lock`` (``cuda`` env).

    The staged project is mounted read-only at ``/moku-ro`` and copied (pixi writes ``.pixi/``);
    the runs bucket is mounted read-write at ``/runs``.
    """
    train = shlex.join(["python", "scripts/train.py", "--run-name", run_name, "--output-dir", "/runs", *train_args])
    return [
        "hf", "jobs", "run", "--detach",
        "--flavor", flavor,
        "--timeout", timeout,
        "--name", run_name,
        "--label", "moku",
        "--secrets", "HF_TOKEN",
        "--env", f"MOKU_GIT={git_state()}",
        "--volume", f"./{JOB_STAGING}:/moku-ro",
        "--volume", f"{BUCKET_PREFIX}{bucket}:/runs",
        PIXI_IMAGE,
        "bash", "-c", f"cp -r /moku-ro /moku && cd /moku && pixi run --frozen -e cuda {train}",
    ]  # fmt: skip


def launch(run_name: str, train_args: list[str], **kwargs) -> str:
    """Stage the project and launch a training job; returns its URL."""
    stage_project()
    proc = subprocess.run(launch_command(run_name, train_args, **kwargs), capture_output=True, text=True)
    match = re.search(r"url=(\S+)", proc.stdout + proc.stderr)
    if proc.returncode != 0 or not match:
        raise RuntimeError(f"hf jobs failed:\n{proc.stdout}\n{proc.stderr}")
    return match.group(1)


def list_runs(bucket: str = RUNS_BUCKET) -> list[str]:
    from huggingface_hub import HfApi

    return sorted(
        f.path.rstrip("/")
        for f in HfApi().list_bucket_tree(bucket)
        if type(f).__name__ == "BucketFolder" and not f.path.startswith("_")
    )


def fetch_run(run: str, bucket: str = RUNS_BUCKET, dest: Path = LOCAL_RUNS) -> Path:
    """Download a run's small files (config, metrics, summary, log) to ``dest/<run>``."""
    from huggingface_hub import HfApi

    files = ["config.json", "metrics.jsonl", "summary.json", "train.log", "best/eval.json"]
    local = dest / run
    HfApi().download_bucket_files(bucket, [(f"{run}/{f}", local / f) for f in files])
    return local


def pull_checkpoint(source: str, dest: Path = LOCAL_RUNS) -> Path:
    """``hf://buckets/<ns>/<bucket>/<run>/<best|last>`` → local directory under ``dest``."""
    from huggingface_hub import HfApi

    namespace, name, *path = source.removeprefix(BUCKET_PREFIX).strip("/").split("/")
    local = dest.joinpath(*path)
    HfApi().sync_bucket(source, str(local), quiet=True)
    return local


def read_run(path: Path) -> dict:
    """Config, summary and metrics (``train`` and ``epoch`` frames) of a fetched run."""

    def load(name: str) -> dict:
        p = path / name
        return json.loads(p.read_text()) if p.exists() else {}

    records = (
        [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
        if (path / "metrics.jsonl").exists()
        else []
    )
    frame = pd.DataFrame(records)
    by_type = {t: frame[frame["type"] == t].dropna(axis=1, how="all") for t in ("train", "epoch")} if len(frame) else {}
    return {
        "config": load("config.json"),
        "summary": load("summary.json"),
        "best": load("best/eval.json"),
        "train": by_type.get("train", pd.DataFrame()),
        "epochs": by_type.get("epoch", pd.DataFrame()),
    }


def plot_run(runs: dict[str, dict], out: Path) -> Path:
    """Training loss, lr and validation curves of one or more runs, as a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("train", "loss", "train loss"),
        ("train", "lr_head", "lr (head)"),
        ("epochs", "val/perfect", "val perfect boards"),
        ("epochs", "val/errors", "val errors / board"),
        ("epochs", "val/corner_fail", "val corner failures"),
        ("epochs", "val/mAP@50", "val mAP@50"),
        ("epochs", "val/stone_tp_score", "stone TP score (median)"),
        ("epochs", "val/corner_tp_score", "corner TP score (median)"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for ax, (kind, column, title) in zip(axes.flat, panels):
        for name, run in runs.items():
            frame = run[kind]
            if column in frame:
                x = frame["iter"] if kind == "train" else frame["epoch"]
                ax.plot(x, frame[column], label=name, lw=1)
        ax.set_title(title)
        ax.set_xlabel("iteration" if kind == "train" else "epoch")
        if column == "loss":
            ax.set_yscale("log")
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=80)
    plt.close(fig)
    return out
