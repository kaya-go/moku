"""``moku`` command line: evaluate, export, publish, build datasets, annotate, generate.

Every command loads ``.env`` from the working directory first (``GEMINI_API_KEY``).
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True, add_completion=False)
dataset_app = typer.Typer(no_args_is_help=True, help="Build, inspect and audit datasets.")
annotate_app = typer.Typer(no_args_is_help=True, help="Prepare and serve annotator workspaces.")
train_app = typer.Typer(no_args_is_help=True, help="Launch training jobs on HF Jobs and preview augmentation.")
runs_app = typer.Typer(no_args_is_help=True, help="Follow training runs (metrics and checkpoints in the runs bucket).")
app.add_typer(dataset_app, name="dataset")
app.add_typer(annotate_app, name="annotate")
app.add_typer(train_app, name="train")
app.add_typer(runs_app, name="runs")
console = Console(width=200)

DEFAULT_DATASET = "kaya-go/moku-v3"
ANNOTATOR_SERVER = Path(__file__).resolve().parents[2] / "tools" / "annotator" / "server.py"


@app.callback()
def _load_env() -> None:
    from dotenv import load_dotenv

    load_dotenv(Path.cwd() / ".env")


def _table(rows: list[dict], title: str) -> Table:
    table = Table(title=title, title_justify="left")
    for key in rows[0]:
        table.add_column(key, justify="left" if key == "model" else "right")
    for row in rows:
        table.add_row(*[_fmt(v) for v in row.values()])
    return table


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _pct(value: float, ci: tuple[float, float] | None = None) -> str:
    return f"{value:.0%}" + (f" [{ci[0]:.0%}–{ci[1]:.0%}]" if ci else "")


def _num(value: float, ci: tuple[float, float] | None = None, signed: bool = False) -> str:
    fmt = "+.1f" if signed else ".1f"
    return f"{value:{fmt}}" + (f" [{ci[0]:{fmt}}, {ci[1]:{fmt}}]" if ci else "")


def _short(source: str) -> str:
    """``org/name`` → ``name``; paths → file or directory name."""
    parts = Path(source).parts
    if source.startswith("hf://buckets/") and len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"  # <run>/<best|last>
    return Path(source).name or source


@app.command("eval")
def eval_cmd(
    models: list[str] = typer.Argument(
        ..., help="Hub repo ids, local dirs, hf://buckets/... checkpoints or .onnx files."
    ),
    splits: list[str] = typer.Option(["test"], "--split", "-s", help="Dataset split(s) to evaluate."),
    dataset: str = typer.Option(DEFAULT_DATASET, help="HF dataset id."),
    threshold: float = typer.Option(0.035, help="Stone score threshold (Kaya's default)."),
    sweep: bool = typer.Option(False, help="Also report board metrics over a range of stone thresholds."),
    figures: Path | None = typer.Option(None, help="Save the worst boards of each model as PNGs here."),
    worst: int = typer.Option(6, help="Number of worst boards to render with --figures."),
    json_out: Path | None = typer.Option(None, "--json", help="Write metrics and per-board tables as JSON."),
    batch_size: int = typer.Option(8),
    device: str | None = typer.Option(None, help="cuda / mps / cpu (default: best available)."),
) -> None:
    """Evaluate models: detection metrics and end-to-end board metrics (Kaya pipeline).

    The first model is the baseline for paired comparisons.
    """
    from datasets import load_dataset

    from moku.evaluation import evaluate, is_perfect, paired_difference, threshold_sweep
    from moku.inference import load_detector

    ds = load_dataset(dataset)
    report: dict = {}
    for split in splits:
        results = []
        for source in models:
            with console.status(f"{source} on {split}…"):
                results.append(evaluate(load_detector(source, device), ds[split], split, threshold, batch_size))
        rows = []
        for r in results:
            d, b = r.detection, r.board
            rows.append(
                {
                    "model": _short(r.model),
                    "mAP@50": d["mAP@50"],
                    "stone cdAP": d["stone_cdAP"],
                    "corner R@4": d["corner_R4"],
                    "perfect boards": _pct(b["perfect"], b["perfect_ci"]),
                    "errors / board": _num(b["errors"], b["errors_ci"]),
                    "≤2 errors": _pct(b["le2_errors"]),
                    "corner fail": _pct(b["corner_fail"]),
                }
            )
        b0 = results[0].board
        title = f"{split}: {b0['boards']} boards ({b0['empty']} empty) from ~{b0['photos']} photos/games (90% CI)"
        console.print(_table(rows, title))
        if len(results) > 1:
            base = results[0].boards.assign(perfect=is_perfect(results[0].boards))
            deltas = []
            for r in results[1:]:
                other = r.boards.assign(perfect=is_perfect(r.boards))
                perfect = paired_difference(base, other, "perfect")
                errors = paired_difference(base, other, "errors")
                deltas.append(
                    {
                        "model": _short(r.model),
                        "Δ perfect boards": f"{perfect['delta']:+.0%} [{perfect['ci'][0]:+.0%}, {perfect['ci'][1]:+.0%}]",
                        "Δ errors / board": _num(errors["delta"], errors["ci"], signed=True),
                    }
                )
            console.print(_table(deltas, f"{split}: paired difference vs {_short(results[0].model)} (90% CI)"))
        if sweep:
            thresholds = [0.01, 0.02, 0.035, 0.05, 0.1, 0.2, 0.3, 0.5]
            sweep_rows = []
            for r in results:
                table = threshold_sweep(r, thresholds)
                sweep_rows.append(
                    {"model": _short(r.model), **{f"{t:g}": _pct(p) for t, p in zip(thresholds, table.perfect)}}
                )
            console.print(_table(sweep_rows, f"{split}: perfect boards vs stone threshold"))
        if figures is not None:
            _save_worst(results, ds[split], split, figures, worst, threshold)
        report[split] = {
            r.model: {"detection": r.detection, "board": r.board, "boards": r.boards.to_dict("records")}
            for r in results
        }
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, indent=2, default=str))
        console.print(f"Wrote {json_out}")


def _save_worst(results, split_ds, split: str, out_dir: Path, worst: int, threshold: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from moku.viz import render_board_prediction

    out_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        slug = r.model.replace("/", "_").replace(":", "_")
        for row in r.boards.sort_values("errors", ascending=False).head(worst).itertuples():
            title = f"{r.model} — {split}[{row.index}] ({row.source}) — {row.errors} wrong intersections"
            fig = render_board_prediction(
                split_ds[int(row.index)]["image"], r.predictions[row.index], r.targets[row.index], threshold, title
            )
            fig.savefig(out_dir / f"{slug}_{split}_{row.index:03d}.png", dpi=90)
            plt.close(fig)
    console.print(f"Saved figures to {out_dir}")


@app.command()
def export(
    source: str = typer.Argument(..., help="Hub repo id (optionally @revision) or local model dir."),
    output: Path = typer.Option(Path("artifacts/model.onnx"), "--output", "-o"),
    opset: int = typer.Option(18),
    verify: bool = typer.Option(True, help="Compare ONNX Runtime outputs with PyTorch."),
    benchmark: bool = typer.Option(True, help="Single-thread CPU latency (proxy for Kaya's WASM)."),
) -> None:
    """Export a detector to ONNX with Kaya's I/O contract (pixel_values → logits, pred_boxes)."""
    from moku.export import benchmark_onnx, export_onnx, verify_onnx

    with console.status("Exporting…"):
        export_onnx(source, output, opset)
    console.print(f"Exported {output} ({output.stat().st_size / 1e6:.1f} MB)")
    if verify:
        console.print("Verification passed:", verify_onnx(source, output))
    if benchmark:
        console.print("Benchmark:", benchmark_onnx(output))


@app.command()
def publish(
    source: str = typer.Argument(..., help="hf://buckets/... checkpoint, local model dir or Hub repo id."),
    repo: str = typer.Option(..., help="Target Hub repo, e.g. kaya-go/moku-v4."),
    onnx: Path | None = typer.Option(None, help="ONNX file to upload as model.onnx (see `moku export`)."),
    public: bool = typer.Option(False, help="Create the repo as public (default: private)."),
    base_model: str = typer.Option("PekingU/rtdetr_r18vd", help="Pretrained checkpoint, for the model card."),
    dataset: str = typer.Option(DEFAULT_DATASET),
) -> None:
    """Push weights, processor, ONNX and a model card with test/validation metrics to the Hub."""
    from datasets import load_dataset

    from moku.evaluation import evaluate
    from moku.hub import model_card, publish_model
    from moku.inference import load_detector, resolve_source

    path = resolve_source(source)
    ds = load_dataset(dataset)
    lines = [
        "| split | mAP@50 | stone cdAP | corner R@4 | perfect boards | errors / board |",
        "|---|---|---|---|---|---|",
    ]
    for split in ("validation", "test"):
        with console.status(f"Evaluating on {split}…"):
            r = evaluate(load_detector(path), ds[split], split)
        d, b = r.detection, r.board
        lines.append(
            f"| {split} | {d['mAP@50']:.3f} | {d['stone_cdAP']:.3f} | {d['corner_R4']:.3f} "
            f"| {b['perfect']:.0%} [{b['perfect_ci'][0]:.0%}, {b['perfect_ci'][1]:.0%}] "
            f"| {b['errors']:.1f} [{b['errors_ci'][0]:.1f}, {b['errors_ci'][1]:.1f}] |"
        )
    card = model_card(repo.split("/")[-1], "\n".join(lines), base_model=base_model, dataset=dataset)
    console.print(card)
    url = publish_model(path, repo, private=not public, onnx_path=onnx, card=card)
    console.print(f"Published {url}")


@dataset_app.command("stats")
def dataset_stats(dataset: str = typer.Option(DEFAULT_DATASET)) -> None:
    """Images, objects and sources per split."""
    from datasets import load_dataset

    from moku.dataset import ID_TO_CATEGORY, compute_split_stats

    rows = []
    for split, ds in load_dataset(dataset).items():
        s = compute_split_stats(ds)
        rows.append(
            {
                "split": split,
                "images": s["num_images"],
                "objects/img": s["avg_objects_per_image"],
                **{ID_TO_CATEGORY[c]: n for c, n in sorted(s["category_counts"].items())},
                "sources": ", ".join(f"{k}: {v}" for k, v in s["source_counts"].most_common()),
            }
        )
    console.print(_table(rows, dataset))


@dataset_app.command("audit")
def dataset_audit(
    dataset: str = typer.Option(DEFAULT_DATASET),
    out: Path | None = typer.Option(None, help="Write flagged images as CSV."),
) -> None:
    """Flag annotations whose corners do not define a clean grid for their stones."""
    from datasets import load_dataset

    from moku.annotations import audit_boards

    flagged = audit_boards(load_dataset(dataset))
    if flagged.empty:
        console.print("No issues found.")
        return
    counts = flagged.groupby(["split", "source"]).size().reset_index(name="flagged")
    console.print(_table(counts.to_dict("records"), f"{dataset} — images with inconsistent board annotations"))
    if out is not None:
        flagged.to_csv(out, index=False)
        console.print(f"Wrote {out}")
    else:
        console.print(flagged.to_string(index=False))


@dataset_app.command("build-v3")
def dataset_build_v3(
    generated_dir: Path = typer.Option(Path("data/annotate_generated"), help="Output of `moku generate`."),
    push: str | None = typer.Option(None, help="Push to this Hub dataset repo (e.g. kaya-go/moku-v3)."),
) -> None:
    """Rebuild moku-v3: moku-v2 real images + generated images (train only)."""
    from moku.dataset import build_v3

    ds = build_v3(generated_dir)
    console.print(ds)
    if push:
        ds.push_to_hub(push, private=False)
        console.print(f"Pushed https://huggingface.co/datasets/{push}")


@dataset_app.command("build-v4")
def dataset_build_v4(
    roboflow: Path = typer.Option(..., help="Roboflow `my-go-detection` COCO export (unzipped)."),
    push: str | None = typer.Option(None, help="Push to this Hub dataset repo (private), e.g. kaya-go/moku-v4."),
) -> None:
    """Build moku-v4: moku-v3 + Roboflow photos split by photo (see moku.external.build_v4)."""
    from moku.external import build_v4

    ds = build_v4(roboflow)
    console.print(ds)
    if push:
        ds.push_to_hub(push, private=True)
        console.print(f"Pushed https://huggingface.co/datasets/{push}")


@dataset_app.command("build-gomrade")
def dataset_build_gomrade(
    root: Path = typer.Option(..., help="Unzipped Kaggle Gomrade archive (contains dataset/, dataset2/)."),
    frames_per_game: int = typer.Option(3),
    push: str | None = typer.Option(None, help="Push to this Hub dataset repo (always private: CC BY-NC-ND)."),
) -> None:
    """Build the Gomrade evaluation set (test split only; never used for training)."""
    from datasets import DatasetDict

    from moku.external import load_gomrade

    ds = DatasetDict({"test": load_gomrade(root, frames_per_game)})
    console.print(ds)
    if push:
        ds.push_to_hub(push, private=True)
        console.print(f"Pushed https://huggingface.co/datasets/{push}")


@annotate_app.command("prepare")
def annotate_prepare(
    dataset: str = typer.Option(DEFAULT_DATASET),
    splits: list[str] = typer.Option(["validation", "test"], "--split", "-s"),
    out: Path = typer.Option(Path("data/annotate")),
    only_flagged: bool = typer.Option(False, help="Export only images flagged by `moku dataset audit`."),
) -> None:
    """Export dataset splits to an annotator workspace, flagging suspicious boards."""
    from datasets import DatasetDict, load_dataset

    from moku.annotations import audit_boards, export_workspace

    ds = load_dataset(dataset)
    subset = DatasetDict({s: ds[s] for s in splits})
    n = export_workspace(subset, out, flagged=audit_boards(subset), only_flagged=only_flagged)
    console.print(f"Exported {n} images to {out}. Next: moku annotate serve --data-dir {out}")


@annotate_app.command("serve")
def annotate_serve(
    data_dir: Path = typer.Option(Path("data/annotate")),
    output: Path | None = typer.Option(None, help="Corrections file (default: <data-dir>/corrected.json)."),
    port: int = typer.Option(8765),
) -> None:
    """Serve the browser annotator (tools/annotator) on a workspace."""
    cmd = [sys.executable, str(ANNOTATOR_SERVER), "--data-dir", str(data_dir), "--port", str(port)]
    if output is not None:
        cmd += ["--output", str(output)]
    subprocess.run(cmd, check=False)


@app.command()
def generate(
    n: int = typer.Option(500, "--n", "-n", help="Total number of images to generate."),
    out_dir: Path = typer.Option(Path("data/annotate_generated"), "--out-dir", "-o", help="Output directory."),
    model: str = typer.Option("gemini-3.1-flash-image-preview", "--model", "-m", help="Model name."),
    prefix: str = typer.Option("gen", "--prefix", "-p", help="Filename prefix."),
    workers: int = typer.Option(10, "--workers", "-w", help="Number of parallel requests."),
    image_size: int = typer.Option(640, "--image-size", help="Synthetic image size."),
) -> None:
    """Generate photorealistic goban images conditioned on synthetic inputs (Gemini, resumable)."""
    from moku.generate import generate_conditioned

    asyncio.run(generate_conditioned(n, out_dir, model, prefix, workers, image_size))


@train_app.command("launch", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def train_launch(
    ctx: typer.Context,
    run_name: str = typer.Argument(..., help="Run name (directory in the runs bucket)."),
    flavor: str = typer.Option("a100-large", help="a10g-large is cheaper but often unavailable."),
    timeout: str = typer.Option("3h"),
    dry_run: bool = typer.Option(False, help="Print the hf jobs command without launching."),
) -> None:
    """Launch scripts/train.py on HF Jobs (pixi image, locked `cuda` env); extra arguments go to the script."""
    from moku.runs import launch, launch_command

    if dry_run:
        console.print(" ".join(launch_command(run_name, ctx.args, flavor=flavor, timeout=timeout)))
        return
    console.print(f"{run_name}: {launch(run_name, ctx.args, flavor=flavor, timeout=timeout)}")


@train_app.command("preview-aug")
def train_preview_aug(
    out: Path = typer.Option(Path("reports/augmentation")),
    n: int = typer.Option(20),
    light: bool = typer.Option(False, help="Preview the final-epochs pipeline (flips only)."),
    dataset: str = typer.Option(DEFAULT_DATASET),
    seed: int = typer.Option(0),
) -> None:
    """Save augmented training samples with their boxes, to eyeball the pipeline."""
    from datasets import load_dataset

    from moku.training.data import DetectionDataset, save_previews, train_augmentation, train_indices

    split = load_dataset(dataset)["train"]
    ds = DetectionDataset(split, train_indices(split), train_augmentation(strong=not light))
    console.print(f"Saved {len(save_previews(ds, out, n, seed))} samples to {out}")


def _run_row(name: str, run: dict) -> dict:
    cfg, epochs, best = run["config"], run["epochs"], run["best"]
    last = epochs.iloc[-1] if len(epochs) else {}
    m = best.get("metrics", {})
    return {
        "run": name,
        "model": cfg.get("model", "?"),
        "status": run["summary"].get("status", "running"),
        "epoch": f"{int(last.get('epoch', 0))}/{cfg.get('epochs', '?')}",
        "img/s": float(last.get("img_per_s", float("nan"))),
        "best ep": best.get("epoch", "-"),
        "val perfect": _pct(m["perfect"]) if m else "-",
        "val errors": _num(m["errors"]) if m else "-",
        "corner fail": _pct(m["corner_fail"]) if m else "-",
        "TP stone/corner": f"{m['stone_tp_score']:.2f}/{m['corner_tp_score']:.2f}" if m else "-",
    }


@runs_app.command("list")
def runs_list(runs: list[str] = typer.Argument(None, help="Run names (default: every run in the bucket).")) -> None:
    """One row per run: progress, throughput and best validation board metrics (EMA weights)."""
    from moku.runs import fetch_run, list_runs, read_run

    rows = [_run_row(name, read_run(fetch_run(name))) for name in (runs or list_runs())]
    if rows:
        console.print(_table(rows, "Training runs (best checkpoint = most perfect validation boards)"))


@runs_app.command("show")
def runs_show(
    runs: list[str] = typer.Argument(..., help="One or more run names."),
    plot: Path | None = typer.Option(None, help="Save training and validation curves to this PNG."),
    every: int = typer.Option(1, help="Print one epoch out of N."),
) -> None:
    """Per-epoch validation metrics of runs (and their curves with --plot)."""
    from moku.runs import fetch_run, plot_run, read_run

    loaded = {name: read_run(fetch_run(name)) for name in runs}
    for name, run in loaded.items():
        epochs = run["epochs"]
        cols = [
            c
            for c in (
                "epoch",
                "img_per_s",
                "val/perfect",
                "val/errors",
                "val/corner_fail",
                "val/mAP@50",
                "val/stone_tp_score",
                "val/corner_tp_score",
            )
            if c in epochs
        ]
        if len(epochs):
            console.print(_table(epochs[cols].iloc[::every].to_dict("records"), name))
    if plot is not None:
        console.print(f"Saved {plot_run(loaded, plot)}")


@runs_app.command("pull")
def runs_pull(run: str, which: str = typer.Option("best", help="best or last")) -> None:
    """Download a run's checkpoint under runs/<run>/<which>."""
    from moku.runs import BUCKET_PREFIX, RUNS_BUCKET, pull_checkpoint

    console.print(f"Downloaded to {pull_checkpoint(f'{BUCKET_PREFIX}{RUNS_BUCKET}/{run}/{which}')}")
