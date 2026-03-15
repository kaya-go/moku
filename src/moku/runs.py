"""Utilities for fetching and analyzing W&B training runs."""

from __future__ import annotations

import tempfile

import pandas as pd

WANDB_ENTITY = "hadim"
WANDB_PROJECT = "moku"


def fetch_wandb_runs(
    project: str = WANDB_PROJECT,
    entity: str = WANDB_ENTITY,
    group: str | None = None,
) -> pd.DataFrame:
    """Fetch run summaries from W&B as a DataFrame.

    Returns one row per run with columns: name, group, state, tags, and
    all summary metrics (eval/map, eval/loss, etc.).
    """
    import wandb

    api = wandb.Api()
    filters = {}
    if group:
        filters["group"] = group
    runs = api.runs(f"{entity}/{project}", filters=filters)

    rows: list[dict] = []
    for r in runs:
        row = {
            "name": r.name,
            "group": r.group,
            "state": r.state,
            "tags": r.tags,
            "id": r.id,
        }
        row.update(dict(r.summary))
        for k, v in r.config.items():
            row[f"config/{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def fetch_wandb_histories(
    project: str = WANDB_PROJECT,
    entity: str = WANDB_ENTITY,
    group: str | None = None,
    keys: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch metric history for all runs in a group, concatenated.

    Returns a DataFrame with a ``run`` column identifying each run.
    """
    import wandb

    api = wandb.Api()
    filters = {}
    if group:
        filters["group"] = group
    runs = api.runs(f"{entity}/{project}", filters=filters)

    frames: list[pd.DataFrame] = []
    for r in runs:
        hist = r.history(pandas=True, samples=50000, keys=keys)
        hist["run"] = r.name
        frames.append(hist)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def list_wandb_model_artifacts(
    project: str = WANDB_PROJECT,
    entity: str = WANDB_ENTITY,
) -> pd.DataFrame:
    """List all model artifacts in a W&B project.

    Returns a DataFrame with columns: name, version, aliases, created_at,
    size_mb, and any metadata fields (epoch, eval_map, etc.).
    """
    import wandb

    api = wandb.Api()
    rows: list[dict] = []
    for collection in api.artifact_type("model", f"{entity}/{project}").collections():
        for artifact in collection.artifacts():
            row = {
                "name": artifact.name,
                "version": artifact.version,
                "aliases": artifact.aliases,
                "created_at": artifact.created_at,
                "size_mb": round(artifact.size / 1e6, 1) if artifact.size else None,
            }
            row.update(artifact.metadata or {})
            if artifact.logged_by():
                row["run"] = artifact.logged_by().name
            rows.append(row)

    return pd.DataFrame(rows)


def load_model_from_wandb(
    artifact_path: str,
    project: str = WANDB_PROJECT,
    entity: str = WANDB_ENTITY,
) -> tuple:
    """Download a model artifact from W&B and load it.

    Args:
        artifact_path: Full artifact path, e.g.
            ``"hadim/moku/model-r4_lr5e-4_cos200:best"`` or
            ``"hadim/moku/model-r4_lr5e-4_cos200:v3"``.
            If no entity/project prefix, it's added automatically.

    Returns:
        ``(image_processor, model)`` tuple.
    """
    import wandb
    from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

    from moku.dataset import CATEGORIES, ID_TO_CATEGORY

    api = wandb.Api()

    if artifact_path.count("/") < 2:
        artifact_path = f"{entity}/{project}/{artifact_path}"

    artifact = api.artifact(artifact_path, type="model")
    artifact_dir = artifact.download()

    ip = RTDetrImageProcessor.from_pretrained(artifact_dir)
    model = RTDetrForObjectDetection.from_pretrained(
        artifact_dir,
        num_labels=len(CATEGORIES),
        id2label=ID_TO_CATEGORY,
        label2id=CATEGORIES,
    )
    return ip, model


def save_model_artifact(
    model,
    image_processor,
    run_name: str,
    metadata: dict | None = None,
) -> None:
    """Save model + image processor as a W&B artifact.

    Must be called within an active ``wandb.run`` context.

    Args:
        model: The RTDetrForObjectDetection model.
        image_processor: The RTDetrImageProcessor.
        run_name: Used for artifact naming (``model-{run_name}``).
        metadata: Extra metadata dict (epoch, eval_map, etc.).
    """
    import wandb

    if wandb.run is None:
        raise RuntimeError("No active W&B run. Call wandb.init() first.")

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)
        image_processor.save_pretrained(tmpdir)

        artifact = wandb.Artifact(
            name=f"model-{run_name}",
            type="model",
            metadata=metadata or {},
        )
        artifact.add_dir(tmpdir)
        wandb.log_artifact(artifact)
