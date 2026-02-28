"""Training utilities for fine-tuning RT-DETR on the moku dataset."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

from moku.dataset import CATEGORIES, ID_TO_CATEGORY

# Model & dataset identifiers
BASE_MODEL = "PekingU/rtdetr_r18vd"
HF_DATASET = "kaya-go/moku-v1"
HF_MODEL = "kaya-go/moku-v1"
NUM_LABELS = len(CATEGORIES)


def load_image_processor(model_name: str = BASE_MODEL) -> RTDetrImageProcessor:
    """Load the RT-DETR image processor."""
    return RTDetrImageProcessor.from_pretrained(model_name)


def load_model(model_name: str = BASE_MODEL) -> RTDetrForObjectDetection:
    """Load RT-DETR model configured for moku's 3 categories."""
    return RTDetrForObjectDetection.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label=ID_TO_CATEGORY,
        label2id=CATEGORIES,
        ignore_mismatched_sizes=True,
    )


def _build_coco_target(example: dict) -> dict:
    """Convert a single HF dataset example to COCO annotation format."""
    return {
        "image_id": example["image_id"],
        "annotations": [
            {
                "id": int(ann_id),
                "category_id": int(cat),
                "bbox": [float(x) for x in bbox],
                "area": float(area),
                "iscrowd": int(iscrowd),
            }
            for ann_id, cat, bbox, area, iscrowd in zip(
                example["objects"]["id"],
                example["objects"]["category"],
                example["objects"]["bbox"],
                example["objects"]["area"],
                example["objects"]["iscrowd"],
            )
        ],
    }


def _unbatch_example(example: dict) -> dict:
    """Unwrap single-element batched example to plain format.

    HF datasets ``set_transform`` may pass examples in batched format (dict of
    lists) even for single-item access. This helper detects that case and
    extracts the first (only) element.
    """
    if isinstance(example.get("image"), list):
        return {k: v[0] if isinstance(v, list) else v for k, v in example.items()}
    return example


def _flip_horizontal(image: Image.Image, annotations: list[dict]) -> tuple[Image.Image, list[dict]]:
    """Flip image and COCO-format bboxes horizontally."""
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    w = image.width
    for ann in annotations:
        x, y, bw, bh = ann["bbox"]
        ann["bbox"] = [w - x - bw, y, bw, bh]
    return image, annotations


def make_train_transform(image_processor: RTDetrImageProcessor, flip_p: float = 0.5):
    """Create a per-example transform for training (with random horizontal flip)."""

    def transform(example: dict) -> dict:
        example = _unbatch_example(example)
        image = example["image"].convert("RGB")
        target = _build_coco_target(example)

        if random.random() < flip_p:
            image, target["annotations"] = _flip_horizontal(image, target["annotations"])

        return image_processor(images=[image], annotations=[target], return_tensors="pt")

    return transform


def make_eval_transform(image_processor: RTDetrImageProcessor):
    """Create a per-example transform for evaluation (no augmentation)."""

    def transform(example: dict) -> dict:
        example = _unbatch_example(example)
        image = example["image"].convert("RGB")
        target = _build_coco_target(example)
        return image_processor(images=[image], annotations=[target], return_tensors="pt")

    return transform


def collate_fn(batch: list[dict]) -> dict:
    """Collate: stack pixel_values, keep labels as a list of per-image dicts."""
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": [x["labels"] for x in batch],
    }


@torch.no_grad()
def evaluate_map(
    model: RTDetrForObjectDetection,
    dataset,
    image_processor: RTDetrImageProcessor,
    batch_size: int = 8,
    threshold: float = 0.3,
    device: str | None = None,
) -> dict:
    """Compute COCO mAP metrics on a dataset.

    Returns a dict with map, map_50, map_75, mar_100, and per-class AP.
    """
    from torchmetrics.detection import MeanAveragePrecision

    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    model = model.to(device)
    model.eval()

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"]

        outputs = model(pixel_values=pixel_values)

        # Post-process predictions to absolute xyxy coordinates
        orig_sizes = torch.stack([lab["orig_size"] for lab in labels]).to(device)
        results = image_processor.post_process_object_detection(outputs, target_sizes=orig_sizes, threshold=threshold)

        preds = [{"boxes": r["boxes"].cpu(), "scores": r["scores"].cpu(), "labels": r["labels"].cpu()} for r in results]

        # Convert ground truth from normalized cxcywh to absolute xyxy
        targets = []
        for lab in labels:
            boxes_cxcywh = lab["boxes"]
            orig_h, orig_w = lab["orig_size"]
            cx, cy, w, h = boxes_cxcywh.unbind(-1)
            abs_boxes = torch.stack(
                [
                    (cx - w / 2) * orig_w,
                    (cy - h / 2) * orig_h,
                    (cx + w / 2) * orig_w,
                    (cy + h / 2) * orig_h,
                ],
                dim=-1,
            )
            targets.append({"boxes": abs_boxes.cpu(), "labels": lab["class_labels"].cpu()})

        metric.update(preds, targets)

    result = metric.compute()
    return {k: v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v for k, v in result.items()}


def format_map_results(metrics: dict) -> pd.DataFrame:
    """Format mAP results as a clean DataFrame for display."""
    rows = [
        {"metric": "mAP", "value": metrics.get("map", float("nan"))},
        {"metric": "mAP@50", "value": metrics.get("map_50", float("nan"))},
        {"metric": "mAP@75", "value": metrics.get("map_75", float("nan"))},
        {"metric": "mAR@100", "value": metrics.get("mar_100", float("nan"))},
    ]
    per_class = metrics.get("map_per_class")
    if per_class is not None and hasattr(per_class, "__iter__"):
        for i, ap in enumerate(per_class):
            name = ID_TO_CATEGORY.get(i, f"class_{i}")
            val = ap.item() if hasattr(ap, "item") else ap
            rows.append({"metric": f"AP({name})", "value": val})
    return pd.DataFrame(rows)


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
        # Extract last eval entry
        eval_entries = [e for e in state.get("log_history", []) if "eval_loss" in e]
        last_eval = eval_entries[-1] if eval_entries else {}
        # Extract training args
        args_file = state_file.parent / "training_args.bin"
        config: dict = {"run": run_name}
        # Read args from config.json if available
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
