"""Training utilities for fine-tuning RT-DETR on the moku dataset."""

from __future__ import annotations

import json
import random
from pathlib import Path

import albumentations as A
import numpy as np
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


# ---------------------------------------------------------------------------
# Albumentations augmentation pipeline
# ---------------------------------------------------------------------------


def build_train_augmentation() -> A.Compose:
    """Build the training augmentation pipeline with albumentations.

    Returns an A.Compose that takes ``image`` (np.ndarray HWC uint8) and
    ``bboxes`` (list of [x_min, y_min, w, h] in pixels) plus ``category_ids``.
    Output bboxes remain in COCO format.
    """
    return A.Compose(
        [
            # Geometric
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Perspective(scale=(0.02, 0.08), p=0.5),
            A.Rotate(limit=15, border_mode=0, p=0.5),
            A.RandomResizedCrop(
                size=(640, 640),
                scale=(0.7, 1.0),
                ratio=(0.8, 1.2),
                p=0.3,
            ),
            # Photometric
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=(3, 7), p=1.0),
                ],
                p=0.3,
            ),
            A.RandomShadow(
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=5,
                shadow_roi=(0, 0, 1, 1),
                p=0.2,
            ),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["category_ids"],
            min_area=1.0,
            min_visibility=0.3,
        ),
    )


def _apply_augmentation(
    image: Image.Image,
    annotations: list[dict],
    aug: A.Compose,
) -> tuple[Image.Image, list[dict]]:
    """Apply albumentations augmentation to image and COCO annotations.

    Returns the augmented PIL image and updated annotations (bboxes that
    get fully cropped out are removed).
    """
    img_np = np.array(image)
    bboxes = [ann["bbox"] for ann in annotations]
    category_ids = [ann["category_id"] for ann in annotations]

    result = aug(image=img_np, bboxes=bboxes, category_ids=category_ids)

    aug_image = Image.fromarray(result["image"])
    aug_bboxes = result["bboxes"]
    aug_cat_ids = result["category_ids"]

    # Rebuild annotations keeping only surviving boxes
    aug_annotations = []
    for bbox, cat_id in zip(aug_bboxes, aug_cat_ids):
        x, y, w, h = bbox
        aug_annotations.append(
            {
                "id": 0,
                "category_id": int(cat_id),
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": 0,
            }
        )

    return aug_image, aug_annotations


def make_train_transform(image_processor: RTDetrImageProcessor, flip_p: float = 0.5):
    """Create a per-example transform for training with albumentations augmentation.

    The ``flip_p`` parameter is kept for API compatibility but is now ignored;
    flip probability is controlled by the albumentations pipeline.
    """
    aug = build_train_augmentation()

    def transform(example: dict) -> dict:
        example = _unbatch_example(example)
        image = example["image"].convert("RGB")
        target = _build_coco_target(example)

        image, target["annotations"] = _apply_augmentation(image, target["annotations"], aug)

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

    # max_detection_thresholds raised to 400 because a full 19x19 goban can have
    # up to 361 stones + 4 board corners = 365 detections per image.
    # backend="faster_coco_eval" is required for correct map with custom thresholds.
    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        max_detection_thresholds=[1, 10, 400],
        class_metrics=True,
        backend="faster_coco_eval",
    )

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
    """Format overall mAP results as a clean DataFrame for display."""
    rows = [
        {"metric": "mAP@50:95", "value": round(float(metrics.get("map", 0)), 4)},
        {"metric": "mAP@50", "value": round(float(metrics.get("map_50", 0)), 4)},
        {"metric": "mAP@75", "value": round(float(metrics.get("map_75", 0)), 4)},
        {"metric": "mAR@400", "value": round(float(metrics.get("mar_400", 0)), 4)},
    ]
    return pd.DataFrame(rows)


def format_map_per_class(metrics: dict) -> pd.DataFrame:
    """Format per-class AP results as a clean DataFrame for display.

    Returns one row per category with AP@50:95 and AP@50 values.
    """
    per_class = metrics.get("map_per_class")
    per_class_50 = metrics.get("map_per_class_50")  # may not be present in all torchmetrics versions

    if per_class is None or not hasattr(per_class, "__iter__"):
        return pd.DataFrame(columns=["category", "AP@50:95"])

    rows = []
    for i, ap in enumerate(per_class):
        name = ID_TO_CATEGORY.get(i, f"class_{i}")
        val = ap.item() if hasattr(ap, "item") else float(ap)
        row: dict = {"category": name, "AP@50:95": round(val, 4)}
        if per_class_50 is not None and hasattr(per_class_50, "__iter__"):
            ap50 = list(per_class_50)[i]
            ap50_val = ap50.item() if hasattr(ap50, "item") else float(ap50)
            row["AP@50"] = round(ap50_val, 4)
        rows.append(row)
    return pd.DataFrame(rows)


@torch.no_grad()
def evaluate_center_distance(
    model: RTDetrForObjectDetection,
    dataset,
    image_processor: RTDetrImageProcessor,
    batch_size: int = 8,
    threshold: float = 0.3,
    distance_thresholds: list[float] | None = None,
    device: str | None = None,
) -> dict:
    """Evaluate detections using center-point distance instead of IoU.

    For each image and class, predictions are matched to ground truth using
    greedy nearest-center matching. A match is "correct" if the Euclidean
    distance between predicted and GT bbox centers is below a threshold
    (expressed as a fraction of the image diagonal).

    Args:
        distance_thresholds: Fractions of image diagonal to use as matching
            thresholds.  Defaults to [0.01, 0.02, 0.05] (1%, 2%, 5%).

    Returns a dict with:
        - per_class: dict[class_name] -> {precision, recall, f1, mean_dist, median_dist}
            at each distance threshold
        - overall: aggregated across classes
        - all_distances: list of all matched center distances (in pixels)
    """
    from scipy.optimize import linear_sum_assignment

    if distance_thresholds is None:
        distance_thresholds = [0.01, 0.02, 0.05]

    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    model = model.to(device)
    model.eval()

    # Collect all per-image, per-class matched distances
    # Structure: class_id -> list of (gt_center, pred_center, distance, img_diag)
    all_matches: dict[int, list[tuple[float, float]]] = {i: [] for i in ID_TO_CATEGORY}
    # Track unmatched GT and preds per class
    unmatched_gt: dict[int, int] = {i: 0 for i in ID_TO_CATEGORY}
    unmatched_pred: dict[int, int] = {i: 0 for i in ID_TO_CATEGORY}

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"]

        outputs = model(pixel_values=pixel_values)

        orig_sizes = torch.stack([lab["orig_size"] for lab in labels]).to(device)
        results = image_processor.post_process_object_detection(outputs, target_sizes=orig_sizes, threshold=threshold)

        for i, (res, lab) in enumerate(zip(results, labels)):
            orig_h, orig_w = lab["orig_size"]
            img_diag = float((orig_h**2 + orig_w**2) ** 0.5)

            # Predicted boxes (xyxy) -> centers
            pred_boxes = res["boxes"].cpu()
            pred_labels = res["labels"].cpu()

            # GT boxes (normalized cxcywh) -> absolute centers
            boxes_cxcywh = lab["boxes"]
            gt_cx = boxes_cxcywh[:, 0] * orig_w
            gt_cy = boxes_cxcywh[:, 1] * orig_h
            gt_labels = lab["class_labels"].cpu()

            # Match per class
            for cls_id in ID_TO_CATEGORY:
                gt_mask = gt_labels == cls_id
                pred_mask = pred_labels == cls_id

                gt_centers = (
                    torch.stack([gt_cx[gt_mask], gt_cy[gt_mask]], dim=-1) if gt_mask.any() else torch.zeros(0, 2)
                )
                pred_centers_xyxy = pred_boxes[pred_mask]
                if pred_centers_xyxy.numel() > 0:
                    pred_cx = (pred_centers_xyxy[:, 0] + pred_centers_xyxy[:, 2]) / 2
                    pred_cy = (pred_centers_xyxy[:, 1] + pred_centers_xyxy[:, 3]) / 2
                    p_centers = torch.stack([pred_cx, pred_cy], dim=-1)
                else:
                    p_centers = torch.zeros(0, 2)

                n_gt = gt_centers.shape[0]
                n_pred = p_centers.shape[0]

                if n_gt == 0:
                    unmatched_pred[cls_id] += n_pred
                    continue
                if n_pred == 0:
                    unmatched_gt[cls_id] += n_gt
                    continue

                # Cost matrix: Euclidean distance between all pairs
                cost = torch.cdist(gt_centers.float(), p_centers.float())  # (n_gt, n_pred)
                gt_idx, pred_idx = linear_sum_assignment(cost.numpy())

                matched = 0
                for gi, pi in zip(gt_idx, pred_idx):
                    dist = float(cost[gi, pi])
                    all_matches[cls_id].append((dist, img_diag))
                    matched += 1

                unmatched_gt[cls_id] += n_gt - matched
                unmatched_pred[cls_id] += n_pred - matched

    # Compute metrics at each distance threshold
    per_class: dict[str, dict] = {}
    for cls_id, name in ID_TO_CATEGORY.items():
        matches = all_matches[cls_id]
        dists_px = [d for d, _ in matches]
        per_threshold = {}
        for dt in distance_thresholds:
            tp = sum(1 for d, diag in matches if d / diag <= dt)
            fn = sum(1 for d, diag in matches if d / diag > dt) + unmatched_gt[cls_id]
            fp = sum(1 for d, diag in matches if d / diag > dt) + unmatched_pred[cls_id]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            per_threshold[f"{dt:.0%}"] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "tp": tp,
            }
        per_class[name] = {
            "thresholds": per_threshold,
            "mean_dist_px": round(float(np.mean(dists_px)), 2) if dists_px else None,
            "median_dist_px": round(float(np.median(dists_px)), 2) if dists_px else None,
            "n_gt": len(matches) + unmatched_gt[cls_id],
            "n_pred": len(matches) + unmatched_pred[cls_id],
            "n_matched": len(matches),
        }

    return {"per_class": per_class, "distance_thresholds": distance_thresholds}


def format_center_distance_results(metrics: dict) -> pd.DataFrame:
    """Format center-distance evaluation results as a DataFrame."""
    rows = []
    for name, data in metrics["per_class"].items():
        row: dict = {
            "category": name,
            "n_gt": data["n_gt"],
            "n_pred": data["n_pred"],
            "matched": data["n_matched"],
            "mean_dist_px": data["mean_dist_px"],
            "median_dist_px": data["median_dist_px"],
        }
        for dt_label, dt_metrics in data["thresholds"].items():
            row[f"P@{dt_label}"] = dt_metrics["precision"]
            row[f"R@{dt_label}"] = dt_metrics["recall"]
            row[f"F1@{dt_label}"] = dt_metrics["f1"]
        rows.append(row)
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
