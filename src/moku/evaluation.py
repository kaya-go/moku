"""Evaluation utilities for moku object detection models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

from moku.dataset import ID_TO_CATEGORY
from moku.model import collate_fn


@torch.no_grad()
def evaluate_map(
    model: RTDetrForObjectDetection,
    dataset,
    image_processor: RTDetrImageProcessor,
    batch_size: int = 8,
    device: str | None = None,
) -> dict:
    """Compute COCO mAP metrics on a dataset.

    All predictions are kept (threshold=0) so that mAP is computed in a
    threshold-free manner as intended by the COCO evaluation protocol.

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
        results = image_processor.post_process_object_detection(outputs, target_sizes=orig_sizes, threshold=0.0)

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
def _collect_raw_predictions(
    model: RTDetrForObjectDetection,
    dataset,
    image_processor: RTDetrImageProcessor,
    batch_size: int = 8,
    device: str | None = None,
) -> list[dict]:
    """Run inference once and return raw per-image prediction data."""
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    model = model.to(device)
    model.eval()

    raw: list[dict] = []
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"]
        outputs = model(pixel_values=pixel_values)
        orig_sizes = torch.stack([lab["orig_size"] for lab in labels]).to(device)
        # Use threshold=0 to keep all predictions
        results = image_processor.post_process_object_detection(outputs, target_sizes=orig_sizes, threshold=0.0)

        for res, lab in zip(results, labels):
            orig_h, orig_w = lab["orig_size"]
            img_diag = float((orig_h**2 + orig_w**2) ** 0.5)

            pred_boxes = res["boxes"].cpu()
            pred_scores = res["scores"].cpu()
            pred_labels = res["labels"].cpu()
            pred_cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
            pred_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
            pred_centers = torch.stack([pred_cx, pred_cy], dim=-1) if pred_boxes.numel() > 0 else torch.zeros(0, 2)

            boxes_cxcywh = lab["boxes"]
            cx, cy, w, h = boxes_cxcywh.unbind(-1) if boxes_cxcywh.numel() > 0 else (torch.zeros(0),) * 4
            gt_cx = cx * orig_w
            gt_cy = cy * orig_h
            gt_centers = torch.stack([gt_cx, gt_cy], dim=-1) if boxes_cxcywh.numel() > 0 else torch.zeros(0, 2)
            gt_boxes_xyxy = (
                torch.stack(
                    [(cx - w / 2) * orig_w, (cy - h / 2) * orig_h, (cx + w / 2) * orig_w, (cy + h / 2) * orig_h],
                    dim=-1,
                )
                if boxes_cxcywh.numel() > 0
                else torch.zeros(0, 4)
            )
            gt_labels = lab["class_labels"].cpu()

            raw.append(
                {
                    "pred_boxes": pred_boxes,
                    "pred_scores": pred_scores,
                    "pred_labels": pred_labels,
                    "pred_centers": pred_centers,
                    "gt_boxes": gt_boxes_xyxy.cpu(),
                    "gt_labels": gt_labels,
                    "gt_centers": gt_centers,
                    "img_diag": img_diag,
                }
            )

    return raw


def _compute_cd_metrics(
    raw: list[dict],
    score_threshold: float,
    distance_thresholds: list[float],
) -> dict:
    """Compute center-distance P/R/F1 from raw predictions at a given score threshold."""
    from scipy.optimize import linear_sum_assignment

    all_matches: dict[int, list[tuple[float, float]]] = {i: [] for i in ID_TO_CATEGORY}
    unmatched_gt: dict[int, int] = {i: 0 for i in ID_TO_CATEGORY}
    unmatched_pred: dict[int, int] = {i: 0 for i in ID_TO_CATEGORY}

    for img in raw:
        score_mask = img["pred_scores"] >= score_threshold
        pred_labels = img["pred_labels"][score_mask]
        pred_centers = img["pred_centers"][score_mask]
        gt_labels = img["gt_labels"]
        gt_centers = img["gt_centers"]
        img_diag = img["img_diag"]

        for cls_id in ID_TO_CATEGORY:
            gt_mask = gt_labels == cls_id
            pred_mask = pred_labels == cls_id

            g_centers = gt_centers[gt_mask] if gt_mask.any() else torch.zeros(0, 2)
            p_centers = pred_centers[pred_mask] if pred_mask.any() else torch.zeros(0, 2)

            n_gt = g_centers.shape[0]
            n_pred = p_centers.shape[0]

            if n_gt == 0:
                unmatched_pred[cls_id] += n_pred
                continue
            if n_pred == 0:
                unmatched_gt[cls_id] += n_gt
                continue

            cost = torch.cdist(g_centers.float(), p_centers.float())
            gt_idx, pred_idx = linear_sum_assignment(cost.numpy())

            matched = 0
            for gi, pi in zip(gt_idx, pred_idx):
                dist = float(cost[gi, pi])
                all_matches[cls_id].append((dist, img_diag))
                matched += 1

            unmatched_gt[cls_id] += n_gt - matched
            unmatched_pred[cls_id] += n_pred - matched

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
    """
    if distance_thresholds is None:
        distance_thresholds = [0.01, 0.02, 0.05]

    raw = _collect_raw_predictions(model, dataset, image_processor, batch_size, device)
    return _compute_cd_metrics(raw, threshold, distance_thresholds)


@torch.no_grad()
def sweep_confidence_threshold(
    model: RTDetrForObjectDetection,
    dataset,
    image_processor: RTDetrImageProcessor,
    batch_size: int = 8,
    distance_threshold: float = 0.02,
    score_thresholds: list[float] | None = None,
    device: str | None = None,
) -> dict:
    """Sweep confidence thresholds and compute center-distance P/R/F1 and mAP.

    Runs inference once, then evaluates at each score threshold.
    Returns score_thresholds, per_class P/R/F1, macro P/R/F1, map, map_50.
    """
    from torchmetrics.detection import MeanAveragePrecision

    if score_thresholds is None:
        score_thresholds = np.arange(0.01, 1.0, 0.01).tolist()

    raw = _collect_raw_predictions(model, dataset, image_processor, batch_size, device)

    per_class: dict[str, dict[str, list[float]]] = {
        name: {"precision": [], "recall": [], "f1": []} for name in ID_TO_CATEGORY.values()
    }
    map_values: list[float] = []
    map_50_values: list[float] = []

    dt_label = f"{distance_threshold:.0%}"
    for st in score_thresholds:
        cd = _compute_cd_metrics(raw, st, [distance_threshold])
        for name in ID_TO_CATEGORY.values():
            t = cd["per_class"][name]["thresholds"][dt_label]
            per_class[name]["precision"].append(t["precision"])
            per_class[name]["recall"].append(t["recall"])
            per_class[name]["f1"].append(t["f1"])

        metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            max_detection_thresholds=[1, 10, 400],
            backend="faster_coco_eval",
        )
        for img in raw:
            mask = img["pred_scores"] >= st
            preds = [
                {
                    "boxes": img["pred_boxes"][mask],
                    "scores": img["pred_scores"][mask],
                    "labels": img["pred_labels"][mask],
                }
            ]
            metric.update(preds, [{"boxes": img["gt_boxes"], "labels": img["gt_labels"]}])
        result = metric.compute()
        map_values.append(float(result["map"]))
        map_50_values.append(float(result["map_50"]))

    # Macro average
    class_names = list(ID_TO_CATEGORY.values())
    macro = {
        "precision": [
            float(np.mean([per_class[c]["precision"][i] for c in class_names])) for i in range(len(score_thresholds))
        ],
        "recall": [
            float(np.mean([per_class[c]["recall"][i] for c in class_names])) for i in range(len(score_thresholds))
        ],
        "f1": [float(np.mean([per_class[c]["f1"][i] for c in class_names])) for i in range(len(score_thresholds))],
    }

    return {
        "score_thresholds": score_thresholds,
        "distance_threshold": distance_threshold,
        "per_class": per_class,
        "macro": macro,
        "map": map_values,
        "map_50": map_50_values,
    }


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
