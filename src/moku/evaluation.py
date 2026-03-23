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


def _compute_cd_ap(
    raw: list[dict],
    distance_thresholds: list[float] | None = None,
) -> dict:
    """Compute center-distance Average Precision (threshold-free).

    Like COCO mAP but uses center-distance matching instead of IoU.
    All predictions are ranked by confidence; the P-R curve is built
    by sweeping the implicit confidence threshold.

    Args:
        raw: Output of ``_collect_raw_predictions()``.
        distance_thresholds: Fractions of image diagonal for matching.
            Defaults to ``[0.01, 0.02, 0.05]``.

    Returns dict with:
        - per_class: {class_name: {"cdAP@1%": ..., "cdAP@2%": ..., ...}}
        - macro: {"cdAP@1%": ..., ...} averaged over classes
        - optimal_thresholds: {class_name: {threshold, f1, precision, recall}}
          at the middle distance threshold (2% by default)
    """
    if distance_thresholds is None:
        distance_thresholds = [0.01, 0.02, 0.05]

    # Primary distance threshold used for optimal-threshold extraction
    primary_dt = distance_thresholds[len(distance_thresholds) // 2]

    per_class_results: dict[str, dict[str, float]] = {}
    optimal_thresholds: dict[str, dict] = {}

    for cls_id, cls_name in ID_TO_CATEGORY.items():
        per_dt: dict[str, float] = {}

        for dt in distance_thresholds:
            # ── Per-image greedy matching (score-ordered) ─────────────
            all_detections: list[tuple[float, bool]] = []  # (score, is_tp)
            total_gt = 0

            for img in raw:
                gt_mask = img["gt_labels"] == cls_id
                pred_mask = img["pred_labels"] == cls_id

                g_centers = img["gt_centers"][gt_mask] if gt_mask.any() else torch.zeros(0, 2)
                p_centers = img["pred_centers"][pred_mask] if pred_mask.any() else torch.zeros(0, 2)
                p_scores = img["pred_scores"][pred_mask] if pred_mask.any() else torch.zeros(0)
                img_diag = img["img_diag"]

                n_gt = g_centers.shape[0]
                total_gt += n_gt

                if p_centers.shape[0] == 0:
                    continue

                if n_gt == 0:
                    for s in p_scores.tolist():
                        all_detections.append((s, False))
                    continue

                # Sort predictions by score descending for greedy matching
                sorted_idx = torch.argsort(p_scores, descending=True)
                p_centers = p_centers[sorted_idx]
                p_scores = p_scores[sorted_idx]

                dists = torch.cdist(p_centers.float(), g_centers.float())
                matched_gt: set[int] = set()

                for pi in range(p_centers.shape[0]):
                    score = float(p_scores[pi])
                    # Find closest unmatched GT
                    best_gi = -1
                    best_dist = float("inf")
                    for gi in range(n_gt):
                        if gi in matched_gt:
                            continue
                        d = float(dists[pi, gi])
                        if d < best_dist:
                            best_dist = d
                            best_gi = gi

                    if best_gi >= 0 and best_dist / img_diag <= dt:
                        all_detections.append((score, True))
                        matched_gt.add(best_gi)
                    else:
                        all_detections.append((score, False))

            if total_gt == 0:
                per_dt[f"cdAP@{dt:.0%}"] = 0.0
                continue

            # ── Build P-R curve ───────────────────────────────────────
            all_detections.sort(key=lambda x: -x[0])

            tp_cum = 0
            fp_cum = 0
            precisions = np.empty(len(all_detections))
            recalls = np.empty(len(all_detections))
            scores_arr = np.empty(len(all_detections))

            for i, (score, is_tp) in enumerate(all_detections):
                if is_tp:
                    tp_cum += 1
                else:
                    fp_cum += 1
                precisions[i] = tp_cum / (tp_cum + fp_cum)
                recalls[i] = tp_cum / total_gt
                scores_arr[i] = score

            # ── 101-point interpolated AP (COCO style) ────────────────
            ap = 0.0
            for t in np.linspace(0, 1, 101):
                mask = recalls >= t
                if mask.any():
                    ap += precisions[mask].max()
            ap /= 101.0

            per_dt[f"cdAP@{dt:.0%}"] = round(float(ap), 4)

            # ── Optimal F1 threshold at the primary distance threshold ─
            if dt == primary_dt:
                denom = precisions + recalls
                f1_values = np.zeros_like(denom)
                nonzero = denom > 0
                f1_values[nonzero] = 2 * precisions[nonzero] * recalls[nonzero] / denom[nonzero]
                best_idx = int(np.argmax(f1_values))
                optimal_thresholds[cls_name] = {
                    "threshold": round(float(scores_arr[best_idx]), 4),
                    "f1": round(float(f1_values[best_idx]), 4),
                    "precision": round(float(precisions[best_idx]), 4),
                    "recall": round(float(recalls[best_idx]), 4),
                }

        per_class_results[cls_name] = per_dt

    # Macro average
    all_keys = list(next(iter(per_class_results.values())).keys())
    macro = {k: round(float(np.mean([per_class_results[c][k] for c in per_class_results])), 4) for k in all_keys}

    return {
        "per_class": per_class_results,
        "macro": macro,
        "optimal_thresholds": optimal_thresholds,
    }


def _compute_corner_recall_at_k(
    raw: list[dict],
    k: int = 4,
    dist_threshold: float = 0.02,
) -> float:
    """Corner recall using the top-*k* predictions per image.

    For each image the *k* most confident ``board_corner`` predictions are
    greedily matched to GT corners (within *dist_threshold* of the image
    diagonal).  Recall = n_matched / n_gt, averaged across images that have
    at least one GT corner.

    This mirrors inference behaviour where only the top-4 corners are kept.
    """
    CORNER_CLS_ID = 2  # board_corner
    recalls: list[float] = []

    for img in raw:
        gt_mask = img["gt_labels"] == CORNER_CLS_ID
        pred_mask = img["pred_labels"] == CORNER_CLS_ID

        g_centers = img["gt_centers"][gt_mask] if gt_mask.any() else torch.zeros(0, 2)
        n_gt = g_centers.shape[0]
        if n_gt == 0:
            continue  # skip images with no GT corners

        p_centers = img["pred_centers"][pred_mask] if pred_mask.any() else torch.zeros(0, 2)
        p_scores = img["pred_scores"][pred_mask] if pred_mask.any() else torch.zeros(0)
        img_diag = img["img_diag"]

        if p_centers.shape[0] == 0:
            recalls.append(0.0)
            continue

        # Take top-k by confidence
        topk = min(k, p_centers.shape[0])
        topk_idx = torch.argsort(p_scores, descending=True)[:topk]
        p_centers = p_centers[topk_idx]

        # Greedy matching
        dists = torch.cdist(p_centers.float(), g_centers.float())
        matched_gt: set[int] = set()
        n_matched = 0

        for pi in range(p_centers.shape[0]):
            best_gi = -1
            best_dist = float("inf")
            for gi in range(n_gt):
                if gi in matched_gt:
                    continue
                d = float(dists[pi, gi])
                if d < best_dist:
                    best_dist = d
                    best_gi = gi

            if best_gi >= 0 and best_dist / img_diag <= dist_threshold:
                matched_gt.add(best_gi)
                n_matched += 1

        recalls.append(n_matched / n_gt)

    return round(float(np.mean(recalls)) if recalls else 0.0, 4)


@torch.no_grad()
def evaluate_cd_ap(
    model: RTDetrForObjectDetection,
    dataset,
    image_processor: RTDetrImageProcessor,
    batch_size: int = 8,
    distance_thresholds: list[float] | None = None,
    device: str | None = None,
) -> dict:
    """Compute center-distance AP and corner recall on a dataset.

    Returns dict with:
    - per_class cdAP, macro cdAP, and optimal F1 thresholds
      (see ``_compute_cd_ap`` for details)
    - ``corner_R4``: top-4 corner recall (matches inference behaviour)
    """
    raw = _collect_raw_predictions(model, dataset, image_processor, batch_size, device)
    result = _compute_cd_ap(raw, distance_thresholds)
    result["corner_R4"] = _compute_corner_recall_at_k(raw, k=4)
    return result
