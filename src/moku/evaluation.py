"""Evaluation: detection metrics and board-level (end-to-end) metrics.

Everything is computed from :class:`~moku.inference.RawPrediction` objects, so
one inference pass feeds all metrics.

Detection metrics (threshold-free, per box):

- ``mAP@50`` / ``mAP``: COCO box AP at IoU 0.5 and averaged over 0.5:0.95;
- ``stone_cdAP``: centre-distance AP of stones, matched within 2% of the image diagonal;
- ``corner_R4``: recall of the 4 highest-scoring corners (Kaya keeps the top 4).

Board metrics — what the user actually gets. Kaya's pipeline (see
:mod:`moku.board`) reconstructs the position, which is compared intersection by
intersection with the position read from the ground-truth annotations:

- ``perfect``: share of boards read exactly — zero wrong intersections **and** the board located
  (no corner more than half a cell off). Without the second condition an empty board would count
  as perfect for a model that detects nothing, whatever its corners;
- ``errors``: mean number of wrong intersections per board;
- ``corner_fail``: share of boards with a corner more than half a grid cell off.

Test sets are small (tens of images, some being augmented copies of the same
photo), so board metrics come with bootstrap confidence intervals resampled
over photo clusters rather than images. A cluster is the position (up to
symmetry), or the ``source_dataset`` itself when it names a group such as
``gomrade/<game>`` (frames of one game are not independent).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from moku.board import (
    CORNER,
    KAYA_STONE_THRESHOLD,
    compare_boards,
    reconstruct_board,
    sigmoid,
    truth_board,
)
from moku.dataset import ID_TO_CATEGORY
from moku.inference import Detector, RawPrediction, run_detector

STONE_CLASSES = (0, 1)


@dataclass
class Target:
    """Ground-truth annotations of one image (COCO ``[x, y, w, h]`` boxes, pixels)."""

    bboxes: np.ndarray  # (N, 4)
    categories: np.ndarray  # (N,)
    width: int
    height: int
    source: str = ""

    @classmethod
    def from_example(cls, example: dict) -> Target:
        objects = example["objects"]
        return cls(
            bboxes=np.asarray(objects["bbox"], dtype=np.float64).reshape(-1, 4),
            categories=np.asarray(objects["category"], dtype=int),
            width=int(example["width"]),
            height=int(example["height"]),
            source=example.get("source_dataset", ""),
        )

    @property
    def centers(self) -> np.ndarray:
        return self.bboxes[:, :2] + self.bboxes[:, 2:] / 2


# ---------------------------------------------------------------------------
# Detection metrics
# ---------------------------------------------------------------------------


def top_detections(pred: RawPrediction, top_k: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``post_process_object_detection`` for sigmoid DETRs: top-k (query, class) pairs.

    Returns ``(boxes_xyxy_pixels, scores, labels)``.
    """
    probs = sigmoid(pred.logits)
    n_queries, n_classes = probs.shape
    top_k = top_k or n_queries
    flat = probs.reshape(-1)
    idx = np.argsort(-flat, kind="stable")[:top_k]
    scores, labels, queries = flat[idx], idx % n_classes, idx // n_classes
    cx, cy, w, h = pred.boxes[queries].T
    scale = np.array([pred.width, pred.height, pred.width, pred.height])
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1) * scale
    return boxes, scores, labels


def coco_map(preds: list[RawPrediction], targets: list[Target]) -> dict[str, float]:
    """COCO mAP (all detections kept; up to 400 per image since a 19×19 can hold 361 stones)."""
    from torchmetrics.detection import MeanAveragePrecision

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        max_detection_thresholds=[1, 10, 400],
        class_metrics=True,
        backend="faster_coco_eval",
    )
    for pred, target in zip(preds, targets):
        boxes, scores, labels = top_detections(pred)
        xyxy = np.concatenate([target.bboxes[:, :2], target.bboxes[:, :2] + target.bboxes[:, 2:]], axis=1)
        metric.update(
            [{"boxes": torch.tensor(boxes), "scores": torch.tensor(scores), "labels": torch.tensor(labels)}],
            [{"boxes": torch.tensor(xyxy), "labels": torch.tensor(target.categories)}],
        )
    result = metric.compute()
    out = {"mAP@50": float(result["map_50"]), "mAP": float(result["map"])}
    for cls_id, ap in zip(result["classes"].tolist(), result["map_per_class"].tolist()):
        out[f"AP/{ID_TO_CATEGORY[cls_id]}"] = float(ap)
    return out


def _greedy_matches(centers: np.ndarray, scores: np.ndarray, gt: np.ndarray, max_dist: float) -> np.ndarray:
    """Score-ordered greedy matching to the nearest unmatched GT. Returns a TP flag per prediction."""
    order = np.argsort(-scores, kind="stable")
    tp = np.zeros(len(centers), dtype=bool)
    if len(gt) == 0:
        return tp
    dists = np.hypot(*(centers[:, None, :] - gt[None, :, :]).transpose(2, 0, 1))
    free = np.ones(len(gt), dtype=bool)
    for i in order:
        if not free.any():
            break
        j = np.where(free, dists[i], np.inf).argmin()
        if dists[i, j] <= max_dist:
            tp[i] = True
            free[j] = False
    return tp


def _average_precision(scores: np.ndarray, tp: np.ndarray, n_gt: int) -> float:
    """101-point interpolated AP (COCO style)."""
    if n_gt == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    tp = tp[order]
    precision = np.cumsum(tp) / np.arange(1, len(tp) + 1)
    recall = np.cumsum(tp) / n_gt
    return float(
        np.mean([precision[recall >= t].max() if (recall >= t).any() else 0.0 for t in np.linspace(0, 1, 101)])
    )


def center_distance_ap(preds: list[RawPrediction], targets: list[Target], cls_id: int, frac: float = 0.02) -> float:
    """AP where a detection matches a GT object of the same class within ``frac`` of the diagonal."""
    all_scores, all_tp, n_gt = [], [], 0
    for pred, target in zip(preds, targets):
        boxes, scores, labels = top_detections(pred)
        keep = labels == cls_id
        centers = (boxes[keep, :2] + boxes[keep, 2:]) / 2
        gt = target.centers[target.categories == cls_id]
        n_gt += len(gt)
        tp = _greedy_matches(centers, scores[keep], gt, frac * np.hypot(target.width, target.height))
        all_scores.append(scores[keep])
        all_tp.append(tp)
    return _average_precision(np.concatenate(all_scores), np.concatenate(all_tp), n_gt)


def corner_recall_at_k(preds: list[RawPrediction], targets: list[Target], k: int = 4, frac: float = 0.02) -> float:
    """Mean per-image recall of GT corners by the ``k`` best corner detections."""
    recalls = []
    for pred, target in zip(preds, targets):
        gt = target.centers[target.categories == CORNER]
        if len(gt) == 0:
            continue
        boxes, scores, labels = top_detections(pred)
        keep = np.where(labels == CORNER)[0][:k]  # already sorted by score
        centers = (boxes[keep, :2] + boxes[keep, 2:]) / 2
        tp = _greedy_matches(centers, scores[keep], gt, frac * np.hypot(target.width, target.height))
        recalls.append(tp.sum() / len(gt))
    return float(np.mean(recalls)) if recalls else float("nan")


def true_positive_scores(preds: list[RawPrediction], targets: list[Target], frac: float = 0.02) -> dict[str, float]:
    """Calibration: median over GT objects of the best same-class score within ``frac`` of the diagonal.

    Kaya thresholds raw scores (0.035 for stones, a 0.005 floor for corners), so
    true objects scoring far above those values make the pipeline robust.
    """
    found: dict[str, list[float]] = {"stone": [], "corner": []}
    for pred, target in zip(preds, targets):
        probs = sigmoid(pred.logits)
        centers = pred.boxes[:, :2] * [pred.width, pred.height]
        radius = frac * np.hypot(target.width, target.height)
        for cls in (*STONE_CLASSES, CORNER):
            gt = target.centers[target.categories == cls]
            if len(gt) == 0:
                continue
            near = np.hypot(*(gt[:, None, :] - centers[None, :, :]).transpose(2, 0, 1)) <= radius
            best = np.where(near, probs[None, :, cls], 0.0).max(axis=1)
            found["corner" if cls == CORNER else "stone"].extend(best.tolist())
    return {f"{k}_tp_score": float(np.median(v)) if v else float("nan") for k, v in found.items()}


def detection_metrics(preds: list[RawPrediction], targets: list[Target]) -> dict[str, float]:
    metrics = coco_map(preds, targets)
    metrics["stone_cdAP"] = float(np.mean([center_distance_ap(preds, targets, c) for c in STONE_CLASSES]))
    metrics["corner_R4"] = corner_recall_at_k(preds, targets)
    metrics.update(true_positive_scores(preds, targets))
    return metrics


# ---------------------------------------------------------------------------
# Board metrics
# ---------------------------------------------------------------------------


def _corner_error_cells(pred_corners: np.ndarray, truth_corners: np.ndarray, board_size: int) -> float:
    """Worst corner error, in grid cells, after aligning which corner is called top-left."""
    span = np.mean([np.hypot(*(truth_corners[i] - truth_corners[(i + 1) % 4])) for i in range(4)])
    cell = span / (board_size - 1)
    best = min(np.hypot(*(np.roll(pred_corners, k, axis=0) - truth_corners).T).max() for k in range(4))
    return float(best / cell)


def _photo_cluster(truth_grid: np.ndarray, index: int) -> str:
    """Group augmented copies of one photo: same position up to rotation/flip.

    Empty boards carry no position to compare, so each stays its own cluster.
    """
    if not truth_grid.any():
        return f"empty-{index}"
    variants = [np.rot90(g, k) for g in (truth_grid, truth_grid.T) for k in range(4)]
    return min(v.tobytes().hex() for v in variants)


def board_table(
    preds: list[RawPrediction],
    targets: list[Target],
    threshold: float = KAYA_STONE_THRESHOLD,
) -> pd.DataFrame:
    """One row per image with a full ground-truth board (4 corners): Kaya pipeline vs truth."""
    rows = []
    for i, (pred, target) in enumerate(zip(preds, targets)):
        truth = truth_board(target.bboxes, target.categories)
        if truth is None:
            continue
        result = reconstruct_board(pred.logits, pred.boxes, pred.width, pred.height, truth.board_size, threshold)
        rows.append(
            {
                "index": i,
                "source": target.source,
                "board_size": truth.board_size,
                "cluster": target.source if "/" in target.source else _photo_cluster(truth.grid, i),
                "corners_found": result.n_corner_candidates,
                "corner_err_cells": _corner_error_cells(result.corners, truth.corners, truth.board_size),
                **compare_boards(result.grid, truth.grid),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_mean_ci(
    values: np.ndarray,
    clusters: np.ndarray,
    n_resamples: int = 5000,
    level: float = 0.9,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile CI of ``mean(values)``, resampling whole photo clusters."""
    rng = np.random.default_rng(seed)
    _, inverse = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inverse, weights=np.asarray(values, dtype=np.float64))
    counts = np.bincount(inverse).astype(np.float64)
    picks = rng.integers(0, len(sums), size=(n_resamples, len(sums)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    alpha = (1 - level) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


CORNER_FAIL_CELLS = 0.5


def is_perfect(table: pd.DataFrame) -> pd.Series:
    """Exact position and located board (see the module docstring)."""
    return (table["errors"] == 0) & (table["corner_err_cells"] <= CORNER_FAIL_CELLS)


def board_summary(table: pd.DataFrame, with_ci: bool = True) -> dict[str, float]:
    perfect = is_perfect(table).to_numpy(dtype=float)
    summary = {
        "boards": len(table),
        "photos": int(table["cluster"].nunique()),
        "empty": int((table["n_truth_stones"] == 0).sum()),
        "perfect": float(perfect.mean()),
        "errors": float(table["errors"].mean()),
        "le2_errors": float((table["errors"] <= 2).mean()),
        "corner_fail": float((table["corner_err_cells"] > CORNER_FAIL_CELLS).mean()),
    }
    if with_ci:
        summary["perfect_ci"] = bootstrap_mean_ci(perfect, table["cluster"].to_numpy())
        summary["errors_ci"] = bootstrap_mean_ci(table["errors"].to_numpy(), table["cluster"].to_numpy())
    return summary


def paired_difference(a: pd.DataFrame, b: pd.DataFrame, column: str = "errors") -> dict[str, float]:
    """Mean per-board difference ``b - a`` with a cluster-bootstrap CI (same boards, same order)."""
    delta = b[column].to_numpy(dtype=float) - a[column].to_numpy(dtype=float)
    return {"delta": float(delta.mean()), "ci": bootstrap_mean_ci(delta, a["cluster"].to_numpy())}


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    model: str
    split: str
    detection: dict[str, float]
    board: dict[str, float]
    boards: pd.DataFrame
    predictions: list[RawPrediction] = field(repr=False)
    targets: list[Target] = field(repr=False)


def logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def stone_offset(threshold: float) -> float:
    """Logit offset under which Kaya's fixed 0.035 threshold acts as ``threshold`` on the raw scores.

    Added to *every* class logit (a per-class offset could change Kaya's argmax), it can be baked
    into the ONNX (``moku export --logit-offset``) so that Kaya needs no per-model setting. The
    corner score floor (0.005) moves with it; corner ranking does not change.
    """
    return logit(KAYA_STONE_THRESHOLD) - logit(threshold)


def shift_logits(preds: list[RawPrediction], offset: float) -> list[RawPrediction]:
    if not offset:
        return preds
    return [RawPrediction(p.logits + offset, p.boxes, p.width, p.height) for p in preds]


def evaluate(
    detector: Detector,
    dataset_split,
    split: str = "test",
    threshold: float = KAYA_STONE_THRESHOLD,
    batch_size: int = 8,
    logit_offset: float = 0.0,
) -> EvalResult:
    """Run ``detector`` on a dataset split and compute detection + board metrics.

    ``logit_offset`` evaluates the model as if exported with that offset baked in.
    """
    targets = [Target.from_example(ex) for ex in dataset_split.remove_columns("image")]
    preds = shift_logits(run_detector(detector, (ex["image"] for ex in dataset_split), batch_size), logit_offset)
    boards = board_table(preds, targets, threshold)
    return EvalResult(
        model=detector.name,
        split=split,
        detection=detection_metrics(preds, targets),
        board=board_summary(boards),
        boards=boards,
        predictions=preds,
        targets=targets,
    )


SWEEP_THRESHOLDS = (0.005, 0.01, 0.015, 0.02, 0.025, 0.035, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5)


def threshold_sweep(result: EvalResult, thresholds=SWEEP_THRESHOLDS) -> pd.DataFrame:
    """Board metrics at several equivalent stone thresholds, applied as logit offsets (as shipped)."""
    rows = []
    for t in thresholds:
        offset = stone_offset(t)
        s = board_summary(board_table(shift_logits(result.predictions, offset), result.targets), with_ci=False)
        rows.append({"threshold": t, "offset": offset, **s})
    return pd.DataFrame(rows)


def best_offset(result: EvalResult, thresholds=SWEEP_THRESHOLDS) -> dict[str, float]:
    """Calibration fitted on ``result`` (validation): most perfect boards, then fewest errors."""
    sweep = threshold_sweep(result, thresholds)
    best = sweep.sort_values(["perfect", "errors"], ascending=[False, True]).iloc[0]
    return {"threshold": float(best["threshold"]), "offset": float(best["offset"])}
