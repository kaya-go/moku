"""Photo → board position: Kaya's post-processing pipeline, ported to Python.

The detector outputs, for every query, class logits and a normalized
``(cx, cy, w, h)`` box. Kaya (``packages/board-recognition/src/moku-postprocess.ts``)
turns them into a position in four steps, reproduced here exactly so the whole
pipeline can be scored offline:

1. Decode every query: sigmoid, argmax class, box centre in image pixels.
2. Pick the board corners: corner queries above a tiny score floor, highest
   first, near-duplicates dropped, then the top 4 — or a geometric completion
   when only 2 or 3 corners are found.
3. Order the corners TL → TR → BR → BL and fit a homography to the unit square.
4. Snap every stone above the stone threshold to its nearest intersection; the
   highest-scoring stone wins when two land on the same point.

Ground-truth positions are built from the annotations with the same geometry,
so a perfect detector scores zero errors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from moku.dataset import CATEGORIES

BLACK = CATEGORIES["black_stone"]
WHITE = CATEGORIES["white_stone"]
CORNER = CATEGORIES["board_corner"]

# Grid cell values (independent of category ids).
EMPTY_CELL, BLACK_CELL, WHITE_CELL = 0, 1, 2
_CELL_OF_CLASS = {BLACK: BLACK_CELL, WHITE: WHITE_CELL}

BOARD_SIZES = (9, 13, 19)

# Constants mirrored from Kaya's moku-postprocess.ts / corners.ts.
KAYA_STONE_THRESHOLD = 0.035
CORNER_MIN_SCORE = 0.005
CORNER_DEDUP_FRACTION = 0.05  # of the image diagonal
DEGENERATE_AREA_FRACTION = 0.02  # of the image area
COLLAPSE_FRACTION = 0.05  # of the image diagonal
FALLBACK_INSET = 0.05  # of the smaller image side

_UNIT_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


@dataclass
class Detections:
    """Decoded queries of one image, in original image pixels."""

    centers: np.ndarray  # (Q, 2) float
    classes: np.ndarray  # (Q,) int
    scores: np.ndarray  # (Q,) float


@dataclass
class BoardResult:
    """A reconstructed position and the corners it was read with."""

    grid: np.ndarray  # (N, N) int8: 0 empty, 1 black, 2 white
    corners: np.ndarray  # (4, 2) TL, TR, BR, BL in image pixels
    corners_detected: bool  # False when Kaya fell back to the image bounds
    n_corner_candidates: int  # corner detections kept after deduplication


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def decode_queries(logits: np.ndarray, boxes: np.ndarray, width: int, height: int) -> Detections:
    """Decode raw ``(Q, C)`` logits and ``(Q, 4)`` normalized cxcywh boxes like Kaya does."""
    probs = sigmoid(np.asarray(logits, dtype=np.float64))
    classes = probs.argmax(axis=1)
    scores = probs[np.arange(len(probs)), classes]
    boxes = np.asarray(boxes, dtype=np.float64)
    centers = np.stack([boxes[:, 0] * width, boxes[:, 1] * height], axis=1)
    return Detections(centers=centers, classes=classes, scores=scores)


def js_round(x: np.ndarray | float) -> np.ndarray:
    """``Math.round`` semantics (halves round towards +inf), unlike Python's banker's rounding."""
    return np.floor(np.asarray(x, dtype=np.float64) + 0.5)


def order_corners(points: np.ndarray) -> np.ndarray:
    """Order 4 points TL → TR → BR → BL (``orderCorners`` in corners.ts)."""
    points = np.asarray(points, dtype=np.float64)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles, kind="stable")]
    top_left = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    return np.roll(ordered, -top_left, axis=0)


def compute_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    """3×3 homography mapping 4 ``src`` points onto 4 ``dst`` points (h33 = 1)."""
    rows, rhs = [], []
    for (sx, sy), (dx, dy) in zip(src, dst):
        rows.append([sx, sy, 1, 0, 0, 0, -dx * sx, -dx * sy])
        rhs.append(dx)
        rows.append([0, 0, 0, sx, sy, 1, -dy * sx, -dy * sy])
        rhs.append(dy)
    a = np.array(rows, dtype=np.float64)
    if abs(np.linalg.det(a)) < 1e-12:
        return None
    h = np.linalg.solve(a, np.array(rhs, dtype=np.float64))
    return np.append(h, 1.0).reshape(3, 3)


def fit_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    """Least-squares homography from N ≥ 4 point pairs (normalized DLT). Not part of Kaya's pipeline."""
    src, dst = np.asarray(src, dtype=np.float64), np.asarray(dst, dtype=np.float64)
    if len(src) < 4:
        return None

    def normalizer(points: np.ndarray) -> np.ndarray:
        mean = points.mean(axis=0)
        scale = np.sqrt(2) / max(np.mean(np.hypot(*(points - mean).T)), 1e-12)
        return np.array([[scale, 0, -scale * mean[0]], [0, scale, -scale * mean[1]], [0, 0, 1]])

    ts, td = normalizer(src), normalizer(dst)
    s = apply_homography(ts, src)
    d = apply_homography(td, dst)
    rows = []
    for (x, y), (u, v) in zip(s, d):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, sing, vt = np.linalg.svd(np.asarray(rows))
    if sing[-2] < 1e-12:
        return None
    h = np.linalg.inv(td) @ vt[-1].reshape(3, 3) @ ts
    return h / h[2, 2]


def apply_homography(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.hstack([points, np.ones((len(points), 1))]) @ h.T
    w = homogeneous[:, 2:3]
    safe = np.abs(w) >= 1e-10
    return np.where(safe, homogeneous[:, :2] / np.where(safe, w, 1.0), points)


def inset_corners(width: int, height: int, fraction: float = FALLBACK_INSET) -> np.ndarray:
    m = min(width, height) * fraction
    return np.array([[m, m], [width - 1 - m, m], [width - 1 - m, height - 1 - m], [m, height - 1 - m]])


def _complete_two_corners(p1: np.ndarray, p2: np.ndarray, width: int, height: int) -> np.ndarray:
    """Infer a square board from 2 corners (diagonal or shared edge), preferring in-image quads."""
    mid = (p1 + p2) / 2
    dx, dy = p2 - p1
    hdx, hdy = dx / 2, dy / 2
    candidates = [
        np.array([p1, [mid[0] + hdy, mid[1] - hdx], p2, [mid[0] - hdy, mid[1] + hdx]]),
        np.array([p1, p2, [p2[0] - dy, p2[1] + dx], [p1[0] - dy, p1[1] + dx]]),
        np.array([p1, p2, [p2[0] + dy, p2[1] - dx], [p1[0] + dy, p1[1] - dx]]),
    ]
    best, best_score = candidates[0], -np.inf
    for quad in candidates:
        mx = np.minimum(quad[:, 0], width - quad[:, 0])
        my = np.minimum(quad[:, 1], height - quad[:, 1])
        score = ((mx >= 0) & (my >= 0)).sum() * 1e6 + (mx + my).sum()
        if score > best_score:
            best, best_score = quad, score
    return best


def _complete_three_corners(points: np.ndarray) -> np.ndarray:
    """Complete the parallelogram whose diagonals are closest in length."""
    best, best_score = None, np.inf
    for diag in range(3):
        a, b, c = points[diag], points[(diag + 1) % 3], points[(diag + 2) % 3]
        p4 = b + c - a
        score = abs(np.linalg.norm(a - p4) - np.linalg.norm(b - c))
        if score < best_score:
            best, best_score = np.array([a, b, c, p4]), score
    return best


def corner_candidates(dets: Detections, width: int, height: int) -> np.ndarray:
    """Corner detections above the score floor, highest first, near-duplicates dropped."""
    keep = (dets.classes == CORNER) & (dets.scores >= CORNER_MIN_SCORE)
    centers, scores = dets.centers[keep], dets.scores[keep]
    centers = centers[np.argsort(-scores, kind="stable")]

    min_dist = np.hypot(width, height) * CORNER_DEDUP_FRACTION
    kept: list[np.ndarray] = []
    for c in centers:
        if all(np.hypot(*(c - k)) >= min_dist for k in kept):
            kept.append(c)
    return np.array(kept).reshape(-1, 2)


def select_corners(dets: Detections, width: int, height: int) -> tuple[np.ndarray, bool, int]:
    """Kaya's corner selection. Returns ``(corners TL..BL, corners_detected, n_candidates)``."""
    kept = corner_candidates(dets, width, height)
    if len(kept) < 2:
        return inset_corners(width, height), False, len(kept)
    if len(kept) == 2:
        quad = _complete_two_corners(kept[0], kept[1], width, height)
    elif len(kept) == 3:
        quad = _complete_three_corners(kept)
    else:
        quad = kept[:4]
    corners = order_corners(quad)

    span = corners.max(axis=0) - corners.min(axis=0)
    if span[0] * span[1] < width * height * DEGENERATE_AREA_FRACTION:
        corners = inset_corners(width, height)
    diag = np.hypot(width, height)
    pairwise = np.hypot(*(corners[:, None, :] - corners[None, :, :]).transpose(2, 0, 1))
    if (pairwise[np.triu_indices(4, k=1)] < diag * COLLAPSE_FRACTION).any():
        corners = inset_corners(width, height)
    return corners, True, min(len(kept), 4)


def snap_to_grid(
    centers: np.ndarray,
    cells: np.ndarray,
    scores: np.ndarray,
    corners: np.ndarray,
    board_size: int,
) -> np.ndarray:
    """Map stone centres to intersections (``mapStonesToGrid``); best score wins collisions."""
    grid = np.zeros((board_size, board_size), dtype=np.int8)
    h = compute_homography(corners, _UNIT_SQUARE)
    if h is None or len(centers) == 0:
        return grid
    rectified = apply_homography(h, centers)
    cols = js_round(rectified[:, 0] * (board_size - 1)).astype(int)
    rows = js_round(rectified[:, 1] * (board_size - 1)).astype(int)
    occupied = np.zeros_like(grid, dtype=bool)
    for i in np.argsort(-scores, kind="stable"):
        r, c = rows[i], cols[i]
        if not (0 <= r < board_size and 0 <= c < board_size) or occupied[r, c]:
            continue
        occupied[r, c] = True
        grid[r, c] = cells[i]
    return grid


def reconstruct_board(
    logits: np.ndarray,
    boxes: np.ndarray,
    width: int,
    height: int,
    board_size: int,
    threshold: float = KAYA_STONE_THRESHOLD,
    corner_method: str = "kaya",
) -> BoardResult:
    """Full Kaya pipeline: raw detector outputs → position on a ``board_size`` grid.

    ``corner_method="fit"`` replaces Kaya's corner choice by the stone-fit prototype
    (:mod:`moku.corner_fit`), which is not in Kaya yet.
    """
    dets = decode_queries(logits, boxes, width, height)
    corners, detected, n_candidates = select_corners(dets, width, height)
    if not detected:
        grid = np.zeros((board_size, board_size), dtype=np.int8)
        return BoardResult(grid=grid, corners=corners, corners_detected=False, n_corner_candidates=n_candidates)
    stone = (dets.classes != CORNER) & (dets.scores >= threshold)
    if corner_method == "fit":
        from moku.corner_fit import fit_corners

        corners = fit_corners(corners, corner_candidates(dets, width, height), dets.centers[stone], board_size)
    elif corner_method != "kaya":
        raise ValueError(f"unknown corner method {corner_method!r}")
    cells = np.vectorize(_CELL_OF_CLASS.get)(dets.classes[stone]) if stone.any() else np.zeros(0, dtype=int)
    grid = snap_to_grid(dets.centers[stone], cells, dets.scores[stone], corners, board_size)
    return BoardResult(grid=grid, corners=corners, corners_detected=True, n_corner_candidates=n_candidates)


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


@dataclass
class TruthBoard:
    """Ground-truth position derived from the annotations of one image."""

    grid: np.ndarray  # (N, N) int8
    board_size: int
    corners: np.ndarray  # (4, 2) TL, TR, BR, BL
    snap_residual: float  # median distance of GT stones to their intersection, in cell units
    size_inferred: bool  # False when there are no stones to infer the board size from
    collisions: int  # GT stones sharing an intersection with another GT stone
    out_of_bounds: int  # GT stones falling outside the grid


def _snap_residual(rectified: np.ndarray, board_size: int) -> float:
    scaled = rectified * (board_size - 1)
    return float(np.median(np.hypot(*(scaled - np.round(scaled)).T)))


def truth_board(
    bboxes: list[list[float]],
    categories: list[int],
    board_size: int | None = None,
) -> TruthBoard | None:
    """Build the ground-truth position from COCO ``[x, y, w, h]`` annotations.

    Needs exactly 4 annotated corners (partial boards are skipped). The board
    size, when not given, is the one whose grid the annotated stones fit best.
    """
    bboxes_arr = np.asarray(bboxes, dtype=np.float64).reshape(-1, 4)
    cats = np.asarray(categories)
    centers = bboxes_arr[:, :2] + bboxes_arr[:, 2:] / 2
    if (cats == CORNER).sum() != 4:
        return None
    corners = order_corners(centers[cats == CORNER])
    h = compute_homography(corners, _UNIT_SQUARE)
    if h is None:
        return None
    is_stone = cats != CORNER
    rectified = apply_homography(h, centers[is_stone])
    cells = np.array([_CELL_OF_CLASS[c] for c in cats[is_stone]], dtype=np.int8)

    size_inferred = board_size is None and len(rectified) > 0
    if board_size is None:
        board_size = min(BOARD_SIZES, key=lambda n: _snap_residual(rectified, n)) if len(rectified) else 19
    residual = _snap_residual(rectified, board_size) if len(rectified) else 0.0

    grid = np.zeros((board_size, board_size), dtype=np.int8)
    cols = js_round(rectified[:, 0] * (board_size - 1)).astype(int)
    rows = js_round(rectified[:, 1] * (board_size - 1)).astype(int)
    collisions = out_of_bounds = 0
    for r, c, cell in zip(rows, cols, cells):
        if not (0 <= r < board_size and 0 <= c < board_size):
            out_of_bounds += 1
        elif grid[r, c]:
            collisions += 1
        else:
            grid[r, c] = cell
    return TruthBoard(
        grid=grid,
        board_size=board_size,
        corners=corners,
        snap_residual=residual,
        size_inferred=size_inferred,
        collisions=collisions,
        out_of_bounds=out_of_bounds,
    )


def compare_boards(pred: np.ndarray, truth: np.ndarray) -> dict[str, int]:
    """Count intersection errors, aligning the prediction on the best of its 4 rotations.

    A rotation only reflects which physical corner was called top-left, which
    both pipelines decide from the same image, so it is not an error.
    """
    best = None
    for k in range(4):
        rotated = np.rot90(pred, k)
        errors = int((rotated != truth).sum())
        if best is None or errors < best[0]:
            best = (errors, rotated)
    errors, aligned = best
    return {
        "errors": errors,
        "missed": int(((truth > 0) & (aligned == 0)).sum()),
        "extra": int(((truth == 0) & (aligned > 0)).sum()),
        "wrong_color": int(((truth > 0) & (aligned > 0) & (aligned != truth)).sum()),
        "n_truth_stones": int((truth > 0).sum()),
        "n_pred_stones": int((aligned > 0).sum()),
    }
