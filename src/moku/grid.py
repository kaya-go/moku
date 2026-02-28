"""Stone grid inference from object detection annotations.

Maps detected bounding boxes (board corners + stones) onto a discrete Go board grid.
This module provides the geometric logic that bridges raw detection output
and the SGF coordinate system.

Uses perspective-corrected mapping via homography from 4 detected board corners.
"""

from __future__ import annotations

import numpy as np

from moku.dataset import CATEGORIES, _sort_corners_clockwise


def _compute_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Compute a 3x3 homography matrix mapping src points to dst points.

    Uses Direct Linear Transform (DLT) with 4 point correspondences.

    Args:
        src: (4, 2) array of source points.
        dst: (4, 2) array of destination points.

    Returns:
        (3, 3) homography matrix H such that dst ~ H @ src (in homogeneous coords).
    """
    A = []
    for (x, y), (xp, yp) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])
    A = np.array(A, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]


def _apply_homography(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply homography to an array of 2D points.

    Args:
        H: (3, 3) homography matrix.
        points: (N, 2) array of points.

    Returns:
        (N, 2) array of transformed points.
    """
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    pts_h = np.hstack([points, ones])  # (N, 3)
    transformed = (H @ pts_h.T).T  # (N, 3)
    return transformed[:, :2] / transformed[:, 2:3]


def annotations_to_grid(
    objects: dict,
    board_size: int = 19,
) -> np.ndarray:
    """Convert object detection annotations to a stone position grid.

    Extracts board_corner detections, sorts them (TL, TR, BR, BL), and uses
    a perspective homography to map stone centers to the rectified grid.

    Args:
        objects: The 'objects' dict from a dataset sample, with keys
                 'bbox' (list of [x, y, w, h]) and 'category' (list of int).
        board_size: Number of lines on the board (9, 13, or 19).

    Returns:
        A (board_size, board_size) int array: 0=empty, 1=black, 2=white.
        Returns all-zeros if fewer than 4 board corners are found.
    """
    grid = np.zeros((board_size, board_size), dtype=int)

    black_cat = CATEGORIES["black_stone"]
    white_cat = CATEGORIES["white_stone"]
    corner_cat = CATEGORIES["board_corner"]

    # Grid values: 0=empty, 1=black, 2=white (independent of category IDs)
    GRID_BLACK = 1
    GRID_WHITE = 2
    cat_to_grid = {black_cat: GRID_BLACK, white_cat: GRID_WHITE}

    # Collect stone centers and board corner centers
    stones: list[tuple[float, float, int]] = []
    corner_centers: list[list[float]] = []

    for bbox, cat_id in zip(objects["bbox"], objects["category"]):
        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2
        if cat_id in (black_cat, white_cat):
            stones.append((cx, cy, cat_to_grid[cat_id]))
        elif cat_id == corner_cat:
            corner_centers.append([cx, cy])

    if not stones or len(corner_centers) < 4:
        return grid

    # Sort corners: TL, TR, BR, BL
    corners = _sort_corners_clockwise(np.array(corner_centers, dtype=np.float64))

    stone_pts = np.array([(s[0], s[1]) for s in stones], dtype=np.float64)
    grid_vals = [s[2] for s in stones]

    # Map board corners to a unit square [0,1]x[0,1]
    dst_corners = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    )
    H = _compute_homography(corners, dst_corners)
    rectified = _apply_homography(H, stone_pts)

    for (rel_x, rel_y), gval in zip(rectified, grid_vals):
        col = int(np.clip(round(rel_x * (board_size - 1)), 0, board_size - 1))
        row = int(np.clip(round(rel_y * (board_size - 1)), 0, board_size - 1))
        grid[row, col] = gval

    return grid
