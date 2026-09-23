"""Corner selection by stone fit: a post-processing prototype beyond Kaya's pipeline (spec 004, lever 2).

Kaya keeps the 4 highest-scoring corner candidates, so one false corner breaks the board. Here the
detected stones arbitrate: every quad built from the best candidates (4 of them, or 3 completed into
a parallelogram) is scored by how well the stones fall on its grid, the best one replaces Kaya's
choice when it fits clearly better, and the result is refined by a least-squares homography from
the stones to their intersections.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from moku.board import (
    _UNIT_SQUARE,
    _complete_three_corners,
    apply_homography,
    compute_homography,
    fit_homography,
    order_corners,
)

TOP_K = 8  # corner candidates combined into quads
MIN_STONES = 8  # fewer stones cannot arbitrate between quads
MAX_RESIDUAL = 0.5  # cells: a stone's cost is its distance to the nearest intersection, clipped
RANK_PENALTY = 0.004  # per mean candidate rank: ties go to the highest-scoring corners
COMPLETED_PENALTY = 0.03  # a corner inferred from 3 others is less trusted than a detected one
KAYA_MARGIN = 0.02  # the stone fit must beat Kaya's quad by this much to replace it
INLIER_CELLS = 0.3  # stones used to refine the homography
MAX_REFINE_SHIFT = 1.0  # cells a refined corner may move


def grid_cost(corners: np.ndarray, stones: np.ndarray, board_size: int) -> float:
    """Mean clipped distance (cells) of the stones to the grid of ``corners``, plus collisions."""
    h = compute_homography(corners, _UNIT_SQUARE)
    if h is None:
        return np.inf
    grid = apply_homography(h, stones) * (board_size - 1)
    lattice = np.clip(np.round(grid), 0, board_size - 1)
    dist = np.minimum(np.hypot(*(grid - lattice).T), MAX_RESIDUAL)
    collisions = len(lattice) - len(np.unique(lattice, axis=0))
    return float(dist.mean() + MAX_RESIDUAL * collisions / len(stones))


def _is_convex(quad: np.ndarray) -> bool:
    edges = np.roll(quad, -1, axis=0) - quad
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    return bool((cross > 0).all() or (cross < 0).all())


def candidate_quads(candidates: np.ndarray, top_k: int = TOP_K) -> list[tuple[np.ndarray, float]]:
    """``(corners TL..BL, prior penalty)`` for every convex quad of 4, or 3 completed, candidates."""
    candidates = candidates[:top_k]
    quads = []
    for size, extra in ((4, 0.0), (3, COMPLETED_PENALTY)):
        for idx in combinations(range(len(candidates)), size):
            points = candidates[list(idx)]
            quad = order_corners(points if size == 4 else _complete_three_corners(points))
            if _is_convex(quad):
                quads.append((quad, RANK_PENALTY * float(np.mean(idx)) + extra))
    return quads


def refine_on_stones(corners: np.ndarray, stones: np.ndarray, board_size: int, iterations: int = 3) -> np.ndarray:
    """Least-squares homography from the stones to their intersections; ``corners`` if it does not help."""
    n = board_size
    h = compute_homography(corners, _UNIT_SQUARE)
    if h is None:
        return corners
    for _ in range(iterations):
        grid = apply_homography(h, stones) * (n - 1)
        lattice = np.round(grid)
        inlier = ((lattice >= 0) & (lattice <= n - 1)).all(axis=1) & (np.hypot(*(grid - lattice).T) < INLIER_CELLS)
        if inlier.sum() < MIN_STONES:
            return corners
        span = lattice[inlier].max(axis=0) - lattice[inlier].min(axis=0)
        if (span < (n - 1) / 2).any():  # too little of the board to extrapolate the corners
            return corners
        h = fit_homography(stones[inlier], lattice[inlier] / (n - 1))
        if h is None:
            return corners
    refined = apply_homography(np.linalg.inv(h), _UNIT_SQUARE)
    cell = np.mean(np.hypot(*(np.roll(corners, -1, axis=0) - corners).T)) / (n - 1)
    if np.hypot(*(refined - corners).T).max() > MAX_REFINE_SHIFT * cell:
        return corners
    if grid_cost(refined, stones, n) >= grid_cost(corners, stones, n):
        return corners
    return refined


def fit_corners(kaya: np.ndarray, candidates: np.ndarray, stones: np.ndarray, board_size: int) -> np.ndarray:
    """Board corners (TL..BL) chosen and refined on the stones; Kaya's when the stones are too few."""
    if len(stones) < MIN_STONES:
        return kaya
    best, best_cost = kaya, grid_cost(kaya, stones, board_size) - KAYA_MARGIN
    for quad, prior in candidate_quads(candidates):
        cost = grid_cost(quad, stones, board_size) + prior
        if cost < best_cost:
            best, best_cost = quad, cost
    return refine_on_stones(best, stones, board_size)
