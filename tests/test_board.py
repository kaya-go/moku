"""The Kaya pipeline port must read a perfect detection back into the exact position."""

import numpy as np
import pytest

from moku.board import (
    BLACK,
    CORNER,
    WHITE,
    apply_homography,
    compare_boards,
    compute_homography,
    decode_queries,
    js_round,
    order_corners,
    reconstruct_board,
    select_corners,
    truth_board,
)

WIDTH, HEIGHT = 640, 480
# A tilted, perspective-distorted board (TL, TR, BR, BL) in image pixels.
CORNERS = np.array([[120.0, 60.0], [540.0, 90.0], [600.0, 430.0], [70.0, 400.0]])


def _scene(board_size: int, seed: int = 0) -> tuple[np.ndarray, list, list]:
    """A random position and its COCO annotations under the ``CORNERS`` perspective."""
    rng = np.random.default_rng(seed)
    grid = rng.choice([0, 1, 2], size=(board_size, board_size), p=[0.6, 0.2, 0.2]).astype(np.int8)
    to_image = compute_homography(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]), CORNERS)
    rows, cols = np.nonzero(grid)
    centers = apply_homography(to_image, np.stack([cols, rows], axis=1) / (board_size - 1))
    bboxes, cats = [], []
    for (x, y), cell in zip(centers, grid[rows, cols]):
        bboxes.append([x - 6, y - 6, 12, 12])
        cats.append(BLACK if cell == 1 else WHITE)
    for x, y in CORNERS:
        bboxes.append([x - 8, y - 8, 16, 16])
        cats.append(CORNER)
    return grid, bboxes, cats


def _perfect_outputs(bboxes: list, cats: list, n_queries: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Detector outputs that reproduce the annotations with confident scores; spare queries are empty."""
    logits = np.full((n_queries, 3), -8.0)
    boxes = np.full((n_queries, 4), 0.01)
    for q, (bbox, cat) in enumerate(zip(bboxes, cats)):
        x, y, w, h = bbox
        logits[q, cat] = 4.0
        boxes[q] = [(x + w / 2) / WIDTH, (y + h / 2) / HEIGHT, w / WIDTH, h / HEIGHT]
    return logits, boxes


@pytest.mark.parametrize("board_size", [9, 13, 19])
def test_truth_board_recovers_position_and_size(board_size):
    grid, bboxes, cats = _scene(board_size)
    truth = truth_board(bboxes, cats)
    assert truth.board_size == board_size
    assert truth.size_inferred
    assert truth.collisions == truth.out_of_bounds == 0
    np.testing.assert_array_equal(truth.grid, grid)


@pytest.mark.parametrize("board_size", [9, 19])
def test_perfect_detection_reconstructs_exact_position(board_size):
    grid, bboxes, cats = _scene(board_size, seed=1)
    logits, boxes = _perfect_outputs(bboxes, cats)
    result = reconstruct_board(logits, boxes, WIDTH, HEIGHT, board_size)
    assert result.corners_detected and result.n_corner_candidates == 4
    assert compare_boards(result.grid, grid)["errors"] == 0


def test_rotated_prediction_is_not_an_error():
    grid, *_ = _scene(19, seed=2)
    assert compare_boards(np.rot90(grid), grid)["errors"] == 0


def test_missing_corner_is_completed_as_parallelogram():
    _, bboxes, cats = _scene(19)
    logits, boxes = _perfect_outputs(bboxes[:-1], cats[:-1])  # drop the last (BL) corner
    corners, detected, n = select_corners(decode_queries(logits, boxes, WIDTH, HEIGHT), WIDTH, HEIGHT)
    assert detected and n == 3
    expected = CORNERS[0] + CORNERS[2] - CORNERS[1]
    assert np.allclose(corners[3], expected, atol=1e-6)


def test_near_duplicate_corners_are_merged():
    _, bboxes, cats = _scene(9)
    bboxes = bboxes + [[CORNERS[0][0] - 5, CORNERS[0][1] - 5, 16, 16]]  # 3 px from the TL corner
    cats = cats + [CORNER]
    logits, boxes = _perfect_outputs(bboxes, cats)
    _, _, n = select_corners(decode_queries(logits, boxes, WIDTH, HEIGHT), WIDTH, HEIGHT)
    assert n == 4


def test_order_corners_and_js_round():
    shuffled = CORNERS[[2, 0, 3, 1]]
    np.testing.assert_array_equal(order_corners(shuffled), CORNERS)
    np.testing.assert_array_equal(js_round(np.array([0.5, 1.5, -0.5, 2.49])), [1, 2, 0, 2])


def test_fit_homography_recovers_exact_mapping():
    from moku.board import fit_homography

    unit = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    h = compute_homography(CORNERS, unit)
    points = np.random.default_rng(0).uniform(100, 500, size=(30, 2))
    fitted = fit_homography(points, apply_homography(h, points))
    np.testing.assert_allclose(apply_homography(fitted, CORNERS), unit, atol=1e-6)


def test_refine_corners_moves_a_misplaced_corner_back():
    from moku.annotations import refine_corners

    grid, bboxes, cats = _scene(19, seed=3)
    moved = [list(b) for b in bboxes]
    corner_idx = [i for i, c in enumerate(cats) if c == CORNER]
    moved[corner_idx[0]][0] += 8  # ~0.3 cell off
    result = refine_corners(moved, cats)
    assert result is not None
    fixed, shift = result
    np.testing.assert_allclose(fixed[corner_idx[0]], bboxes[corner_idx[0]], atol=0.5)
    assert 0.1 < shift < 0.6


def test_stone_fit_rejects_a_confident_false_corner():
    grid, bboxes, cats = _scene(19, seed=3)
    # A false corner out-scores the true BL corner: Kaya keeps it and breaks the board.
    logits, boxes = _perfect_outputs(bboxes + [[300 - 8, 250 - 8, 16, 16]], cats + [CORNER])
    logits[len(bboxes), CORNER] = 6.0
    kaya = reconstruct_board(logits, boxes, WIDTH, HEIGHT, 19)
    fit = reconstruct_board(logits, boxes, WIDTH, HEIGHT, 19, corner_method="fit")
    assert compare_boards(kaya.grid, grid)["errors"] > 0
    assert compare_boards(fit.grid, grid)["errors"] == 0
    np.testing.assert_allclose(fit.corners, CORNERS, atol=1.0)


def test_stone_fit_keeps_correct_corners():
    grid, bboxes, cats = _scene(13, seed=4)
    logits, boxes = _perfect_outputs(bboxes, cats)
    fit = reconstruct_board(logits, boxes, WIDTH, HEIGHT, 13, corner_method="fit")
    assert compare_boards(fit.grid, grid)["errors"] == 0
    np.testing.assert_allclose(fit.corners, CORNERS, atol=1.0)
