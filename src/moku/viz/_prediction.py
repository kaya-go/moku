"""Render what the Kaya pipeline makes of one prediction."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from moku.board import CORNER, KAYA_STONE_THRESHOLD, decode_queries, reconstruct_board, truth_board
from moku.viz._render import render_grid


def render_board_prediction(
    image: Image.Image,
    pred,
    target,
    threshold: float = KAYA_STONE_THRESHOLD,
    title: str = "",
) -> plt.Figure:
    """Photo with detections and the corners Kaya picks, next to predicted and true positions.

    ``pred`` is a :class:`~moku.inference.RawPrediction`, ``target`` a
    :class:`~moku.evaluation.Target`. Green circles are ground-truth corners,
    red crosses the 4 best corner detections, yellow crosses the runners-up.
    """
    truth = truth_board(target.bboxes, target.categories)
    board_size = truth.board_size if truth is not None else 19
    dets = decode_queries(pred.logits, pred.boxes, pred.width, pred.height)
    result = reconstruct_board(pred.logits, pred.boxes, pred.width, pred.height, board_size, threshold)

    fig, (ax_img, ax_pred, ax_true) = plt.subplots(1, 3, figsize=(18, 6), width_ratios=[1.4, 1, 1])
    ax_img.imshow(image.convert("RGB"))
    if truth is not None:
        ax_img.scatter(*truth.corners.T, s=160, facecolors="none", edgecolors="lime", linewidths=2)
    corner = dets.classes == CORNER
    order = np.argsort(-dets.scores[corner])[:8]
    for rank, j in enumerate(order):
        (x, y), score = dets.centers[corner][j], dets.scores[corner][j]
        color = "red" if rank < 4 else "yellow"
        ax_img.scatter(x, y, marker="x", s=70, c=color)
        ax_img.annotate(f"{score:.2f}", (x, y), color=color, fontsize=7, xytext=(3, 3), textcoords="offset points")
    stones = (~corner) & (dets.scores >= threshold)
    colors = np.where(dets.classes[stones] == 0, "black", "white")
    ax_img.scatter(*dets.centers[stones].T, s=8, c=colors, edgecolors="magenta", linewidths=0.5)
    ax_img.plot(*np.vstack([result.corners, result.corners[:1]]).T, c="red", lw=1)
    ax_img.set_title(title or f"{stones.sum()} stones ≥ {threshold}", fontsize=9)
    ax_img.axis("off")

    render_grid(result.grid, ax=ax_pred)
    ax_pred.set_title("Kaya pipeline", fontsize=9)
    if truth is not None:
        render_grid(truth.grid, ax=ax_true)
        ax_true.set_title("Ground truth", fontsize=9)
    else:
        ax_true.axis("off")
    fig.tight_layout()
    return fig
