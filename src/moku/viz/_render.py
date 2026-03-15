"""Rendering utilities for moku datasets (grid, sample, sample+grid)."""

from __future__ import annotations

import io

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from moku.dataset import ID_TO_CATEGORY
from moku.grid import annotations_to_grid
from moku.viz._constants import CATEGORY_COLORS, CATEGORY_LINEWIDTHS, HOSHI_POINTS


def render_grid(grid: np.ndarray, ax: plt.Axes | None = None) -> plt.Axes:
    """Render a stone grid as a Go board diagram.

    Args:
        grid: (board_size, board_size) array with 0=empty, 1=black, 2=white.
        ax: Optional matplotlib axes. Created if None.

    Returns:
        The matplotlib axes.
    """
    board_size = grid.shape[0]

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.set_facecolor("#DCB35C")

    # Grid lines
    for i in range(board_size):
        ax.plot([0, board_size - 1], [i, i], color="black", linewidth=0.5)
        ax.plot([i, i], [0, board_size - 1], color="black", linewidth=0.5)

    # Star points
    for row, col in HOSHI_POINTS.get(board_size, []):
        ax.plot(col, row, "o", color="black", markersize=4, zorder=2)

    # Stones as circle patches for clean rendering
    stone_radius = 0.43
    for row in range(board_size):
        for col in range(board_size):
            if grid[row, col] == 1:  # black stone
                circle = patches.Circle(
                    (col, row),
                    stone_radius,
                    facecolor="black",
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=3,
                )
                ax.add_patch(circle)
            elif grid[row, col] == 2:  # white stone
                circle = patches.Circle(
                    (col, row),
                    stone_radius,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=3,
                )
                ax.add_patch(circle)

    ax.set_xlim(-0.6, board_size - 0.4)
    ax.set_ylim(board_size - 0.4, -0.6)  # invert y so (0,0) is top-left
    ax.set_aspect("equal")
    ax.set_xticks(range(board_size))
    ax.set_yticks(range(board_size))
    ax.tick_params(labelsize=6, length=0)
    ax.set_title(f"Inferred grid ({board_size}\u00d7{board_size})", fontsize=10)

    return ax


def render_sample(sample: dict, show_labels: bool = False, figsize: tuple = (10, 8), dpi: int = 100) -> bytes:
    """Render a dataset sample with bounding boxes and return PNG bytes.

    Args:
        sample: A single row from the harmonized HF dataset.
        show_labels: Whether to draw category labels on boxes.
        figsize: Matplotlib figure size.
        dpi: Output resolution.

    Returns:
        PNG image bytes.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(sample["image"])

    for bbox, cat_id in zip(sample["objects"]["bbox"], sample["objects"]["category"]):
        x, y, w, h = bbox
        color = CATEGORY_COLORS.get(cat_id, "red")
        lw = CATEGORY_LINEWIDTHS.get(cat_id, 2)
        rect = patches.Rectangle((x, y), w, h, linewidth=lw, edgecolor=color, facecolor="none")
        ax.add_patch(rect)

        if show_labels:
            label = ID_TO_CATEGORY[cat_id]
            ax.text(
                x,
                y - 2,
                label,
                fontsize=7,
                color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8),
            )

    ax.axis("off")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def render_sample_with_grid(
    sample: dict,
    board_size: int = 19,
    show_labels: bool = False,
    dpi: int = 100,
) -> bytes:
    """Render annotated photo and inferred stone grid side by side.

    Args:
        sample: A single row from the harmonized HF dataset.
        board_size: Number of lines on the board (9, 13, or 19).
        show_labels: Whether to draw category labels on boxes.
        dpi: Output resolution.

    Returns:
        PNG image bytes.
    """
    fig, (ax_img, ax_grid) = plt.subplots(1, 2, figsize=(16, 7))

    # Left panel: annotated photo
    ax_img.imshow(sample["image"])
    for bbox, cat_id in zip(sample["objects"]["bbox"], sample["objects"]["category"]):
        x, y, w, h = bbox
        color = CATEGORY_COLORS.get(cat_id, "red")
        lw = CATEGORY_LINEWIDTHS.get(cat_id, 2)
        rect = patches.Rectangle((x, y), w, h, linewidth=lw, edgecolor=color, facecolor="none")
        ax_img.add_patch(rect)
        if show_labels:
            label = ID_TO_CATEGORY[cat_id]
            ax_img.text(
                x,
                y - 2,
                label,
                fontsize=7,
                color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8),
            )

    ax_img.axis("off")
    ax_img.set_title("Annotated photo", fontsize=10)

    # Right panel: inferred grid
    grid = annotations_to_grid(sample["objects"], board_size=board_size)
    render_grid(grid, ax=ax_grid)

    # Add stone count comparison to grid title
    n_stones_ann = sum(1 for c in sample["objects"]["category"] if c in (0, 1))
    n_stones_grid = int(np.count_nonzero(grid))
    ax_grid.set_title(
        f"Inferred grid ({board_size}\u00d7{board_size}) \u2014 {n_stones_grid}/{n_stones_ann} stones mapped",
        fontsize=10,
    )

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf.read()
