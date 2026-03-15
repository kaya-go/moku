"""Visualization utilities for moku datasets."""

from __future__ import annotations

import io
from collections import Counter

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from moku.dataset import CATEGORIES, ID_TO_CATEGORY
from moku.grid import annotations_to_grid

# High-contrast colors for bounding boxes
CATEGORY_COLORS = {
    0: "#e7298a",  # black_stone — magenta/pink
    1: "#1b9e77",  # white_stone — teal/green
    2: "#e6ab02",  # board_corner — yellow/gold
}

CATEGORY_LINEWIDTHS = {
    0: 2,
    1: 2,
    2: 2,
}

# Standard star point (hoshi) positions, 0-indexed
HOSHI_POINTS: dict[int, list[tuple[int, int]]] = {
    9: [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)],
    13: [(3, 3), (3, 6), (3, 9), (6, 3), (6, 6), (6, 9), (9, 3), (9, 6), (9, 9)],
    19: [
        (3, 3),
        (3, 9),
        (3, 15),
        (9, 3),
        (9, 9),
        (9, 15),
        (15, 3),
        (15, 9),
        (15, 15),
    ],
}


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


def sample_metadata_html(sample: dict, split: str, idx: int, split_size: int) -> str:
    """Return an HTML string summarizing sample metadata."""
    cats = sample["objects"]["category"]
    counts = Counter(cats)
    n_black = counts.get(0, 0)
    n_white = counts.get(1, 0)
    n_corners = counts.get(2, 0)
    total = len(cats)

    return (
        f"<div style='font-family: monospace; font-size: 14px; line-height: 1.6'>"
        f"<b>Split:</b> {split} &nbsp;|&nbsp; "
        f"<b>Index:</b> {idx} / {split_size - 1}<br>"
        f"<b>Source:</b> {sample['source_dataset']}<br>"
        f"<b>Image:</b> {sample['image'].width}&times;{sample['image'].height}<br>"
        f"<b>Objects:</b> {total} total &mdash; "
        f"<span style='color:{CATEGORY_COLORS[0]}'>&#9679; {n_black} black</span> &nbsp; "
        f"<span style='color:{CATEGORY_COLORS[1]}'>&#9679; {n_white} white</span> &nbsp; "
        f"<span style='color:{CATEGORY_COLORS[2]}'>&#9670; {n_corners} corners</span>"
        f"</div>"
    )


def browse_dataset(ds_dict) -> None:
    """Create an interactive ipywidgets browser for a DatasetDict.

    Provides prev/next buttons, a slider, a split dropdown,
    and a board size selector to navigate and inspect annotated samples
    with an inferred stone grid alongside the annotated photo.
    """
    import ipywidgets as widgets
    from IPython.display import display as ipy_display

    split_selector = widgets.Dropdown(
        options=list(ds_dict.keys()),
        value="train",
        description="Split:",
    )
    board_size_selector = widgets.Dropdown(
        options=[9, 13, 19],
        value=19,
        description="Board:",
    )
    index_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(ds_dict["train"]) - 1,
        description="Index:",
        layout=widgets.Layout(width="400px"),
    )
    prev_btn = widgets.Button(description="\u25c0 Prev", layout=widgets.Layout(width="80px"))
    next_btn = widgets.Button(description="Next \u25b6", layout=widgets.Layout(width="80px"))

    image_out = widgets.Image(
        format="png",
        layout=widgets.Layout(max_width="1200px", max_height="600px"),
    )
    info_html = widgets.HTML()

    def _update(_=None):
        split = split_selector.value
        idx = index_slider.value
        sample = ds_dict[split][idx]
        board_size = board_size_selector.value

        image_out.value = render_sample_with_grid(sample, board_size=board_size)
        info_html.value = sample_metadata_html(sample, split, idx, len(ds_dict[split]))

    def _on_split_change(change):
        index_slider.max = len(ds_dict[change["new"]]) - 1
        index_slider.value = 0

    def _on_prev(_):
        if index_slider.value > 0:
            index_slider.value -= 1

    def _on_next(_):
        if index_slider.value < index_slider.max:
            index_slider.value += 1

    split_selector.observe(_on_split_change, names="value")
    board_size_selector.observe(_update, names="value")
    index_slider.observe(_update, names="value")
    prev_btn.on_click(_on_prev)
    next_btn.on_click(_on_next)

    nav = widgets.HBox([prev_btn, index_slider, next_btn])
    controls = widgets.HBox([split_selector, board_size_selector])
    ui = widgets.VBox([controls, nav, info_html, image_out])

    _update()
    ipy_display(ui)


# ---------------------------------------------------------------------------
# mAP bar-chart helpers
# ---------------------------------------------------------------------------

_METRIC_KEYS = ["map", "map_50", "map_75", "mar_400"]
_METRIC_LABELS = ["mAP@50:95", "mAP@50", "mAP@75", "mAR@400"]
_DEFAULT_COLORS = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#666666",
]


def extract_map_summary(metrics: dict) -> dict:
    """Extract key mAP values from a torchmetrics result dict.

    Returns a dict with keys ``overall`` (list of floats for the 4 main
    metrics) and ``per_class`` (list of per-class AP@50:95 floats).
    """
    overall = [float(metrics.get(k, 0)) for k in _METRIC_KEYS]
    per_class_raw = metrics.get("map_per_class")
    per_class: list[float] = []
    if per_class_raw is not None and hasattr(per_class_raw, "__iter__"):
        per_class = [float(v.item() if hasattr(v, "item") else v) for v in per_class_raw]
    return {"overall": overall, "per_class": per_class}


def plot_map_comparison(
    runs: dict[str, dict],
    title: str = "mAP Comparison",
    figsize: tuple[float, float] = (10, 4),
) -> None:
    """Plot grouped bar charts comparing mAP metrics across runs.

    Args:
        runs: ``{label: metrics_dict}`` where *metrics_dict* is the raw
            output of ``evaluate_map`` (torchmetrics).
        title: suptitle for the figure.
        figsize: matplotlib figure size.
    """
    summaries = {label: extract_map_summary(m) for label, m in runs.items()}
    labels = list(summaries.keys())
    n = len(labels)
    colors = _DEFAULT_COLORS[:n]
    class_names = [ID_TO_CATEGORY[i] for i in range(len(CATEGORIES))]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Overall metrics ---
    x = np.arange(len(_METRIC_LABELS))
    total_width = 0.8
    gap = 0.05
    w = (total_width - (n - 1) * gap) / n
    for i, label in enumerate(labels):
        offset = -total_width / 2 + w / 2 + i * (w + gap)
        vals = summaries[label]["overall"]
        bars = axes[0].bar(x + offset, vals, w, label=label, color=colors[i])
        axes[0].bar_label(bars, fmt="%.3f", fontsize=max(6, 9 - n), padding=2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(_METRIC_LABELS)
    axes[0].set_ylim(0, 1.15)
    axes[0].set_title(f"{title} — Overall")
    axes[0].legend(fontsize=8)

    # --- Per-class AP ---
    has_per_class = all(len(s["per_class"]) == len(class_names) for s in summaries.values())
    if has_per_class:
        x2 = np.arange(len(class_names))
        for i, label in enumerate(labels):
            offset = -total_width / 2 + w / 2 + i * (w + gap)
            vals = summaries[label]["per_class"]
            bars = axes[1].bar(x2 + offset, vals, w, label=label, color=colors[i])
            axes[1].bar_label(bars, fmt="%.3f", fontsize=max(6, 9 - n), padding=2)
        axes[1].set_xticks(x2)
        axes[1].set_xticklabels(class_names)
        axes[1].set_ylim(0, 1.15)
        axes[1].set_title(f"{title} — Per-Class AP@50:95")
        axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.show()
