"""Visualization utilities for moku datasets."""

from __future__ import annotations

import io
from collections import Counter

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from moku.dataset import ID_TO_CATEGORY

# High-contrast colors for bounding boxes
CATEGORY_COLORS = {
    0: "#d95f02",  # board — orange
    1: "#e7298a",  # black_stone — magenta/pink
    2: "#1b9e77",  # white_stone — teal/green
}

CATEGORY_LINEWIDTHS = {
    0: 2,
    1: 2,
    2: 2,
}


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


def sample_metadata_html(sample: dict, split: str, idx: int, split_size: int) -> str:
    """Return an HTML string summarizing sample metadata."""
    cats = sample["objects"]["category"]
    counts = Counter(cats)
    n_black = counts.get(1, 0)
    n_white = counts.get(2, 0)
    n_board = counts.get(0, 0)
    total = len(cats)

    return (
        f"<div style='font-family: monospace; font-size: 14px; line-height: 1.6'>"
        f"<b>Split:</b> {split} &nbsp;|&nbsp; "
        f"<b>Index:</b> {idx} / {split_size - 1}<br>"
        f"<b>Source:</b> {sample['source_dataset']}<br>"
        f"<b>Image:</b> {sample['image'].width}&times;{sample['image'].height}<br>"
        f"<b>Objects:</b> {total} total &mdash; "
        f"<span style='color:{CATEGORY_COLORS[1]}'>&#9679; {n_black} black</span> &nbsp; "
        f"<span style='color:{CATEGORY_COLORS[2]}'>&#9679; {n_white} white</span> &nbsp; "
        f"<span style='color:{CATEGORY_COLORS[0]}'>&#9632; {n_board} board</span>"
        f"</div>"
    )


def browse_dataset(ds_dict) -> None:
    """Create an interactive ipywidgets browser for a DatasetDict.

    Provides prev/next buttons, a slider, and a split dropdown
    to navigate and inspect annotated samples.
    """
    import ipywidgets as widgets
    from IPython.display import display as ipy_display

    split_selector = widgets.Dropdown(
        options=list(ds_dict.keys()),
        value="train",
        description="Split:",
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
        layout=widgets.Layout(max_width="800px", max_height="600px"),
    )
    info_html = widgets.HTML()

    def _update(_=None):
        split = split_selector.value
        idx = index_slider.value
        sample = ds_dict[split][idx]

        image_out.value = render_sample(sample)
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
    index_slider.observe(_update, names="value")
    prev_btn.on_click(_on_prev)
    next_btn.on_click(_on_next)

    nav = widgets.HBox([prev_btn, index_slider, next_btn])
    ui = widgets.VBox([split_selector, nav, info_html, image_out])

    _update()
    ipy_display(ui)
