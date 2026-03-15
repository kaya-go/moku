"""Interactive dataset browser widget."""

from __future__ import annotations

from collections import Counter

from moku.viz._constants import CATEGORY_COLORS
from moku.viz._render import render_sample_with_grid


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
