"""Prediction rendering and interactive prediction browser."""

from __future__ import annotations

import io

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from moku.viz._constants import CATEGORY_COLORS, CATEGORY_LINEWIDTHS


def _get_raw_sample(ds_split, idx: int) -> dict:
    """Retrieve a raw (untransformed) sample from a HF Dataset split.

    If a transform has been set via ``set_transform``, it is temporarily
    cleared so that the returned dict contains the original columns
    (``image``, ``objects``, etc.).
    """
    saved_transform = ds_split._format_kwargs.get("transform")
    if saved_transform is not None:
        ds_split.reset_format()
    sample = ds_split[idx]
    if saved_transform is not None:
        ds_split.set_transform(saved_transform)
    return sample


def render_prediction(
    sample: dict,
    model,
    image_processor,
    threshold: float = 0.5,
    board_size: int = 19,
    dpi: int = 100,
) -> bytes:
    """Render ground-truth vs model prediction side-by-side and return PNG bytes.

    Left panel: ground-truth bounding boxes.
    Right panel: model predictions above the threshold.
    """
    import torch

    image = sample["image"].convert("RGB")

    # Run inference
    inputs = image_processor(images=[image], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    orig_size = torch.tensor([[image.height, image.width]], device=device)
    results = image_processor.post_process_object_detection(outputs, target_sizes=orig_size, threshold=threshold)[0]

    pred_boxes = results["boxes"].cpu().numpy()
    pred_labels = results["labels"].cpu().numpy()
    pred_scores = results["scores"].cpu().numpy()

    fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(16, 7))

    # --- GT panel ---
    ax_gt.imshow(image)
    for bbox, cat_id in zip(sample["objects"]["bbox"], sample["objects"]["category"]):
        x, y, w, h = bbox
        color = CATEGORY_COLORS.get(cat_id, "red")
        lw = CATEGORY_LINEWIDTHS.get(cat_id, 2)
        rect = patches.Rectangle((x, y), w, h, linewidth=lw, edgecolor=color, facecolor="none")
        ax_gt.add_patch(rect)
    n_gt = len(sample["objects"]["category"])
    ax_gt.set_title(f"Ground Truth ({n_gt} objects)", fontsize=10)
    ax_gt.axis("off")

    # --- Prediction panel ---
    ax_pred.imshow(image)
    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        x1, y1, x2, y2 = box
        cat_id = int(label)
        color = CATEGORY_COLORS.get(cat_id, "red")
        lw = CATEGORY_LINEWIDTHS.get(cat_id, 2)
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=lw, edgecolor=color, facecolor="none")
        ax_pred.add_patch(rect)
        ax_pred.text(
            x1,
            y1 - 2,
            f"{score:.2f}",
            fontsize=6,
            color="white",
            bbox=dict(boxstyle="round,pad=0.15", facecolor=color, alpha=0.7),
        )
    ax_pred.set_title(f"Predictions ({len(pred_boxes)} detections, thr={threshold})", fontsize=10)
    ax_pred.axis("off")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def browse_predictions(
    models: dict[str, dict],
    datasets: dict[str, object],
    threshold: float = 0.5,
) -> None:
    """Interactive widget to browse model predictions on dataset samples.

    Args:
        models: ``{label: {"model": RTDetrForObjectDetection,
                           "image_processor": RTDetrImageProcessor}}``
        datasets: ``{label: DatasetDict}`` — each value must support
                  ``ds[split][idx]`` access.
        threshold: Default confidence threshold for predictions.
    """
    import ipywidgets as widgets
    from IPython.display import display as ipy_display

    model_selector = widgets.Dropdown(
        options=list(models.keys()),
        description="Model:",
    )
    dataset_selector = widgets.Dropdown(
        options=list(datasets.keys()),
        description="Dataset:",
    )

    # Resolve initial splits
    first_ds_key = list(datasets.keys())[0]
    first_ds = datasets[first_ds_key]
    split_options = list(first_ds.keys())

    split_selector = widgets.Dropdown(
        options=split_options,
        value=split_options[0],
        description="Split:",
    )

    first_split_len = len(first_ds[split_options[0]])
    index_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=first_split_len - 1,
        description="Index:",
        layout=widgets.Layout(width="400px"),
    )
    threshold_slider = widgets.FloatSlider(
        value=threshold,
        min=0.0,
        max=1.0,
        step=0.01,
        description="Threshold:",
        layout=widgets.Layout(width="300px"),
        readout=False,
    )
    threshold_text = widgets.FloatText(
        value=threshold,
        min=0.0,
        max=1.0,
        step=0.01,
        layout=widgets.Layout(width="70px"),
    )
    widgets.link((threshold_slider, "value"), (threshold_text, "value"))

    prev_btn = widgets.Button(description="\u25c0 Prev", layout=widgets.Layout(width="80px"))
    next_btn = widgets.Button(description="Next \u25b6", layout=widgets.Layout(width="80px"))

    image_out = widgets.Image(
        format="png",
        layout=widgets.Layout(max_width="1400px", max_height="700px"),
    )
    info_html = widgets.HTML()

    def _current_ds():
        return datasets[dataset_selector.value]

    def _update(_=None):
        ds = _current_ds()
        split = split_selector.value
        idx = index_slider.value

        # Get the raw sample (bypasses any set_transform)
        sample = _get_raw_sample(ds[split], idx)

        entry = models[model_selector.value]
        m = entry["model"]
        ip = entry["image_processor"]

        device_name = "cpu"
        try:
            device_name = str(next(m.parameters()).device)
        except StopIteration:
            pass

        thr = threshold_slider.value
        image_out.value = render_prediction(sample, m, ip, threshold=thr)
        info_html.value = (
            f"<div style='font-family: monospace; font-size: 13px; line-height: 1.5'>"
            f"<b>Model:</b> {model_selector.value} &nbsp;|&nbsp; "
            f"<b>Device:</b> {device_name} &nbsp;|&nbsp; "
            f"<b>Dataset:</b> {dataset_selector.value} &nbsp;|&nbsp; "
            f"<b>Split:</b> {split} &nbsp;|&nbsp; "
            f"<b>Index:</b> {idx}/{len(ds[split]) - 1}"
            f"</div>"
        )

    def _on_dataset_change(change):
        ds = datasets[change["new"]]
        new_splits = list(ds.keys())
        split_selector.options = new_splits
        split_selector.value = new_splits[0]

    def _on_split_change(_=None):
        ds = _current_ds()
        split = split_selector.value
        index_slider.max = len(ds[split]) - 1
        index_slider.value = 0

    def _on_prev(_):
        if index_slider.value > 0:
            index_slider.value -= 1

    def _on_next(_):
        if index_slider.value < index_slider.max:
            index_slider.value += 1

    dataset_selector.observe(_on_dataset_change, names="value")
    split_selector.observe(_on_split_change, names="value")
    index_slider.observe(_update, names="value")
    model_selector.observe(_update, names="value")
    threshold_slider.observe(_update, names="value")
    prev_btn.on_click(_on_prev)
    next_btn.on_click(_on_next)

    nav = widgets.HBox([prev_btn, index_slider, next_btn])
    controls = widgets.HBox([model_selector, dataset_selector, split_selector])
    threshold_box = widgets.HBox([threshold_slider, threshold_text])
    ui = widgets.VBox([controls, widgets.HBox([nav, threshold_box]), info_html, image_out])

    _update()
    ipy_display(ui)
