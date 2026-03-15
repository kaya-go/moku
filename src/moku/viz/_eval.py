"""mAP bar-chart helpers for model evaluation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from moku.dataset import CATEGORIES, ID_TO_CATEGORY

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
