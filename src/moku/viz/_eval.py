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


def plot_center_distance_comparison(
    runs: dict[str, dict],
    title: str = "Center Distance",
    figsize: tuple[float, float] = (10, 4),
) -> None:
    """Plot grouped bar charts comparing center-distance F1 across runs.

    Args:
        runs: ``{label: cd_dict}`` where *cd_dict* is the output of
            ``evaluate_center_distance``.
        title: suptitle for the figure.
        figsize: matplotlib figure size.
    """
    labels = list(runs.keys())
    n = len(labels)
    colors = _DEFAULT_COLORS[:n]
    class_names = [ID_TO_CATEGORY[i] for i in range(len(CATEGORIES))]

    # Discover threshold labels from first run
    first = next(iter(runs.values()))
    first_class = next(iter(first["per_class"].values()))
    thresh_labels = list(first_class["thresholds"].keys())

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    total_width = 0.8
    gap = 0.05
    w = (total_width - (n - 1) * gap) / n

    # --- Left: Macro-averaged F1 at each distance threshold ---
    x = np.arange(len(thresh_labels))
    metric_labels = [f"F1@{t}" for t in thresh_labels]
    for i, label in enumerate(labels):
        offset = -total_width / 2 + w / 2 + i * (w + gap)
        vals = []
        for t in thresh_labels:
            class_f1s = [
                runs[label]["per_class"][c]["thresholds"][t]["f1"] for c in class_names if c in runs[label]["per_class"]
            ]
            vals.append(np.mean(class_f1s) if class_f1s else 0.0)
        bars = axes[0].bar(x + offset, vals, w, label=label, color=colors[i])
        axes[0].bar_label(bars, fmt="%.3f", fontsize=max(6, 9 - n), padding=2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metric_labels)
    axes[0].set_ylim(0, 1.15)
    axes[0].set_title(f"{title} — Macro-avg F1")
    axes[0].legend(fontsize=8)

    # --- Right: Per-class F1 at middle threshold ---
    mid_thresh = thresh_labels[len(thresh_labels) // 2]
    x2 = np.arange(len(class_names))
    for i, label in enumerate(labels):
        offset = -total_width / 2 + w / 2 + i * (w + gap)
        vals = [
            runs[label]["per_class"].get(c, {}).get("thresholds", {}).get(mid_thresh, {}).get("f1", 0.0)
            for c in class_names
        ]
        bars = axes[1].bar(x2 + offset, vals, w, label=label, color=colors[i])
        axes[1].bar_label(bars, fmt="%.3f", fontsize=max(6, 9 - n), padding=2)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(class_names)
    axes[1].set_ylim(0, 1.15)
    axes[1].set_title(f"{title} — Per-Class F1@{mid_thresh}")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_threshold_sweep(
    sweep: dict,
    title: str = "Confidence Threshold Sweep",
    figsize: tuple[float, float] | None = None,
) -> None:
    """Plot P/R/F1 and mAP vs confidence threshold from ``sweep_confidence_threshold``.

    Top plot: macro-averaged P/R/F1 with optimal threshold marked.
    Middle plot: per-class F1 curves.
    Bottom plot: mAP@50:95 and mAP@50 vs threshold (if present in sweep).
    """
    thresholds = sweep["score_thresholds"]
    dt = sweep["distance_threshold"]
    macro = sweep["macro"]
    per_class = sweep["per_class"]
    class_names = list(per_class.keys())
    has_map = "map" in sweep and "map_50" in sweep

    nrows = 3 if has_map else 2
    if figsize is None:
        figsize = (10, 4 * nrows)
    fig, axes = plt.subplots(nrows, 1, figsize=figsize)

    # --- Left: Macro P/R/F1 ---
    axes[0].plot(thresholds, macro["precision"], label="Precision", linewidth=2)
    axes[0].plot(thresholds, macro["recall"], label="Recall", linewidth=2)
    axes[0].plot(thresholds, macro["f1"], label="F1", linewidth=2, color="black")

    best_idx = int(np.argmax(macro["f1"]))
    best_t = thresholds[best_idx]
    best_f1 = macro["f1"][best_idx]
    axes[0].axvline(best_t, color="red", linestyle="--", alpha=0.7)
    axes[0].plot(best_t, best_f1, "ro", markersize=8)
    axes[0].annotate(
        f"best={best_t:.2f}\nF1={best_f1:.3f}",
        xy=(best_t, best_f1),
        xytext=(10, -20),
        textcoords="offset points",
        fontsize=9,
        color="red",
    )

    axes[0].set_xlabel("Confidence Threshold")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xlim(0, 0.5)
    axes[0].set_title(f"{title} — Macro (dist≤{dt:.0%})")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # --- Center: Per-class F1 ---
    colors_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for j, name in enumerate(class_names):
        color = colors_cycle[j % len(colors_cycle)]
        f1_vals = per_class[name]["f1"]
        axes[1].plot(thresholds, f1_vals, label=name, linewidth=2, color=color)
        ci = int(np.argmax(f1_vals))
        best_t_cls = thresholds[ci]
        best_f1_cls = f1_vals[ci]
        axes[1].plot(best_t_cls, best_f1_cls, "o", markersize=6, color=color)
        axes[1].axvline(best_t_cls, color=color, linestyle="--", alpha=0.5)
        axes[1].annotate(
            f"{best_t_cls:.2f}",
            xy=(best_t_cls, best_f1_cls),
            xytext=(5, 5 - 15 * j),
            textcoords="offset points",
            fontsize=8,
            color=color,
        )

    axes[1].set_xlabel("Confidence Threshold")
    axes[1].set_ylabel("F1")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlim(0, 0.5)
    axes[1].set_title(f"{title} — Per-Class F1 (dist≤{dt:.0%})")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    # --- Right: mAP vs threshold ---
    if has_map:
        axes[2].plot(thresholds, sweep["map"], label="mAP@50:95", linewidth=2, color="#1b9e77")
        axes[2].plot(thresholds, sweep["map_50"], label="mAP@50", linewidth=2, color="#d95f02")

        best_map_idx = int(np.argmax(sweep["map"]))
        best_map_t = thresholds[best_map_idx]
        best_map_v = sweep["map"][best_map_idx]
        axes[2].axvline(best_map_t, color="red", linestyle="--", alpha=0.7)
        axes[2].plot(best_map_t, best_map_v, "ro", markersize=8)
        axes[2].annotate(
            f"best={best_map_t:.2f}\nmAP={best_map_v:.3f}",
            xy=(best_map_t, best_map_v),
            xytext=(10, -20),
            textcoords="offset points",
            fontsize=9,
            color="red",
        )

        axes[2].set_xlabel("Confidence Threshold")
        axes[2].set_ylabel("mAP")
        axes[2].set_ylim(0, 1.05)
        axes[2].set_xlim(0, 0.5)
        axes[2].set_title(f"{title} — mAP")
        axes[2].legend(fontsize=9)
        axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
