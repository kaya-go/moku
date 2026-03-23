"""Evaluation visualization helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_model_comparison(
    summary_df: pd.DataFrame,
    metrics: list[str] | None = None,
    title: str = "Model Comparison",
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
    color_map: dict[str, str | tuple] | None = None,
) -> dict[str, str | tuple]:
    """Bar chart comparing models — one subplot per metric, shared legend.

    Args:
        summary_df: DataFrame with a ``model`` column and one column per metric.
        metrics: Column names to plot.  Defaults to all numeric columns.
        title: Figure suptitle.
        ncols: Number of columns in the subplot grid.
        figsize: Figure size. Auto-scaled when ``None``.
        color_map: Optional mapping of model name → color for consistent colors.

    Returns:
        Color map (model name → color) that can be passed to subsequent calls.
    """
    import math

    if metrics is None:
        metrics = [c for c in summary_df.columns if c != "model" and pd.api.types.is_numeric_dtype(summary_df[c])]

    n_metrics = len(metrics)
    n_models = len(summary_df)
    nrows = math.ceil(n_metrics / ncols)

    # Build/extend colour map
    cmap = plt.get_cmap("tab10")
    if color_map is None:
        color_map = {}
    next_idx = len(color_map)
    for _, row in summary_df.iterrows():
        if row["model"] not in color_map:
            color_map[row["model"]] = cmap(next_idx % 10)
            next_idx += 1

    if figsize is None:
        figsize = (5 * ncols, 3.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    x = np.arange(n_models)
    bar_width = 0.7

    for idx, metric in enumerate(metrics):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        vals = summary_df[metric].values

        for i, (_, row) in enumerate(summary_df.iterrows()):
            ax.bar(
                x[i],
                row[metric],
                bar_width,
                color=color_map[row["model"]],
                label=row["model"] if idx == 0 else None,
            )
            ax.text(
                x[i],
                row[metric] + 0.005,
                f"{row[metric]:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_xticks([])
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="y", labelsize=10)

        # Auto y-range with some headroom for labels
        ymin = min(0, float(np.nanmin(vals)))
        ymax = float(np.nanmax(vals))
        margin = (ymax - ymin) * 0.15 if ymax > ymin else 0.1
        ax.set_ylim(ymin, ymax + margin)

    # Hide unused axes
    for idx in range(n_metrics, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.99)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="upper center",
        ncol=min(n_models, 3),
        fontsize=11,
        bbox_to_anchor=(0.5, 0.94),
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.82])
    plt.show()
    return color_map
