"""Publish a trained detector to the Hugging Face Hub (weights, processor, ONNX, model card)."""

from __future__ import annotations

from pathlib import Path

from moku.inference import resolve_source

CARD_TEMPLATE = """---
library_name: transformers
pipeline_tag: object-detection
license: agpl-3.0
base_model: {base_model}
datasets:
- {dataset}
tags:
- go
- baduk
- board-recognition
- onnx
---

# {name}

Go board detector used by [Kaya](https://github.com/kaya-go/kaya) to turn a photo
of a goban into a position (SGF). Trained with [moku](https://github.com/kaya-go/moku).

It detects three classes: `black_stone` (0), `white_stone` (1) and `board_corner` (2).
Kaya computes a homography from the 4 corners and snaps every stone onto the grid.

## Files

- `model.safetensors`, `config.json`, `preprocessor_config.json`: 🤗 transformers checkpoint.
- `model.onnx`: what Kaya runs (ONNX Runtime Web). Input `pixel_values` `(batch, 3, 640, 640)`,
  RGB scaled to [0, 1] with no mean/std normalization; outputs `logits` `(batch, 300, 3)`
  (apply a sigmoid) and `pred_boxes` `(batch, 300, 4)`, normalized `(cx, cy, w, h)`.{extra_outputs}

## Evaluation

{metrics}

Board metrics run Kaya's own post-processing (stone threshold {threshold}{pipeline}) and compare the
resulting position with the one read from the annotations: *perfect* is the share of boards
without a single wrong intersection and no corner more than half a cell off. Intervals are 90%
bootstrap CIs over photos.
"""


def model_card(
    name: str,
    metrics_markdown: str,
    base_model: str,
    dataset: str = "kaya-go/moku-v3",
    threshold: float = 0.035,
    corner_head: bool = False,
    corners: str = "kaya",
    logit_offset: float = 0.0,
) -> str:
    extra = ""
    if corner_head:
        extra += (
            "\n  With the corner head, a third output `corner_points` `(batch, 8, 3)` holds the 8 best"
            "\n  board-corner peaks `(x, y, score)`, x and y normalized to [0, 1] (class-agnostic)."
        )
    if logit_offset:
        extra += (
            f"\n  The stone-threshold calibration is baked into `logits` (offset {logit_offset:+.2f} on every"
            "\n  class logit), so Kaya's fixed 0.035 threshold needs no per-model setting."
        )
    pipeline = "" if corners == "kaya" else f", corners from `{corners}`"
    return CARD_TEMPLATE.format(
        name=name,
        metrics=metrics_markdown,
        base_model=base_model,
        dataset=dataset,
        threshold=threshold,
        extra_outputs=extra,
        pipeline=pipeline,
    )


def load_for_publish(path: str):
    """The detector as it should land on the Hub, corner head included.

    ``from_pretrained`` drops ``corner_head.*`` (not part of the transformers architecture), so
    the head is re-attached from the checkpoint. Without it, the Hub weights cannot reproduce
    the ONNX's ``corner_points`` (``moku export kaya-go/moku-v4`` would export no corner head).
    """
    from transformers import AutoModelForObjectDetection

    from moku.corner_head import load_corner_head

    model = AutoModelForObjectDetection.from_pretrained(path)
    load_corner_head(model, path)
    return model


def publish_model(
    source: str,
    repo_id: str,
    private: bool = True,
    onnx_path: Path | None = None,
    card: str | None = None,
) -> str:
    """Push ``source`` (bucket checkpoint, local dir or Hub repo) to ``repo_id``."""
    from huggingface_hub import HfApi
    from transformers import AutoImageProcessor

    path = resolve_source(source)
    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    load_for_publish(path).push_to_hub(repo_id)
    AutoImageProcessor.from_pretrained(path).push_to_hub(repo_id)
    if onnx_path is not None:
        api.upload_file(path_or_fileobj=str(onnx_path), path_in_repo="model.onnx", repo_id=repo_id)
    if card is not None:
        api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo_id)
    return f"https://huggingface.co/{repo_id}"
