"""Model loading, data transforms, and collation for RT-DETR on the moku dataset."""

from __future__ import annotations

import torch
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

from moku.dataset import CATEGORIES, ID_TO_CATEGORY

# Model & dataset identifiers
BASE_MODEL = "PekingU/rtdetr_r18vd"
HF_DATASET = "kaya-go/moku-v2"
HF_MODEL = "kaya-go/moku-v2"
NUM_LABELS = len(CATEGORIES)


def load_image_processor(model_name: str = BASE_MODEL) -> RTDetrImageProcessor:
    """Load the RT-DETR image processor."""
    return RTDetrImageProcessor.from_pretrained(model_name)


def _build_coco_target(example: dict) -> dict:
    """Convert a single HF dataset example to COCO annotation format."""
    return {
        "image_id": example["image_id"],
        "annotations": [
            {
                "id": int(ann_id),
                "category_id": int(cat),
                "bbox": [float(x) for x in bbox],
                "area": float(area),
                "iscrowd": int(iscrowd),
            }
            for ann_id, cat, bbox, area, iscrowd in zip(
                example["objects"]["id"],
                example["objects"]["category"],
                example["objects"]["bbox"],
                example["objects"]["area"],
                example["objects"]["iscrowd"],
            )
        ],
    }


def _unbatch_example(example: dict) -> dict:
    """Unwrap single-element batched example to plain format.

    HF datasets ``set_transform`` may pass examples in batched format (dict of
    lists) even for single-item access. This helper detects that case and
    extracts the first (only) element.
    """
    if isinstance(example.get("image"), list):
        return {k: v[0] if isinstance(v, list) else v for k, v in example.items()}
    return example


def make_eval_transform(image_processor: RTDetrImageProcessor):
    """Create a per-example transform for evaluation (no augmentation)."""

    def transform(example: dict) -> dict:
        example = _unbatch_example(example)
        image = example["image"].convert("RGB")
        target = _build_coco_target(example)
        return image_processor(images=[image], annotations=[target], return_tensors="pt")  # type: ignore[call-arg]

    return transform


def collate_fn(batch: list[dict]) -> dict:
    """Collate: stack pixel_values, keep labels as a list of per-image dicts."""
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": [x["labels"] for x in batch],
    }
