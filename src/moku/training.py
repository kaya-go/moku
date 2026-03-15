"""Training utilities for fine-tuning RT-DETR on the moku dataset."""

from __future__ import annotations

import albumentations as A
import numpy as np
import torch
from PIL import Image
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


def load_model(model_name: str = BASE_MODEL) -> RTDetrForObjectDetection:
    """Load RT-DETR model configured for moku's 3 categories."""
    return RTDetrForObjectDetection.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label=ID_TO_CATEGORY,
        label2id=CATEGORIES,
        ignore_mismatched_sizes=True,
    )


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


def _flip_horizontal(image: Image.Image, annotations: list[dict]) -> tuple[Image.Image, list[dict]]:
    """Flip image and COCO-format bboxes horizontally."""
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    w = image.width
    for ann in annotations:
        x, y, bw, bh = ann["bbox"]
        ann["bbox"] = [w - x - bw, y, bw, bh]
    return image, annotations


# ---------------------------------------------------------------------------
# Albumentations augmentation pipeline
# ---------------------------------------------------------------------------


def build_train_augmentation() -> A.Compose:
    """Build the training augmentation pipeline with albumentations.

    Designed to be aggressive enough to combat overfitting on a small dataset
    while reflecting realistic conditions for mobile goban photos: variable
    angles, lighting, compression artifacts, and partial views.

    Returns an A.Compose that takes ``image`` (np.ndarray HWC uint8) and
    ``bboxes`` (list of [x_min, y_min, w, h] in pixels) plus ``category_ids``.
    Output bboxes remain in COCO format.
    """
    return A.Compose(
        [
            # ── Geometric (phone angles, partial views) ──
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Perspective(scale=(0.03, 0.12), p=0.6),
            A.Rotate(limit=30, border_mode=0, p=0.5),
            A.RandomResizedCrop(
                size=(640, 640),
                scale=(0.5, 1.0),
                ratio=(0.75, 1.33),
                p=0.5,
            ),
            # ── Photometric (variable lighting) ──
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.08, p=0.6),
            A.RandomGamma(gamma_limit=(70, 130), p=0.3),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.2),
            # ── Blur & noise (mobile camera) ──
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=(3, 9), p=1.0),
                ],
                p=0.4,
            ),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
            # ── Shadows & occlusion (real lighting) ──
            A.RandomShadow(
                num_shadows_lower=1,
                num_shadows_upper=3,
                shadow_dimension=5,
                shadow_roi=(0, 0, 1, 1),
                p=0.3,
            ),
            # ── Compression artifacts (JPEG from phones) ──
            A.ImageCompression(quality_range=(40, 95), p=0.3),
            A.Downscale(scale_range=(0.5, 0.9), p=0.2),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["category_ids"],
            min_area=1.0,
            min_visibility=0.3,
        ),
    )


def _apply_augmentation(
    image: Image.Image,
    annotations: list[dict],
    aug: A.Compose,
) -> tuple[Image.Image, list[dict]]:
    """Apply albumentations augmentation to image and COCO annotations.

    Returns the augmented PIL image and updated annotations (bboxes that
    get fully cropped out are removed).
    """
    img_np = np.array(image)
    bboxes = [ann["bbox"] for ann in annotations]
    category_ids = [ann["category_id"] for ann in annotations]

    result = aug(image=img_np, bboxes=bboxes, category_ids=category_ids)

    aug_image = Image.fromarray(result["image"])
    aug_bboxes = result["bboxes"]
    aug_cat_ids = result["category_ids"]

    # Rebuild annotations keeping only surviving boxes
    aug_annotations = []
    for bbox, cat_id in zip(aug_bboxes, aug_cat_ids):
        x, y, w, h = bbox
        aug_annotations.append(
            {
                "id": 0,
                "category_id": int(cat_id),
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": 0,
            }
        )

    return aug_image, aug_annotations


def make_train_transform(image_processor: RTDetrImageProcessor, flip_p: float = 0.5):
    """Create a per-example transform for training with albumentations augmentation.

    The ``flip_p`` parameter is kept for API compatibility but is now ignored;
    flip probability is controlled by the albumentations pipeline.
    """
    aug = build_train_augmentation()

    def transform(example: dict) -> dict:
        example = _unbatch_example(example)
        image = example["image"].convert("RGB")
        target = _build_coco_target(example)

        image, target["annotations"] = _apply_augmentation(image, target["annotations"], aug)

        return image_processor(images=[image], annotations=[target], return_tensors="pt")

    return transform


def make_eval_transform(image_processor: RTDetrImageProcessor):
    """Create a per-example transform for evaluation (no augmentation)."""

    def transform(example: dict) -> dict:
        example = _unbatch_example(example)
        image = example["image"].convert("RGB")
        target = _build_coco_target(example)
        return image_processor(images=[image], annotations=[target], return_tensors="pt")

    return transform


def collate_fn(batch: list[dict]) -> dict:
    """Collate: stack pixel_values, keep labels as a list of per-image dicts."""
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": [x["labels"] for x in batch],
    }



