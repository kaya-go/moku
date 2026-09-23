"""Training data: augmentation, oversampling and a torch ``Dataset`` over a moku split.

Images are squashed to 640×640 with plain bilinear interpolation, the same
resize Kaya applies before inference, and scaled to [0, 1] without mean/std
normalization. Targets follow the ``transformers`` DETR convention:
``class_labels`` and normalized ``(cx, cy, w, h)`` boxes.
"""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch

INPUT_SIZE = 640


def train_augmentation(strong: bool = True, size: int = INPUT_SIZE) -> A.Compose:
    """Phone-photo augmentation; ``strong=False`` keeps only flips (the final "no-aug" epochs).

    A goban is symmetric under flips, so flips stay on in both modes.
    """
    ops: list = [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3)]
    if strong:
        ops += [
            # Geometry: phone angles, partial views.
            A.Perspective(scale=(0.03, 0.12), p=0.6),
            A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.RandomResizedCrop(size=(size, size), scale=(0.5, 1.0), ratio=(0.75, 1.33), p=0.5),
            # Lighting.
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.08, p=0.6),
            A.RandomGamma(gamma_limit=(70, 130), p=0.3),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.2),
            # Camera blur and noise.
            A.OneOf([A.GaussianBlur(blur_limit=(3, 7)), A.MotionBlur(blur_limit=(3, 9))], p=0.4),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.3),  # sigma 5-20 on a 0-255 scale
            A.RandomShadow(num_shadows_limit=(1, 3), shadow_dimension=5, shadow_roi=(0, 0, 1, 1), p=0.3),
            # Phone JPEGs.
            A.ImageCompression(quality_range=(40, 95), p=0.3),
            A.Downscale(scale_range=(0.5, 0.9), p=0.2),
        ]
    ops.append(A.Resize(size, size, interpolation=cv2.INTER_LINEAR))
    return A.Compose(
        ops,
        bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"], min_area=1.0, min_visibility=0.3),
    )


def clip_boxes(bboxes, categories, width: int, height: int) -> tuple[list[list[float]], list[int]]:
    """Clip COCO ``[x, y, w, h]`` boxes to the image; drop the empty ones.

    Some source annotations spill slightly outside the image, which albumentations rejects.
    """
    out_boxes, out_cats = [], []
    for (x, y, w, h), cat in zip(bboxes, categories):
        x0, y0 = min(max(float(x), 0.0), width), min(max(float(y), 0.0), height)
        x1, y1 = min(max(float(x) + float(w), 0.0), width), min(max(float(y) + float(h), 0.0), height)
        if x1 - x0 > 0 and y1 - y0 > 0:
            out_boxes.append([x0, y0, x1 - x0, y1 - y0])
            out_cats.append(int(cat))
    return out_boxes, out_cats


def to_target(bboxes, categories, size: int = INPUT_SIZE) -> dict[str, torch.Tensor]:
    """COCO pixel boxes on a ``size²`` image → ``transformers`` DETR labels."""
    boxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
    cxcywh = np.concatenate([boxes[:, :2] + boxes[:, 2:] / 2, boxes[:, 2:]], axis=1) / size
    return {
        "class_labels": torch.as_tensor(np.asarray(categories, dtype=np.int64)),
        "boxes": torch.from_numpy(np.clip(cxcywh, 0.0, 1.0)),
    }


class DetectionDataset(torch.utils.data.Dataset):
    """Augmented ``(pixel_values, labels)`` pairs from rows ``indices`` of a HF split."""

    def __init__(self, split, indices: list[int], transform: A.Compose):
        self.split = split
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def sample(self, i: int) -> tuple[np.ndarray, list[list[float]], list[int]]:
        """Augmented image (HWC uint8), COCO boxes and categories."""
        example = self.split[self.indices[i]]
        image = np.asarray(example["image"].convert("RGB"))
        height, width = image.shape[:2]
        bboxes, cats = clip_boxes(example["objects"]["bbox"], example["objects"]["category"], width, height)
        out = self.transform(image=image, bboxes=bboxes, category_ids=cats)
        return out["image"], [list(b) for b in out["bboxes"]], [int(c) for c in out["category_ids"]]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image, bboxes, cats = self.sample(i)
        pixel_values = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().div_(255.0)
        return pixel_values, to_target(bboxes, cats, image.shape[0])


def collate(batch: list) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    images, labels = zip(*batch)
    return torch.stack(images), list(labels)


def train_indices(split, oversample_real: int = 1, source: str | None = None) -> list[int]:
    """Row indices of the training set: real photos repeated ``oversample_real`` times, generated once.

    ``source`` keeps only ``"real"`` or ``"generated"`` images.
    """
    sources = split["source_dataset"]
    real = [i for i, s in enumerate(sources) if s != "generated"]
    generated = [i for i, s in enumerate(sources) if s == "generated"]
    if source == "real":
        generated = []
    elif source == "generated":
        real = []
    elif source is not None:
        raise ValueError(f"unknown source filter {source!r}")
    return real * max(oversample_real, 1) + generated


def save_previews(dataset: DetectionDataset, out_dir: Path, n: int = 20, seed: int = 0) -> list[Path]:
    """Save ``n`` augmented samples with their boxes drawn, to eyeball the pipeline."""
    import random

    from moku.viz import CATEGORY_COLORS

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    paths = []
    for k in range(n):
        i = rng.randrange(len(dataset))
        image, bboxes, cats = dataset.sample(i)
        canvas = np.ascontiguousarray(image[:, :, ::-1])  # RGB → BGR for cv2
        for (x, y, w, h), cat in zip(bboxes, cats):
            color = tuple(int(CATEGORY_COLORS[cat][j : j + 2], 16) for j in (5, 3, 1))  # hex → BGR
            cv2.rectangle(canvas, (int(x), int(y)), (int(x + w), int(y + h)), color, 1)
        path = out_dir / f"aug_{k:02d}_row{dataset.indices[i]}.jpg"
        cv2.imwrite(str(path), canvas)
        paths.append(path)
    return paths
