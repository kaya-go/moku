"""Dataset loading and harmonization utilities for moku.

Loads raw COCO-format datasets exported from Roboflow and harmonizes them
into a single HuggingFace dataset with unified categories.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict
from datasets import Image as HFImage
from scipy.spatial import ConvexHull
from sklearn.model_selection import GroupShuffleSplit

# Harmonized category definitions
CATEGORIES = {
    "black_stone": 0,
    "white_stone": 1,
    "board_corner": 2,
}

ID_TO_CATEGORY = {v: k for k, v in CATEGORIES.items()}

# Map source category names to harmonized names.
# Categories not listed here are dropped.
CATEGORY_MAP = {
    # Go Game detection v10
    "black_stone": "black_stone",
    "white_stone": "white_stone",
    "board_corner": "board_corner",
    # go-chess 2 v3 (goboard is dropped; corners are synthesized from segmentation)
}


def load_coco_split(annotation_file: Path) -> dict:
    """Load a single COCO annotation file and return the parsed JSON."""
    with open(annotation_file) as f:
        return json.load(f)


def load_coco_dataset(dataset_path: Path) -> dict[str, dict]:
    """Load COCO annotations for all splits (train/valid/test) in a dataset directory.

    Returns a dict mapping split name to COCO data dict.
    """
    splits = {}
    for split in ["train", "valid", "test"]:
        ann_file = dataset_path / split / "_annotations.coco.json"
        if ann_file.exists():
            splits[split] = load_coco_split(ann_file)
    return splits


def get_base_image_name(filename: str) -> str:
    """Extract base image name from a Roboflow-augmented filename.

    Roboflow names augmented images as '<base>_jpg.rf.<hex_hash>.jpg'.
    This strips the augmentation suffix to get the original base name.
    """
    return re.sub(r"_jpg\.rf\.[a-f0-9]+\.jpg$", "", filename)


def get_source_categories(coco_data: dict) -> dict[int, str]:
    """Extract category id-to-name mapping from COCO data."""
    return {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}


def _sort_corners_clockwise(corners: np.ndarray) -> np.ndarray:
    """Sort 4 corner points in order: top-left, top-right, bottom-right, bottom-left.

    Uses angle from centroid to each point. In image coordinates (y-axis down),
    sorting by atan2 angle gives clockwise order starting from top-left.
    Robust to arbitrary board rotations and perspective distortions.
    """
    centroid = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0])
    order = np.argsort(angles)
    sorted_corners = corners[order]

    # After angle sort the order is clockwise from upper-left in image coords.
    # Identify TL as the point with smallest x+y, then rotate the sequence.
    sums = sorted_corners[:, 0] + sorted_corners[:, 1]
    tl_idx = int(np.argmin(sums))
    return np.roll(sorted_corners, -tl_idx, axis=0)


def _corners_from_segmentation(segmentation: list) -> list[list[float]]:
    """Extract 4 board corners from a COCO segmentation polygon.

    Computes the convex hull and identifies the 4 extreme corner points
    using sum/diff projections on hull vertices.

    Note: for go_chess, the polygon traces the physical board edge which is
    slightly outside the grid intersections. Corner adjustment (if needed)
    is left to the inference/UI layer.

    Returns empty list if 4 unique corners cannot be found.
    """
    coords = segmentation[0]
    points = np.array(coords, dtype=np.float64).reshape(-1, 2)
    hull = ConvexHull(points)
    hull_pts = points[hull.vertices]

    sums = hull_pts[:, 0] + hull_pts[:, 1]
    diffs = hull_pts[:, 0] - hull_pts[:, 1]

    indices = [np.argmin(sums), np.argmax(diffs), np.argmax(sums), np.argmin(diffs)]
    if len(set(indices)) < 4:
        return []

    corners = hull_pts[indices]
    corners = _sort_corners_clockwise(corners)

    return corners.tolist()


# Default size for synthesized board_corner bounding boxes (go_chess).
# Matches the average size of real board_corner bboxes in go_game_v10.
_SYNTHETIC_CORNER_SIZE = 20.0


def harmonize_coco_split(
    coco_data: dict,
    split_dir: Path,
    source_name: str,
    annotation_id_start: int = 0,
) -> list[dict]:
    """Convert a single COCO split into harmonized rows for HuggingFace.

    Each row follows the standard HF object detection format (same as cppe-5):
    - image_path: absolute path to the image file
    - image_id: original image id
    - width, height: image dimensions
    - source_dataset: name of the source dataset
    - objects: dict with id, bbox, category, area, iscrowd lists

    Bboxes remain in COCO format: [x_min, y_min, width, height].

    Categories: black_stone, white_stone, board_corner.
    - go_game_v10: board_corner annotations pass through directly.
    - go_chess: board_corner bboxes are synthesized from board segmentation polygons.
    - board/goboard bboxes are dropped (corners replace them).
    """
    source_cats = get_source_categories(coco_data)
    board_cat_names = {"board", "goboard"}

    # Group annotations by image_id
    img_to_anns: dict[int, list[dict]] = {}
    for ann in coco_data.get("annotations", []):
        img_to_anns.setdefault(ann["image_id"], []).append(ann)

    ann_id = annotation_id_start
    rows = []
    for img_info in coco_data["images"]:
        img_id = img_info["id"]
        img_anns = img_to_anns.get(img_id, [])

        ann_ids = []
        bboxes = []
        category_ids = []
        areas = []
        iscrowd = []
        board_seg = None

        for ann in img_anns:
            source_cat_name = source_cats.get(ann["category_id"])

            # Capture board segmentation for corner synthesis (go_chess)
            if source_cat_name in board_cat_names and ann.get("segmentation"):
                board_seg = ann["segmentation"]

            harmonized_name = CATEGORY_MAP.get(source_cat_name)
            if harmonized_name is None:
                continue  # Drop unmapped categories (board, goboard, empty, etc.)

            ann_ids.append(ann_id)
            ann_id += 1
            bboxes.append(ann["bbox"])
            category_ids.append(CATEGORIES[harmonized_name])
            areas.append(ann.get("area", ann["bbox"][2] * ann["bbox"][3]))
            iscrowd.append(ann.get("iscrowd", 0))

        # Synthesize board_corner annotations from segmentation (go_chess)
        if board_seg is not None:
            corners = _corners_from_segmentation(board_seg)
            half = _SYNTHETIC_CORNER_SIZE / 2
            for cx, cy in corners:
                ann_ids.append(ann_id)
                ann_id += 1
                bboxes.append([cx - half, cy - half, _SYNTHETIC_CORNER_SIZE, _SYNTHETIC_CORNER_SIZE])
                category_ids.append(CATEGORIES["board_corner"])
                areas.append(_SYNTHETIC_CORNER_SIZE * _SYNTHETIC_CORNER_SIZE)
                iscrowd.append(0)

        # Skip images with no kept annotations
        if not bboxes:
            continue

        rows.append(
            {
                "image_path": str(split_dir / img_info["file_name"]),
                "image_id": img_id,
                "width": img_info["width"],
                "height": img_info["height"],
                "source_dataset": source_name,
                "objects": {
                    "id": ann_ids,
                    "bbox": bboxes,
                    "category": category_ids,
                    "area": areas,
                    "iscrowd": iscrowd,
                },
            }
        )

    return rows


def build_dataset(
    raw_data_dir: Path,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """Build a harmonized HuggingFace DatasetDict from raw COCO datasets.

    All images from all sources and Roboflow splits are pooled, then re-split
    into train/validation/test at the base-image level (grouped) to prevent
    data leakage from Roboflow augmentations.

    Args:
        raw_data_dir: Path to directory containing raw COCO dataset folders.
        val_fraction: Fraction of data for validation split.
        test_fraction: Fraction of data for test split.
        seed: Random seed for reproducibility.

    Returns:
        A DatasetDict with train/validation/test splits, images cast to HFImage.
    """
    # Define which datasets to include and their directory names
    dataset_dirs = {
        "go_game_v10": "Go Game detection.v10i.coco",
        # v1 is skipped: fully contained in v10
        "go_chess": "go-chess 2.v3-go-chess.v1.coco",
    }

    # Pool all images from all sources and all Roboflow splits
    all_rows: list[dict] = []
    ann_id_counter = 0

    for source_name, dir_name in dataset_dirs.items():
        dataset_path = raw_data_dir / dir_name
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        coco_splits = load_coco_dataset(dataset_path)

        source_total = 0
        for coco_split, coco_data in coco_splits.items():
            split_dir = dataset_path / coco_split
            rows = harmonize_coco_split(coco_data, split_dir, source_name, annotation_id_start=ann_id_counter)
            # Advance counter past all annotation IDs used
            for r in rows:
                ann_id_counter += len(r["objects"]["id"])
            all_rows.extend(rows)
            source_total += len(rows)
        print(f"  {source_name}: {source_total} images")

    print(f"  Total pooled: {len(all_rows)} images")

    # Extract base image names and source labels for grouped stratified split
    base_names = np.array([get_base_image_name(Path(r["image_path"]).name) for r in all_rows])
    sources = np.array([r["source_dataset"] for r in all_rows])

    # First split: separate test set
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    rest_idx, test_idx = next(gss_test.split(all_rows, groups=base_names))

    # Second split: separate validation from remaining train
    val_fraction_adjusted = val_fraction / (1 - test_fraction)
    rest_base_names = base_names[rest_idx]
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_fraction_adjusted, random_state=seed)
    train_sub_idx, val_sub_idx = next(gss_val.split(rest_idx, groups=rest_base_names))
    train_idx = rest_idx[train_sub_idx]
    val_idx = rest_idx[val_sub_idx]

    # Report split sizes
    split_map = {"train": train_idx, "validation": val_idx, "test": test_idx}
    for split_name, idx in split_map.items():
        n_bases = len(set(base_names[idx]))
        src_counts = Counter(sources[idx])
        src_str = ", ".join(f"{k}: {v}" for k, v in sorted(src_counts.items()))
        n_with_corners = sum(
            1 for i in idx if sum(1 for c in all_rows[i]["objects"]["category"] if c == CATEGORIES["board_corner"]) == 4
        )
        print(f"  {split_name}: {len(idx)} images ({n_bases} base) [{src_str}] — {n_with_corners} with 4 corners")

    # Build DatasetDict
    dataset_dict = {}
    for split_name, idx in split_map.items():
        rows = [all_rows[i] for i in idx]
        ds = Dataset.from_list(rows)
        ds = ds.cast_column("image_path", HFImage())
        ds = ds.rename_column("image_path", "image")
        dataset_dict[split_name] = ds

    return DatasetDict(dataset_dict)


def compute_split_stats(ds: Dataset) -> dict:
    """Compute annotation statistics for a dataset split."""
    all_categories: list[int] = []
    sources: list[str] = []
    total_objects = 0

    for sample in ds:
        all_categories.extend(sample["objects"]["category"])
        sources.append(sample["source_dataset"])
        total_objects += len(sample["objects"]["bbox"])

    return {
        "num_images": len(ds),
        "total_objects": total_objects,
        "avg_objects_per_image": total_objects / len(ds) if len(ds) > 0 else 0,
        "category_counts": Counter(all_categories),
        "source_counts": Counter(sources),
    }
