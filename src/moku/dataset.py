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
import pandas as pd
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
# Size is computed proportionally to board span; these are the tuning constants.
_SYNTHETIC_CORNER_SIZE_FRACTION = 0.028  # ~2.8% of avg board span
_SYNTHETIC_CORNER_SIZE_MIN = 15.0  # minimum 15px regardless of board size
_SYNTHETIC_CORNER_SIZE_FALLBACK = 20.0  # used when board span cannot be computed


def _corner_bbox_size(corners: list[list[float]]) -> float:
    """Compute board_corner bbox size proportional to the board span.

    Uses ~2.8% of the average board span (mean of width and height),
    with a 15 px minimum. Falls back to 20 px if corners are degenerate.

    Corners must be in TL, TR, BR, BL order (from ``_sort_corners_clockwise``).
    """
    if len(corners) < 4:
        return _SYNTHETIC_CORNER_SIZE_FALLBACK
    pts = np.array(corners)
    # Board width: average of top edge (TL→TR) and bottom edge (BL→BR)
    widths = [np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[3])]
    # Board height: average of left edge (TL→BL) and right edge (TR→BR)
    heights = [np.linalg.norm(pts[3] - pts[0]), np.linalg.norm(pts[2] - pts[1])]
    span = (sum(widths) + sum(heights)) / 4
    if span < 1.0:
        return _SYNTHETIC_CORNER_SIZE_FALLBACK
    return max(_SYNTHETIC_CORNER_SIZE_MIN, span * _SYNTHETIC_CORNER_SIZE_FRACTION)


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
            if corners:
                size = _corner_bbox_size(corners)
                half = size / 2
                for cx, cy in corners:
                    ann_ids.append(ann_id)
                    ann_id += 1
                    bboxes.append([cx - half, cy - half, size, size])
                    category_ids.append(CATEGORIES["board_corner"])
                    areas.append(size * size)
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


# ---------------------------------------------------------------------------
# Corner annotation audit & correction utilities
# ---------------------------------------------------------------------------


def _audit_split(ds: Dataset, split_name: str, expected_count: int) -> list[dict]:
    """Run corner audit on a single dataset split. Returns rows for flagged images."""
    corner_cat = CATEGORIES["board_corner"]
    rows = []

    for sample in ds:
        objects = sample["objects"]
        corner_boxes = [(bbox, cat) for bbox, cat in zip(objects["bbox"], objects["category"]) if cat == corner_cat]
        n_corners = len(corner_boxes)
        issues: list[str] = []

        if n_corners != expected_count:
            issues.append(f"wrong_count ({n_corners} vs {expected_count})")

        if n_corners >= 2:
            w, h = sample["width"], sample["height"]
            cx_img, cy_img = w / 2, h / 2

            # Check corners are in distinct quadrants
            quadrants: set[str] = set()
            for (x, y, bw, bh), _ in corner_boxes:
                bcx = x + bw / 2
                bcy = y + bh / 2
                q = ("T" if bcy < cy_img else "B") + ("L" if bcx < cx_img else "R")
                quadrants.add(q)

            if len(quadrants) < n_corners:
                issues.append(f"duplicate_quadrant ({sorted(quadrants)})")

            if n_corners == expected_count and len(quadrants) < expected_count:
                issues.append("not_in_4_quadrants")

            # Check bbox size relative to image area
            img_area = w * h
            for (x, y, bw, bh), _ in corner_boxes:
                rel = (bw * bh) / img_area
                if rel < 1e-5:
                    issues.append(f"too_small ({bw * bh:.1f} px\u00b2)")
                    break
                if rel > 0.01:
                    issues.append(f"too_large ({bw * bh:.1f} px\u00b2)")
                    break

        if issues:
            rows.append(
                {
                    "split": split_name,
                    "image_id": sample.get("image_id", -1),
                    "source_dataset": sample.get("source_dataset", "unknown"),
                    "n_corners": n_corners,
                    "issues": ", ".join(issues),
                }
            )

    return rows


def audit_corners(dataset_or_dict, expected_count: int = 4) -> pd.DataFrame:
    """Audit board_corner annotations for quality issues.

    For each image, checks:
    - Correct number of corners (default: 4)
    - Corners appear in 4 distinct quadrants of the image
    - Corner bbox sizes are reasonable relative to image area

    Args:
        dataset_or_dict: A ``Dataset`` or ``DatasetDict``. If ``DatasetDict``,
            all splits are audited and a ``split`` column is added.
        expected_count: Expected number of board_corner annotations per image.

    Returns:
        DataFrame with one row per flagged image: split, image_id,
        source_dataset, n_corners, issues.
    """
    if hasattr(dataset_or_dict, "items"):
        rows: list[dict] = []
        for split_name, ds in dataset_or_dict.items():
            rows.extend(_audit_split(ds, split_name, expected_count))
        return pd.DataFrame(rows)

    return pd.DataFrame(_audit_split(dataset_or_dict, "dataset", expected_count))


def apply_corner_corrections(dataset: DatasetDict, corrections: dict) -> DatasetDict:
    """Apply human-corrected board_corner annotations to a DatasetDict.

    Corrected boxes (from the annotator) replace the existing
    board_corner annotations for each image. Non-corner annotations are
    kept unchanged.

    Args:
        dataset: The original ``DatasetDict``.
        corrections: Dict mapping filename (``"{split}_{source}_{imageId}.jpg"``)
            to ``{"boxes": [{"id", "x", "y", "w", "h", "category"}, ...]}``.

    Returns:
        A new ``DatasetDict`` with corrected corner annotations.
    """
    corner_cat = CATEGORIES["board_corner"]

    # Build lookup: (split, source, image_id) → correction entry
    corr_lookup: dict[tuple[str, str, str], dict] = {}
    for fname, val in corrections.items():
        # Parse filename: "{split}_{source}_{imageId}.jpg"
        stem = fname.rsplit(".", 1)[0]  # drop .jpg
        parts = stem.split("_", 1)  # split on first _ to get split name
        if len(parts) < 2:
            continue
        # Split name may itself contain underscores (e.g. no, but be safe)
        # Format is: {split}_{source}_{imageId}
        # Split is always train/validation/test
        for known_split in ("validation", "train", "test"):
            if stem.startswith(known_split + "_"):
                rest = stem[len(known_split) + 1 :]
                # rest = "{source}_{imageId}" — imageId is the last segment
                last_underscore = rest.rfind("_")
                if last_underscore >= 0:
                    source = rest[:last_underscore]
                    image_id = rest[last_underscore + 1 :]
                    corr_lookup[(known_split, source, image_id)] = val
                break

    def _make_apply(split_name: str):
        def _apply(sample: dict, idx: int) -> dict:
            source = sample.get("source_dataset", "unknown")
            image_id = str(sample["image_id"])
            corr = corr_lookup.get((split_name, source, image_id))
            if corr is None:
                return sample

            corr_boxes = [b for b in corr.get("boxes", []) if b.get("category") == corner_cat]

            objects = sample["objects"]
            non_corner_idx = [i for i, c in enumerate(objects["category"]) if c != corner_cat]

            new_ids = [objects["id"][i] for i in non_corner_idx]
            new_bboxes = [objects["bbox"][i] for i in non_corner_idx]
            new_cats = [objects["category"][i] for i in non_corner_idx]
            new_areas = [objects["area"][i] for i in non_corner_idx]
            new_iscrowd = [objects["iscrowd"][i] for i in non_corner_idx]

            for box in corr_boxes:
                new_ids.append(int(box["id"]))
                new_bboxes.append([box["x"], box["y"], box["w"], box["h"]])
                new_cats.append(corner_cat)
                new_areas.append(box["w"] * box["h"])
                new_iscrowd.append(0)

            return {
                **sample,
                "objects": {
                    "id": new_ids,
                    "bbox": new_bboxes,
                    "category": new_cats,
                    "area": new_areas,
                    "iscrowd": new_iscrowd,
                },
            }

        return _apply

    return DatasetDict({split: ds.map(_make_apply(split), with_indices=True) for split, ds in dataset.items()})


def load_annotated_generated(
    images_dir: Path,
    corrections_path: Path,
    images_json_path: Path,
) -> Dataset:
    """Load human-corrected annotations for generated (e.g. Gemini) images.

    Args:
        images_dir: Directory containing the image files.
        corrections_path: Path to ``corrected.json`` from the annotator.
        images_json_path: Path to ``images.json`` with image metadata.

    Returns:
        An HF ``Dataset`` with the same schema as real/synthetic datasets.
        Only images present in ``corrections_path`` are included.
    """
    with open(corrections_path) as f:
        corrections = json.load(f)
    with open(images_json_path) as f:
        images_meta = json.load(f)

    # Build width/height lookup from images.json
    meta_lookup = {img["filename"]: img for img in images_meta["images"]}

    rows = []
    ann_id = 0
    for i, (filename, corr) in enumerate(sorted(corrections.items())):
        if corr.get("excluded"):
            continue

        meta = meta_lookup.get(filename, {})
        width = meta.get("width", 1024)
        height = meta.get("height", 1024)
        image_path = str(images_dir / filename)

        boxes = corr.get("boxes", [])
        objects = {
            "id": [],
            "bbox": [],
            "category": [],
            "area": [],
            "iscrowd": [],
        }
        for box in boxes:
            objects["id"].append(ann_id)
            objects["bbox"].append([box["x"], box["y"], box["w"], box["h"]])
            objects["category"].append(box["category"])
            objects["area"].append(box["w"] * box["h"])
            objects["iscrowd"].append(0)
            ann_id += 1

        rows.append(
            {
                "image": image_path,
                "image_id": i,
                "width": width,
                "height": height,
                "source_dataset": "generated",
                "objects": objects,
            }
        )

    ds = Dataset.from_list(rows)
    ds = ds.cast_column("image", HFImage())
    return ds
