"""Annotation quality.

- audit: corner sanity checks, and board-level consistency of the ground truth
  (does the position read from the annotations snap cleanly onto a grid?);
- corrections: move misplaced corner labels onto the grid defined by the stones (``refine_corners``);
- generated images: load Gemini-generated images with their inherited annotations.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from datasets import Image as HFImage

from moku.board import _UNIT_SQUARE, apply_homography, compute_homography, fit_homography, truth_board
from moku.dataset import CATEGORIES


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


def load_annotated_generated(
    images_dir: Path,
    images_json_path: Path,
    corrections_path: Path | None = None,
) -> Dataset:
    """Load annotations for generated (e.g. Gemini) images.

    When ``corrections_path`` is provided and exists, only images present in
    that file are included (with human-corrected boxes).  Otherwise, all
    per-image JSON files in ``images_dir`` are loaded directly.

    Args:
        images_dir: Directory containing the image files and per-image JSONs.
        images_json_path: Path to ``images.json`` with image metadata.
        corrections_path: Optional path to a ``corrected.json`` of manually
            corrected boxes.  If ``None`` or the file does not exist, original
            annotations from the per-image JSONs are used instead.

    Returns:
        An HF ``Dataset`` with the same schema as real/synthetic datasets.
    """
    with open(images_json_path) as f:
        images_meta = json.load(f)

    # Build width/height lookup from images.json
    meta_lookup = {img["filename"]: img for img in images_meta["images"]}

    use_corrections = corrections_path is not None and Path(corrections_path).exists()

    if use_corrections:
        with open(corrections_path) as f:
            corrections = json.load(f)
        entries = sorted(corrections.items())
    else:
        # Load per-image JSON files from images_dir
        entries = []
        for json_path in sorted(images_dir.glob("*.json")):
            with open(json_path) as f:
                data = json.load(f)
            filename = data["image"]["filename"]
            entries.append((filename, data))

    rows = []
    ann_id = 0
    for i, (filename, corr) in enumerate(entries):
        if isinstance(corr, dict) and corr.get("excluded"):
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


# ---------------------------------------------------------------------------
# Board-level consistency of the ground truth
# ---------------------------------------------------------------------------


def refine_corners(
    bboxes, categories, min_stones: int = 8, max_shift_cells: float = 1.0, iterations: int = 3
) -> tuple[list[list[float]], float] | None:
    """Move the 4 annotated corners onto the grid defined by the annotated stones.

    Stones are snapped to intersections with the annotated corners, a least-squares homography
    is fitted from the stones to their intersections (repeated a few times), and the corners are
    re-projected from it. Returns ``(bboxes with moved corner boxes, largest shift in cells)``, or
    ``None`` when the stones cannot be trusted to define the grid (too few, too clustered, off the
    grid, colliding) or when the fit does not improve the snapping residual.
    """
    truth = truth_board(bboxes, categories)
    cats = np.asarray(categories)
    boxes = np.asarray(bboxes, dtype=np.float64).reshape(-1, 4)
    if truth is None or (cats != CATEGORIES["board_corner"]).sum() < min_stones:
        return None
    n = truth.board_size
    stones = boxes[cats != CATEGORIES["board_corner"], :2] + boxes[cats != CATEGORIES["board_corner"], 2:] / 2
    h = compute_homography(truth.corners, _UNIT_SQUARE)
    residual_before = truth.snap_residual
    for _ in range(iterations):
        grid = apply_homography(h, stones) * (n - 1)
        lattice = np.round(grid)
        if (lattice < 0).any() or (lattice > n - 1).any() or len(np.unique(lattice, axis=0)) < len(lattice):
            return None
        span = lattice.max(axis=0) - lattice.min(axis=0)
        if (span < (n - 1) / 2).any():  # too little of the board to extrapolate the corners
            return None
        h = fit_homography(stones, lattice / (n - 1))
        if h is None:
            return None
    grid = apply_homography(h, stones) * (n - 1)
    residual_after = float(np.median(np.hypot(*(grid - np.round(grid)).T)))
    to_image = np.linalg.inv(h)
    corners = apply_homography(to_image, _UNIT_SQUARE)
    cell = np.mean([np.hypot(*(truth.corners[i] - truth.corners[(i + 1) % 4])) for i in range(4)]) / (n - 1)
    shift = float(np.hypot(*(corners - truth.corners).T).max() / cell)
    if residual_after >= residual_before or shift > max_shift_cells:
        return None
    out = boxes.copy()
    for i in np.where(cats == CATEGORIES["board_corner"])[0]:
        center = out[i, :2] + out[i, 2:] / 2
        nearest = corners[np.hypot(*(corners - center).T).argmin()]
        out[i, :2] = nearest - out[i, 2:] / 2
    return out.tolist(), shift


def audit_boards(dataset: DatasetDict, max_residual: float = 0.3) -> pd.DataFrame:
    """Check that each image's annotations describe a consistent position.

    The 4 annotated corners define a grid; every annotated stone should land
    near one intersection, alone. Stones colliding on one intersection, falling
    off the grid, or a large snapping residual point to a misplaced corner or a
    wrong board size — the same failures Kaya would hit on a perfect detector.

    Returns one row per image with issues (``residual`` is in grid cells).
    """
    rows = []
    for split, ds in dataset.items():
        for i, ex in enumerate(ds.remove_columns("image")):
            truth = truth_board(ex["objects"]["bbox"], ex["objects"]["category"])
            n_corners = sum(c == CATEGORIES["board_corner"] for c in ex["objects"]["category"])
            if truth is None:
                issues = [f"{n_corners} corners"]
                row = {"board_size": None, "residual": None, "collisions": None, "out_of_bounds": None}
            else:
                issues = []
                if truth.collisions:
                    issues.append(f"{truth.collisions} collisions")
                if truth.out_of_bounds:
                    issues.append(f"{truth.out_of_bounds} off-grid")
                if truth.snap_residual > max_residual:
                    issues.append(f"residual {truth.snap_residual:.2f}")
                row = {
                    "board_size": truth.board_size,
                    "residual": round(truth.snap_residual, 3),
                    "collisions": truth.collisions,
                    "out_of_bounds": truth.out_of_bounds,
                }
            if issues:
                rows.append(
                    {
                        "split": split,
                        "index": i,
                        "image_id": ex["image_id"],
                        "source": ex["source_dataset"],
                        **row,
                        "issues": ", ".join(issues),
                    }
                )
    return pd.DataFrame(rows)
