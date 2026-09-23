"""External datasets converted to the moku format (image + COCO ``objects``).

- **Roboflow ``my-go-detection``** (CC BY 4.0): 367 phone photos (9×9, 13×13, 19×19, up to
  4096 px) labelled with the moku classes. Split by photo into train / validation / test.
- **Gomrade** (Kaggle, CC BY-NC-ND 4.0): video frames of ~60 real 19×19 games with the position
  of every frame (text grid) and the 4 grid corners clicked once per game. Evaluation only:
  the license forbids derivatives, so it never enters training nor a public dataset.

Gomrade has no boxes; they are synthesized from the position and the corners (stones at their
intersection, through the corners' homography), so ``moku eval`` scores it unchanged: its board
metrics recover exactly the annotated position.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from datasets import Dataset, Features, List, Value
from datasets import Image as HFImage

from moku.board import apply_homography, compute_homography
from moku.dataset import CATEGORIES

FEATURES = Features(
    {
        "image": HFImage(),
        "image_id": Value("int64"),
        "width": Value("int64"),
        "height": Value("int64"),
        "source_dataset": Value("string"),
        "objects": {
            "area": List(Value("float64")),
            "bbox": List(List(Value("float64"))),
            "category": List(Value("int64")),
            "id": List(Value("int64")),
            "iscrowd": List(Value("int64")),
        },
    }
)

_UNIT_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def _objects(bboxes: list[list[float]], categories: list[int], first_id: int = 0) -> dict:
    return {
        "area": [float(w * h) for _, _, w, h in bboxes],
        "bbox": [[float(v) for v in b] for b in bboxes],
        "category": [int(c) for c in categories],
        "id": list(range(first_id, first_id + len(bboxes))),
        "iscrowd": [0] * len(bboxes),
    }


# ---------------------------------------------------------------------------
# Roboflow COCO export
# ---------------------------------------------------------------------------


def photo_key(file_name: str) -> str:
    """Original photo name of a Roboflow export file (``<name>_jpg.rf.<hash>.jpg`` → ``<name>``)."""
    stem = re.sub(r"\.rf\.[0-9a-f]+\.\w+$", "", file_name)
    return re.sub(r"[_.](jpe?g|png|webp)$", "", stem, flags=re.IGNORECASE).lower()


def load_roboflow_coco(root: Path, source: str) -> tuple[Dataset, list[str]]:
    """All images of a Roboflow COCO export (every split) and their photo keys.

    Categories are mapped by name onto the moku classes; others (e.g. ``go-board``) are dropped.
    """
    rows, keys = [], []
    for ann_file in sorted(root.glob("*/_annotations.coco.json")):
        coco = json.loads(ann_file.read_text())
        to_moku = {c["id"]: CATEGORIES[c["name"]] for c in coco["categories"] if c["name"] in CATEGORIES}
        by_image: dict[int, list[dict]] = {}
        for ann in coco["annotations"]:
            if ann["category_id"] in to_moku:
                by_image.setdefault(ann["image_id"], []).append(ann)
        for im in coco["images"]:
            anns = by_image.get(im["id"], [])
            rows.append(
                {
                    "image": str(ann_file.parent / im["file_name"]),
                    "image_id": len(rows),
                    "width": im["width"],
                    "height": im["height"],
                    "source_dataset": source,
                    "objects": _objects([a["bbox"] for a in anns], [to_moku[a["category_id"]] for a in anns]),
                }
            )
            keys.append(photo_key(im.get("extra", {}).get("name") or im["file_name"]))
    return Dataset.from_list(rows, features=FEATURES), keys


def split_by_group(groups: list[str], fractions: dict[str, float], seed: int = 0) -> dict[str, list[int]]:
    """Assign whole groups (photos) to splits, so no photo straddles two splits."""
    unique = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    bounds = np.cumsum([fractions[s] for s in fractions]) * len(unique)
    split_of = {}
    for rank, group in enumerate(unique):
        split_of[group] = list(fractions)[int(np.searchsorted(bounds, rank, side="right"))]
    return {s: [i for i, g in enumerate(groups) if split_of[g] == s] for s in fractions}


# ---------------------------------------------------------------------------
# Gomrade
# ---------------------------------------------------------------------------

_CELL = {".": 0, "B": 1, "W": 2}


def read_gomrade_position(path: Path) -> np.ndarray:
    """19×19 text grid (``.``/``B``/``W``) → int8 grid (0 empty, 1 black, 2 white)."""
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    return np.array([[_CELL[c] for c in row] for row in rows], dtype=np.int8)


def gomrade_corners(game_dir: Path) -> np.ndarray | None:
    """The 4 grid corners clicked for a game (``board_extractor_state.yml``), in frame pixels.

    Click order is the order of the position file: row 0 runs from the first click to the second.
    """
    import yaml

    state = game_dir / "board_extractor_state.yml"
    if not state.exists():
        return None
    clicks = yaml.safe_load(state.read_text()).get("pts_clicks")
    return np.asarray(clicks, dtype=np.float64) if clicks and len(clicks) == 4 else None


def synthesize_objects(grid: np.ndarray, corners: np.ndarray) -> dict:
    """COCO boxes for a position seen through ``corners`` (grid TL, TR, BR, BL in pixels)."""
    n = grid.shape[0]
    to_image = compute_homography(_UNIT_SQUARE, np.asarray(corners, dtype=np.float64))
    cell = np.mean([np.hypot(*(corners[i] - corners[(i + 1) % 4])) for i in range(4)]) / (n - 1)
    rows, cols = np.nonzero(grid)
    centers = apply_homography(to_image, np.stack([cols, rows], axis=1) / (n - 1)) if len(rows) else np.zeros((0, 2))
    bboxes, cats = [], []
    stone = 0.9 * cell
    for (x, y), value in zip(centers, grid[rows, cols]):
        bboxes.append([x - stone / 2, y - stone / 2, stone, stone])
        cats.append(CATEGORIES["black_stone"] if value == 1 else CATEGORIES["white_stone"])
    corner = 0.4 * cell
    for x, y in corners:
        bboxes.append([x - corner / 2, y - corner / 2, corner, corner])
        cats.append(CATEGORIES["board_corner"])
    return _objects(bboxes, cats)


def gomrade_games(root: Path) -> dict[str, list[Path]]:
    """Game folder → its frames (images with a position file), sorted by name."""
    games = {}
    for state in sorted(root.rglob("board_extractor_state.yml")):
        frames = sorted(p for p in state.parent.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        frames = [p for p in frames if p.with_suffix(".txt").exists()]
        if frames:
            games[str(state.parent.relative_to(root))] = frames
    return games


def sample_frames(frames: list[Path], k: int) -> list[Path]:
    """``k`` frames spread over the game by stone count (empty boards skipped)."""
    counts = np.array([(read_gomrade_position(f.with_suffix(".txt")) > 0).sum() for f in frames])
    nonempty = np.where(counts > 0)[0]
    if len(nonempty) == 0:
        return []
    targets = [counts.max() * (j + 1) / k for j in range(k)]
    picked = sorted({int(nonempty[np.abs(counts[nonempty] - t).argmin()]) for t in targets})
    return [frames[i] for i in picked]


# Folders of rendered diagrams (screen captures of lessons), not photos of real boards.
GOMRADE_DIGITAL = {"dataset/2", "dataset/3", "dataset/4", "dataset/9", "dataset/11", "dataset/12"}


def load_gomrade(root: Path, frames_per_game: int = 3) -> Dataset:
    """Sampled Gomrade frames of real boards in the moku format; ``source_dataset`` is ``gomrade/<game>``."""
    from PIL import Image

    rows = []
    for game, frames in gomrade_games(root).items():
        if game in GOMRADE_DIGITAL:
            continue
        corners = gomrade_corners(root / game)
        if corners is None:
            continue
        for frame in sample_frames(frames, frames_per_game):
            grid = read_gomrade_position(frame.with_suffix(".txt"))
            with Image.open(frame) as im:
                width, height = im.size
            rows.append(
                {
                    "image": str(frame),
                    "image_id": len(rows),
                    "width": width,
                    "height": height,
                    "source_dataset": f"gomrade/{game}",
                    "objects": synthesize_objects(grid, corners),
                }
            )
    return Dataset.from_list(rows, features=FEATURES)


# ---------------------------------------------------------------------------
# Dataset builds
# ---------------------------------------------------------------------------

ROBOFLOW_SOURCE = "my_go_detection"
TRAIN_MAX_SIDE = 1280


def photo_groups(ds: Dataset, keys: list[str]) -> list[str]:
    """Group images that show the same photo (same file name) or the same position (up to symmetry)."""
    from moku.board import truth_board
    from moku.evaluation import _photo_cluster

    parent = {k: k for k in keys}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    first_key_of_position: dict[str, str] = {}
    for i, (objects, key) in enumerate(zip(ds.remove_columns("image")["objects"], keys)):
        truth = truth_board(objects["bbox"], objects["category"])
        if truth is None or not truth.grid.any():
            continue
        position = _photo_cluster(truth.grid, i)
        other = first_key_of_position.setdefault(position, key)
        parent[find(key)] = find(other)
    return [find(k) for k in keys]


def _downscale(example: dict, max_side: int = TRAIN_MAX_SIDE) -> dict:
    """Shrink large photos (and their boxes) so training decodes and augments them quickly."""
    image = example["image"]
    scale = max_side / max(image.size)
    if scale >= 1:
        return example
    size = (round(image.width * scale), round(image.height * scale))
    objects = example["objects"]
    return {
        **example,
        "image": image.convert("RGB").resize(size),
        "width": size[0],
        "height": size[1],
        "objects": {
            **objects,
            "bbox": [[v * scale for v in b] for b in objects["bbox"]],
            "area": [a * scale**2 for a in objects["area"]],
        },
    }


def build_v4(roboflow_dir: Path, base: str = "kaya-go/moku-v3", seed: int = 0):
    """moku-v4: moku-v3 plus the Roboflow photos, split by photo 50/25/25 into train/validation/test.

    v3's own splits are kept intact (``source_dataset`` tells them apart), so every v3 number can
    still be reproduced on the v3 subset. Training copies are downscaled to 1280 px; evaluation
    copies keep the full phone resolution, as Kaya receives it.
    """
    from datasets import DatasetDict, concatenate_datasets, load_dataset

    v3 = load_dataset(base)
    extra, keys = load_roboflow_coco(roboflow_dir, ROBOFLOW_SOURCE)
    parts = split_by_group(photo_groups(extra, keys), {"train": 0.5, "validation": 0.25, "test": 0.25}, seed)
    out = {}
    for split, indices in parts.items():
        subset = extra.select(indices)
        if split == "train":
            subset = subset.map(_downscale, features=FEATURES)
        base_split = v3[split].cast(FEATURES)
        subset = subset.map(lambda ex, i: {"image_id": len(base_split) + i}, with_indices=True, features=FEATURES)
        out[split] = concatenate_datasets([base_split, subset])
    return DatasetDict(out)
