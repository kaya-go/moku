# Dataset: kaya-go/moku-v1 → v2

## Overview

The dataset on Hugging Face Hub is a harmonized object detection dataset for Go board (goban) recognition. It combines multiple COCO-format datasets from Roboflow into a single unified dataset with consistent categories.

## Harmonized Categories

| ID | Name | Description |
|----|------|-------------|
| 0 | `black_stone` | Individual black stone |
| 1 | `white_stone` | Individual white stone |
| 2 | `board_corner` | Board corner point (small bbox at each corner) |

Board corners enable perspective-corrected grid inference via homography (see [architecture.md](architecture.md)).

## Source Datasets

Raw datasets are stored locally at `/Users/hadim/Data/moku/raw/` (not committed to repo).

### Go Game detection v10 (`Go Game detection.v10i.coco`)

- **Source**: [Roboflow](https://universe.roboflow.com/aleksei-gorovoi-gogamedn/go-game-detection)
- **Images**: 256 (train: 241, valid: 14, test: 1)
- **Original categories**: go-game, black_stone, board, board_corner, empty, empty_corner, empty_edge, white_stone
- **Kept**: board_corner, black_stone, white_stone
- **Dropped**: go-game, empty, empty_corner, empty_edge
- **Notes**: Superset of v1. Contains the same 92 base images as v1 plus 3 additional ones, with Roboflow augmentations applied. Real `board_corner` annotations are used directly.

### Go game detection v1 (`Go game detection.v1i.coco`)

- **Source**: [Roboflow](https://universe.roboflow.com/test-yyxee/go-game-detection-mfkll)
- **Images**: 243 (train: 230, valid: 12, test: 1)
- **Status**: **SKIPPED** — fully contained within v10 (92/92 base images overlap).
- **Original categories**: go-board, black_stone, board, board_corner, empty, empty_corner, empty_edge, white_stone

### go-chess 2 v3 (`go-chess 2.v3-go-chess.v1.coco`)

- **Source**: [Roboflow](https://universe.roboflow.com/zhejiang-af-university-j6jfz/go-chess-2)
- **Images**: 236 (train: 206, valid: 20, test: 10)
- **Original categories**: go-stone-FMbU, black_stone, goboard, white_stone
- **Mapping**: goboard → board_corner (synthetic corners from convex hull of board segmentation polygon), black_stone → black_stone, white_stone → white_stone
- **Dropped**: go-stone-FMbU
- **Notes**: Different source images from the other datasets (0 overlap). Has segmentation polygons used to synthesize `board_corner` bboxes (tight bbox from convex hull of polygon vertices, replacing earlier fixed 20×20 px bboxes).

## Category Harmonization Rules

The harmonization maps source categories to the 3 unified categories:

```python
CATEGORY_MAP = {
    # Go Game detection v10 — real board_corner annotations used directly
    "board_corner": "board_corner",
    "black_stone": "black_stone",
    "white_stone": "white_stone",
    # go-chess 2 v3 — synthetic corners generated from segmentation polygon convex hull
    "goboard": "board_corner",
}
```

All other categories are dropped:
- `go-game`, `go-board`, `go-stone-FMbU` — redundant or unused board-level labels
- `empty`, `empty_edge`, `empty_corner` — empty intersections (inferred from geometry)
- `board` (full bbox) — replaced by 4 `board_corner` points for homography

## HuggingFace Dataset Format

The dataset follows the standard HF object detection format (same schema as `cppe-5`).

Each row in the dataset contains:

| Field | Type | Description |
|-------|------|-------------|
| `image` | Image | The goban photo |
| `image_id` | int | Unique image identifier |
| `width` | int | Image width in pixels |
| `height` | int | Image height in pixels |
| `source_dataset` | str | Source dataset name (extra metadata) |
| `objects` | dict | Bounding box annotations |
| `objects.id` | list[int] | Globally unique annotation IDs |
| `objects.bbox` | list[list[float]] | Bounding boxes in COCO format [x, y, w, h] (absolute pixels) |
| `objects.category` | list[int] | Category IDs (0=black_stone, 1=white_stone, 2=board_corner) |
| `objects.area` | list[float] | Area of each bounding box |
| `objects.iscrowd` | list[int] | Crowd flag (always 0) |

## Split Strategy

**Splits**: train / validation / test (all three).

- **val** is used during training for early stopping and hyperparameter tuning.
- **test** is a held-out set never seen during tuning, used only for final evaluation.

**Why we re-split instead of using Roboflow's splits:**

1. Roboflow augments images (flips, crops, color jitter) from a smaller set of base photos. Multiple augmented variants of the same base image may end up in different splits, causing **data leakage**.
2. The Roboflow splits were made per-dataset before concatenation, so proportions are uneven (e.g., test set of only 11 images).

**Re-split approach:**

- Pool all images from all sources, ignoring Roboflow's original splits.
- Group by **base image name** (strip the Roboflow augmentation hash suffix). All augmented variants of the same base image go to the same split.
- Stratify by `source_dataset` so both sources are represented proportionally in every split.
- Target ratio: **80/10/10** (train/val/test).
- Fixed random seed for reproducibility.

**No cross-validation**: Training RT-DETR is expensive. A single fixed split with a proper held-out test set is standard for fine-tuning large pretrained models.

## Partial Board Views

The model must support photos with partial goban visibility (1–3 corners visible, close-up shots, angled photos). The current 3 categories handle this without extra labels. Dataset v2 explicitly includes partial board images (via synthetic generator crop mode) to improve robustness.

## Reproducibility

The dataset is built by running `notebooks/01_Build_Dataset.ipynb`. The harmonization logic lives in `src/moku/dataset.py` for reusability.

---

## Dataset v2: kaya-go/moku-v2

### Motivation

v1 model reaches mAP@50:95 **0.39**. Main identified weaknesses:
- `board_corner` AP is the lowest of the 3 classes (synthetic 20×20 bboxes in go-chess, limited image variety)
- Only ~320 training images
- Minimal augmentation (horizontal flip only)

### v2 Data Sources

| Source | Type | Approx. size | Notes |
|--------|------|-------------|-------|
| moku-v1 (fixed corners) | Real images | ~320 | go-chess corners replaced with tight hull bboxes |
| Synthetic generator | Generated | ~500 | `src/moku/synthetic.py`, perfect annotations |
| Pseudo-labeled real images | Real images, human-corrected | ~100+ | Scraped from Flickr CC + Reddit r/baduk |

### v2 Augmentation Pipeline

Replaces the v1 horizontal-flip-only pipeline with albumentations:

| Category | Transforms |
|----------|-----------|
| Geometric | `Perspective`, `Rotate` (±15°), `HorizontalFlip`, `VerticalFlip`, `RandomCrop` |
| Photometric | `ColorJitter`, `RandomGamma`, `GaussianBlur`, `MotionBlur`, `RandomShadow` |

`Perspective` distortion is the highest-priority augment for improving `board_corner` detection.

### v2 Build Process

1. Fix go-chess corner bboxes in `dataset.py` (`build_dataset_v2()`)
2. Generate synthetic images with `notebooks/05_Generate_Synthetic.ipynb`
3. Scrape + pseudo-label + correct real images with `notebooks/06_Pseudolabel.ipynb`
4. Combine all sources with the same stratified 80/10/10 re-split
5. Push `kaya-go/moku-v2` to HF Hub via updated `notebooks/01_Build_Dataset.ipynb`
