# Progress

## Current Status

**Phase**: Dataset v2 & Model v2 planning

## v1 Results

- **Dataset**: `kaya-go/moku-v1` — ~320 train images, 3 categories (black_stone, white_stone, board_corner)
- **Model**: `kaya-go/moku-v1` — RT-DETR r18vd, fine-tuned 50 epochs
- **Test mAP@50:95**: **0.39** — moderate; data quality/quantity is the main bottleneck
- **Main weakness**: `board_corner` detections — partially due to synthetic 20×20 bboxes in go-chess source, minimal augmentation, and small dataset size

## v1 Completed

- [x] Collected 3 raw COCO datasets from Roboflow
- [x] Analyzed dataset structures, categories, and overlap
- [x] Confirmed v1 is a subset of v10 (skip v1)
- [x] Confirmed go-chess has no overlap with v10 (different source images)
- [x] Defined harmonized category scheme: black_stone (0), white_stone (1), board_corner (2)
- [x] Chose RT-DETR r18vd as the target model
- [x] Set up project structure with `src/moku/` library and notebooks
- [x] Created documentation (AGENTS.md, docs/)
- [x] Built dataset harmonization code in `src/moku/dataset.py`
- [x] Created training utilities in `src/moku/training.py`
- [x] Created `01_Build_Dataset.ipynb` — built and pushed `kaya-go/moku-v1`
- [x] Created `02_Train_Model.ipynb` — baseline + HP grid search training
- [x] Created `03_Evaluate.ipynb` — mAP evaluation, prediction visualization, run comparison
- [x] Created `04_Export_ONNX.ipynb` — ONNX export and artifact saved at `artifacts/moku-v1.onnx`

---

## v2 Plan

**Goal**: Push mAP from 0.39 → ~0.55–0.65 via better data quality, aggressive augmentation, and more training images.

**Key insight**: Architecture is not the bottleneck. RT-DETR r18vd stays. Data is.

### Phase 0 — Evaluation Enhancement *(quick win)*

- [ ] Surface per-class AP breakdown in `03_Evaluate.ipynb` (black_stone AP, white_stone AP, board_corner AP separately as a DataFrame)
- Confirms hypothesis that `board_corner` AP is significantly lower than stones

### Phase 1 — Data Quality: Corner Re-annotation

- [ ] **Audit script/cell**: flag suspicious board_corners (wrong quadrant, unexpected size ratio)
- [ ] **Fix go-chess synthetic corners** in `dataset.py`: replace fixed 20×20 bboxes with tight bboxes derived from convex hull of board segmentation polygon vertices
- [ ] **Build HTML/JS annotator** (`tools/annotator/`):
  - Served locally via `python -m http.server` (zero extra deps)
  - Magnifier overlay (canvas zoom on cursor) for precise corner placement
  - Click to place/drag corner bboxes, save annotations as JSON
  - `notebooks/05_Annotate.ipynb` to drive the re-annotation workflow

### Phase 2 — Aggressive Augmentation

- [ ] `pixi add albumentations`
- [ ] Rewrite transforms in `src/moku/training.py` using albumentations:
  - **Geometric**: `Perspective` (most impactful for corners), `Rotate` (±15°), `HorizontalFlip`, `VerticalFlip`, `RandomCrop`
  - **Photometric**: `ColorJitter`, `RandomGamma`, `GaussianBlur`, `MotionBlur`, `RandomShadow`
  - Adapt bbox conversion COCO ↔ albumentations format
- [ ] **Benchmark isolation run**: train on v1 dataset + new augmentation only to measure delta before adding new data

### Phase 3 — Synthetic Data Generator

- [ ] Create `src/moku/synthetic.py`:
  - `generate_board_texture()`: Perlin/simplex noise tinted brown → wood grain
  - `draw_grid()`: proportioned grid lines
  - `draw_stone(pos, color)`: ellipse + specular highlight + drop shadow
  - `apply_perspective()`: random homographic distortion
  - `add_lighting()`: vignette + gradient overlay
  - `crop_partial_board()`: simulate partial views (1–3 visible corners)
  - Output: (PIL image, COCO annotations) with perfect corner annotations
- [ ] New `notebooks/05_Generate_Synthetic.ipynb`: preview + generate 500/100/100 + push HF

### Phase 4 — Pseudo-labeling on Real Scraped Images

- [ ] Create `src/moku/scraper.py`:
  - Flickr API (search "go game goban baduk", CC license filter)
  - Reddit r/baduk (public JSON API, top posts image URLs)
- [ ] New `notebooks/06_Pseudolabel.ipynb`:
  - Scrape ≥100 images from Flickr + Reddit
  - Run v1 model → pseudo-annotations
  - Open annotator for human review/correction
  - Export corrected annotations to COCO format

### Phase 5 — Dataset v2 Build

- [ ] Update `src/moku/dataset.py`: add `build_dataset_v2()` combining v1 (fixed corners) + synthetic + pseudo-labeled
- [ ] Update `notebooks/01_Build_Dataset.ipynb` with v2 section
- [ ] Push `kaya-go/moku-v2` to HF Hub

### Phase 6 — Model v2 Training

- [ ] Update `notebooks/02_Train_Model.ipynb` to target `kaya-go/moku-v2`
- [ ] HP tuning grid with new augmentation pipeline (same LR × WD grid)
- [ ] Optional: compare RT-DETR r34vd backbone — 2× params, same ONNX pipeline
- [ ] Push best model as `kaya-go/moku-v2` on HF Hub

---

## Files to Create / Modify

| File | Change |
|------|--------|
| `src/moku/training.py` | Rewrite augmentation pipeline (albumentations) |
| `src/moku/dataset.py` | Fix go-chess corners, add `build_dataset_v2()` |
| `src/moku/synthetic.py` | **NEW** — synthetic goban generator |
| `src/moku/scraper.py` | **NEW** — web image scraper (Flickr, Reddit) |
| `tools/annotator/` | **NEW** — HTML/JS annotator app |
| `notebooks/03_Evaluate.ipynb` | Per-class mAP table |
| `notebooks/01_Build_Dataset.ipynb` | v2 section |
| `notebooks/02_Train_Model.ipynb` | Point to `kaya-go/moku-v2` |
| `notebooks/05_Generate_Synthetic.ipynb` | **NEW** |
| `notebooks/06_Pseudolabel.ipynb` | **NEW** |

---

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-28 | Skip Go game detection v1 dataset | Fully contained in v10 (92/92 base images overlap) |
| 2026-02-28 | Drop empty intersection categories | Inferred from geometry; reduces model complexity |
| 2026-02-28 | Use RT-DETR r18vd | No NMS, small model, HF transformers native, no vendor libs |
| 2026-02-28 | 3 categories: black_stone, white_stone, board_corner | Minimal set for SGF; corners enable homography-based grid inference |
| 2026-02-28 | Simple grid search over LR × WD | Small dataset; LR is most impactful HP; 8 runs is manageable |
| 2026-02-28 | Random horizontal flip as sole augmentation | Minimal viable v1; albumentations pipeline planned for v2 |
| 2026-03-13 | Keep RT-DETR r18vd for v2 | Architecture not the bottleneck at current dataset size; try r34vd only as optional experiment |
| 2026-03-13 | HTML/JS annotator (not Gradio) | Zero extra deps, magnifier trivial in canvas, no persistent server needed |
| 2026-03-13 | LLM image generation not prioritized | High cost, inconsistent quality for precise annotations; synthetic generator gives full control |
| 2026-03-13 | Scrape from Flickr CC + Reddit r/baduk | Free, real-world variety; Sensei's Library + BGG as secondary sources |
