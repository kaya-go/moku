# Progress

## Current Status

**Phase**: v3 model published — `model-r10_os3_lr3e-4_cosmin100` selected as moku-v3

## v1 Results

- **Dataset**: `kaya-go/moku-v1` — ~320 train images, 3 categories (black_stone, white_stone, board_corner)
- **Model**: `kaya-go/moku-v1` — RT-DETR r18vd, fine-tuned 50 epochs
- **Test mAP@50:95**: **0.39** — moderate; data quality/quantity is the main bottleneck
- **Main weakness**: `board_corner` detections — partially due to synthetic 20×20 bboxes in go-chess source, minimal augmentation, and small dataset size

### v1 Metrics (baseline)

| Metric | Value |
|--------|-------|
| mAP@50:95 | 0.3995 |
| mAP@50 | 0.7207 |
| mAP@75 | 0.3725 |
| mAR@400 | 0.5072 |

| Category | AP@50:95 |
|----------|----------|
| black_stone | 0.5135 |
| white_stone | 0.4706 |
| board_corner | 0.2142 |

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
- [x] Created training utilities in `src/moku/model.py`
- [x] Created `01_Build_Dataset.ipynb` — built and pushed `kaya-go/moku-v1`
- [x] Created `02_Train_Model.ipynb` — baseline + HP grid search training
- [x] Created `03_Evaluate.ipynb` — mAP evaluation, prediction visualization, run comparison
- [x] Created `04_Export_ONNX.ipynb` — ONNX export and artifact saved at `artifacts/moku-v1.onnx`

---

## v2 Plan

**Goal**: Push mAP from 0.39 → ~0.55–0.65 via better data quality, aggressive augmentation, synthetic data, and two-stage training.

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

- [x] `pixi add albumentations`
- [x] Rewrite transforms in `scripts/train.py` using albumentations:
  - **Geometric**: `Perspective` (most impactful for corners), `Rotate` (±15°), `HorizontalFlip`, `VerticalFlip`, `RandomResizedCrop`
  - **Photometric**: `ColorJitter`, `RandomGamma`, `GaussianBlur`, `MotionBlur`, `RandomShadow`
  - Bbox conversion handled natively by albumentations COCO format with min_area/min_visibility filtering
- [ ] **Benchmark isolation run**: train on v1 dataset + new augmentation only to measure delta before adding new data

### Phase 3 — Synthetic Data Generator

- [x] Create `src/moku/synthetic.py`:
  - Diverse backgrounds (7 wood palettes + 5 solid surfaces)
  - 3D perspective transform (pitch/yaw/roll via pinhole camera model)
  - Lighting simulation (brightness/contrast, color temperature, directional light, vignette)
  - Stone position jitter (Gaussian offset from intersection center)
  - Corner visibility guarantee (retry with reduced angles if corners leave bounds)
  - Output: (PIL image, COCO annotations) with perfect corner annotations
- [ ] New `notebooks/05_Generate_Synthetic.ipynb`: preview + generate 1000/250/250 + push HF

### Phase 4 — Dataset v2 Build

- [ ] Update `src/moku/dataset.py`: add `build_dataset_v2()` combining v1 (fixed corners) + synthetic
- [ ] Update `notebooks/01_Build_Dataset.ipynb` with v2 section
- [ ] Push `kaya-go/moku-v2` to HF Hub

### Phase 5 — Model v2: Two-Stage Training

Training uses a **synthetic pre-train → real fine-tune** strategy:

| | Stage 1: Pre-train | Stage 2: Fine-tune |
|---|---|---|
| **Data** | ~1500 synthetic images | ~320 real images (v1 dataset, fixed corners) |
| **Purpose** | Learn general structure (grid, stones, corners) | Adapt to real-world domain (textures, camera noise) |
| **Epochs** | ~30 | ~500 |
| **Learning rate** | 1e-4 (standard) | Sweep: 2e-4, 5e-4, 1e-3 (run 3) |
| **HP tuning** | None needed — just verify loss decreases | LR sweep across 3 values |
| **WD / batch size** | Same as v1 | Same as v1 |

**Why two-stage over mixing?**
Mixing synthetic + real data risks over-representing the synthetic domain (4:1 ratio). Two-stage training lets the model first learn the general structure, then cleanly adapt to the real domain. The literature (Domain Randomization, sim-to-real transfer) consistently shows two-stage outperforms naive mixing when the domain gap is significant.

**HP tuning rationale**: Full grid search is unnecessary for v2. The critical HP is the stage 2 LR — too high destroys pre-trained features, too low under-fits the real domain. 3 runs per sweep is sufficient to find the sweet spot.

**Training runs history**:
- Run 1: LR sweep 1e-5, 2e-5, 5e-5 (50 epochs) — under-fit, mAP well below v1
- Run 2: LR sweep 5e-5, 8e-5, 1e-4, 2e-4 (50 epochs) — still below v1 mAP
- Run 3 (current): LR sweep 2e-4, 5e-4, 1e-3 (500 epochs) — trying higher LRs and longer training

### v2 Run 2 Metrics — Stage 2 lr=2e-4, 50 epochs (real test)

| Metric | Value |
|--------|-------|
| mAP@50:95 | 0.3380 |
| mAP@50 | 0.5994 |
| mAP@75 | 0.3569 |
| mAR@400 | 0.4232 |

| Category | AP@50:95 |
|----------|----------|
| black_stone | 0.2745 |
| white_stone | 0.4211 |
| board_corner | 0.3183 |

**Observations**: v2 lr=2e-4 (50 epochs) trails v1 overall (0.338 vs 0.400 mAP), but board_corner AP improved significantly (0.318 vs 0.214) thanks to corrected corner annotations and synthetic pre-training. Stone detection dropped — likely due to too-short fine-tuning (50 epochs). Run 3 with 500 epochs should recover stone performance.

- [x] Update `notebooks/10_Train_Model.ipynb` with two-stage training sections
- [x] Stage 1: pre-train on synthetic dataset
- [x] Stage 2: fine-tune on real dataset — run 3 with LR sweep (2e-4, 5e-4, 1e-3)
- [x] Round 4: 10-run HP grid search (LR, scheduler, weight decay, combined strategies)
- [x] W&B artifact saving for best model checkpoints (by mAP@50:95)
- [x] `20_Analyze_Runs.ipynb`: W&B run analysis (loss/mAP curves, ranking)
- [x] `30_Publish_Model.ipynb`: select W&B artifact → push to HF Hub
- [x] Round 5: LR + scheduler sweep (6 runs, 500 epochs, with fixed augmentation)
- [x] Round 6: long training sweep (4 runs, 1000 epochs)
- [x] Push best model as `kaya-go/moku-v2` on HF Hub (`model-r6_lr4e-4_cosmin1000`)

### Round 5 Results (6 runs, 500 epochs)

**Config**: LR {2e-4, 3e-4, 4e-4, 5e-4} × scheduler {linear, cosine}, batch 32, fixed albumentations augmentation.

| Run | LR | Scheduler | Best mAP@50:95 | Best Epoch | Mean mAP (last 100) |
|-----|----|-----------|---------------|------------|---------------------|
| r5_lr4e-4_linear500 | 4e-4 | linear | **0.6075** | 413 | 0.5092 |
| r5_lr3e-4_cos500 | 3e-4 | cosine | 0.5841 | 318 | **0.5365** |
| r5_lr5e-4_linear500 | 5e-4 | linear | 0.5742 | 369 | 0.4933 |
| r5_lr2e-4_cos500 | 2e-4 | cosine | 0.5577 | 354 | 0.5043 |
| r5_lr5e-4_cos500 | 5e-4 | cosine | 0.5355 | 180 | 0.4173 |
| r5_lr2e-4_linear500 | 2e-4 | linear | 0.5280 | 410 | 0.4889 |

**Key findings**:
- Best raw mAP: lr=4e-4 + linear (0.6075), but noisy/spiky eval curve
- Best smoothed/mean mAP: lr=3e-4 + cosine — higher average performance across all 100-epoch windows
- Linear schedule generally outperforms cosine on long runs (cosine kills LR too early)
- Augmentation (fixed in R5) significantly improved over R4
- Best runs still improving at epoch 400-500 → not plateaued, longer training warranted

### Round 6 Plan (4 runs, 1000 epochs) — in progress

**Goal**: Double training length to find true plateau; combine best LRs with improved scheduler.

| Run | LR | Scheduler | Epochs |
|-----|----|-----------|--------|
| r6_lr4e-4_linear1000 | 4e-4 | linear | 1000 |
| r6_lr3e-4_linear1000 | 3e-4 | linear | 1000 |
| r6_lr4e-4_cosmin1000 | 4e-4 | cosine_with_min_lr (min=5e-5) | 1000 |
| r6_lr3e-4_cosmin1000 | 3e-4 | cosine_with_min_lr (min=5e-5) | 1000 |

**Rationale**: lr=3e-4 included because smoothed analysis showed it competitive with 4e-4. `cosine_with_min_lr` avoids the LR=0 problem of plain cosine while maintaining the annealing benefit.

---

## v3 — Data-Centric Expansion via Gemini Style Transfer

**Goal**: Expand real training data via Gemini-generated photorealistic goban images. HP tuning is saturated on 382 images — more diverse data is the main lever.

**Approach**: Use the synthetic generator (`src/moku/synthetic.py`) to produce images with perfect annotations, then use Gemini (`src/moku/generate.py`) for **style transfer** — transforming synthetic images into photorealistic photographs while preserving the exact board position. Annotations are **inherited directly** from the synthetic input because Gemini preserves stone positions and camera angle. No pseudo-labeling or manual correction is needed.

### Phase 1 — Synthetic → Photorealistic Generation

- [x] Create `src/moku/generate.py`: Gemini style transfer (synthetic → photorealistic)
- [x] `src/moku/synthetic.py` generates input images with perfect COCO annotations
- [x] Gemini transforms synthetic images into photorealistic photos while preserving stone positions
- [x] Generate 1000 images at 1024×1024 → `data/annotate_generated/images/`
- [x] Annotations inherited from synthetic inputs (stored as per-image JSON files)

### Phase 2 — Verification (`notebooks/06_Annotate_Generated.ipynb`)

- [x] Visual spot-checks confirm Gemini preserves stone positions accurately
- [x] Generate `data/annotate_generated/images.json` for metadata
- [x] No manual correction needed — synthetic annotations are reliable

### Phase 3 — Dataset v3 Build (`notebooks/07_Build_Dataset_v3.ipynb`)

- [x] `load_annotated_generated()` utility in `src/moku/dataset.py`
- [x] `07_Build_Dataset_v3.ipynb` notebook created
- [x] Build and push v3 with 1000 generated images

**v3 data composition**:
- **train**: v2 real train (~306) + 1000 generated images = ~1306 total
- **val/test**: unchanged from v2 (real images only, ~38 each)
- No synthetic split — v3 dropped procedural synthetic data entirely
- **HF Hub**: `kaya-go/moku-v3` (single config, no sub-configs)

### Phase 4 — v3 Training

**Strategy chosen**: Single-stage training (COCO pretrained → fine-tune on all v3 data)
- Simpler pipeline, no two-stage complexity
- v3 dataset has enough real+generated data to train directly

- [x] Update `scripts/train.py` with `--dataset` and `--dataset-config` args for v3 support
- [x] Round 7: first HP grid search on v3 (4 runs, 500 epochs, bs=32)
- [x] Round 8: batch_size=128, LR sweep (3 runs, 500 epochs)
- [x] Evaluate on v3 test set (same as v2 test set for direct comparison)
- [x] Round 9: oversample real images (9 runs, 500 epochs)
- [x] Round 10: calibrated scheduler + __getitems__ fix (5 runs, 80-100 epochs)
- [x] Publish best r10 model as `kaya-go/moku-v3`

### Critical Bug Fixed: `__getitems__` batching (2026-03-23)

PyTorch ≥ 2.4 `DataLoader` calls `dataset.__getitems__([i,j,...])` instead of per-item `__getitem__(i)`. HF `datasets` with `set_transform` passes ALL indices as a batch to the transform function. The `_unbatch_example()` helper in `train.py` took only the FIRST item, discarding the rest → effective batch_size=1 regardless of configured batch_size. **All rounds r5–r9 were affected.** Fix: `dataset[split].__getitems__ = None` after `set_transform()` to force per-item indexing.

### r7/r8 Results — Generated Data Dilution

**Problem identified**: v3 single-stage underperforms v2 two-stage (r5/r6). The 1000 generated images (~77% of train) dilute the real image signal. Val/test sets are 100% real, so the model overfits to the generated domain.

| Round | Dataset | Mode | Best val mAP@50 | Best test mAP@50 | Test corner_R4 |
|-------|---------|------|-----------------|-----------------|----------------|
| r5 (v2) | 306 real | two-stage | 0.921 | 0.785 | 0.679 |
| r6 (v2) | 306 real | two-stage | 0.776 | 0.807 | 0.893 |
| r7 (v3) | 306 real + 1000 gen | single-stage, bs=32 | 0.624 | 0.448 | 0.036 |
| r8 (v3) | 306 real + 1000 gen | single-stage, bs=128 | 0.774 | 0.582 | 0.393 |

### Round 9 Results (9 runs, 500 epochs)

**Goal**: Fix data dilution via real image oversampling.

**Note**: r9 was affected by the `__getitems__` bug (effective bs=1). Metrics are not reliable.

Best runs (by peak mAP@50 on val):

| Run | mAP@50 | corner_R4 | stone_cdAP |
|-----|--------|-----------|------------|
| r9_os5_lr5e-4_cosmin500 | 0.900 | 0.881 | 0.926 |
| r9_os5_lr3e-4_cosmin500 | 0.900 | 0.887 | 0.943 |
| r9_os3_lr3e-4_cosmin500 | 0.859 | 0.826 | 0.946 |

**Finding**: OS5 (5× real oversampling) consistently outperformed OS3 and no-oversample on val metrics. However, all r9 runs crashed (4h timeout with 500 epochs was too long).

### Round 10 Results (5 runs, 80-100 epochs) — SELECTED

**Goal**: Calibrated scheduler (epochs matched to 4h budget), fix `__getitems__` bug.

**Changes from r9**:
- `__getitems__` bug fixed → real batch_size=16
- num_epochs calibrated: 80 (OS5), 100 (OS3) to fit 4h budget
- warmup_ratio=0.05 (was 0.10 in r9 → 60% of training was warmup)
- LR range: 1e-4–3e-4 (narrowed from r9 sweet spot)

| Run | Best ep | val mAP@50 | val corner_R4 | val stone_cdAP |
|-----|---------|-----------|---------------|----------------|
| r10_os3_lr2e-4_cosmin100 | 27 | **0.936** | 0.901 | **0.983** |
| r10_os5_lr2e-4_cosmin80 | 52 | 0.924 | **0.906** | 0.980 |
| r10_os5_lr3e-4_cosmin80 | 50 | 0.892 | 0.815 | 0.974 |
| r10_os3_lr3e-4_cosmin100 | 39 | 0.889 | 0.840 | 0.944 |
| r10_os5_lr1e-4_cosmin80 | 26 | 0.844 | 0.717 | 0.922 |

**Test set evaluation** (all r10 + best from r5-r9 for comparison):

| Model | test mAP@50 | test corner_R4 | test stone_cdAP |
|-------|------------|----------------|------------------|
| **r10_os3_lr3e-4_cosmin100** | **0.862** | **0.825** | 0.919 |
| r10_os3_lr2e-4_cosmin100 | 0.833 | 0.750 | 0.937 |
| r10_os5_lr1e-4_cosmin80 | 0.816 | 0.725 | 0.917 |
| r6_lr4e-4_cosmin1000 (moku-v2) | 0.724 | 0.810 | 0.933 |

**Selected model**: `model-r10_os3_lr3e-4_cosmin100` as **moku-v3**
- Best test mAP@50 (0.862) and corner_R4 (0.825)
- Small val→test gap (0.902→0.862) — minimal overfitting
- +14 points test mAP@50 over moku-v2

### v3 Milestones

| # | Milestone | Status |
|---|-----------|--------|
| 1 | Synthetic image generation (1000 inputs) | done |
| 2 | Gemini style transfer (1000 images) | done |
| 3 | Annotation verification (inherited from synthetic) | done |
| 4 | `load_annotated_generated()` in dataset.py | done |
| 5 | `07_Build_Dataset_v3.ipynb` notebook | done |
| 6 | Build & push v3 dataset (1000 gen images) | done |
| 7 | Round 7: single-stage HP grid search | done |
| 8 | Round 8: batch_size=128, LR sweep | done |
| 9 | Fix `__getitems__` batching bug | done |
| 10 | Round 9: oversample real images | done |
| 11 | Round 10: calibrated scheduler post-fix | done |
| 12 | Select & publish moku-v3 model | done |

---

## Files to Create / Modify

| File | Change |
|------|--------|
| `src/moku/model.py` | Model loading, eval transforms, collation utilities |
| `src/moku/runs.py` | W&B API utilities (fetch runs, artifacts, load models) |
| `src/moku/dataset.py` | Harmonization, corner corrections, v3 build (`load_annotated_generated`) |
| `src/moku/synthetic.py` | Synthetic goban generator |
| `src/moku/generate.py` | Gemini image generation for goban photos |
| `src/moku/annotator.py` | Annotator server utilities |
| `src/moku/grid.py` | HP grid search helpers |
| `scripts/train.py` | Two-stage training, W&B artifact saving |
| `tools/annotator/` | HTML/JS annotator app |
| `notebooks/05_Generate_Images.ipynb` | Gemini image generation |
| `notebooks/06_Annotate_Generated.ipynb` | Pseudo-label + correct generated images |
| `notebooks/07_Build_Dataset_v3.ipynb` | Merge v2 + generated → v3 dataset |
| `notebooks/21_Evaluate.ipynb` | mAP + center-distance eval (HF Hub or W&B artifacts) |
| `notebooks/30_Publish_Model.ipynb` | Select W&B artifact → push to HF Hub |
| `notebooks/40_Export_ONNX.ipynb` | ONNX export for browser inference |

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
| 2026-03-14 | Enhanced synthetic generator: 3D perspective, diverse backgrounds, lighting, stone jitter | Bridges domain gap to real photos; improves corner detection robustness |
| 2026-03-14 | Two-stage training (synthetic pre-train → real fine-tune) | Better than mixing when domain gap is significant; avoids synthetic over-representation |
| 2026-03-14 | Defer pseudo-labeling to post-v2 | Focus v2 on synthetic + improved real data; pseudo-labeling adds complexity for uncertain gain at current model quality |
| 2026-03-14 | Minimal HP tuning: LR sweep only (3 values) at stage 2 | Stage 1 HP insensitive; stage 2 LR is the only critical HP; 3 runs sufficient |
| 2026-03-14 | Run 1–2 LR sweeps under-performed v1 mAP (0.39) | Low LRs (1e-5 to 2e-4) and short training (50 epochs) insufficient |
| 2026-03-14 | Run 3: higher LRs (2e-4, 5e-4, 1e-3) + 500 epochs | Trying stronger fine-tuning signal and longer training to recover v1 mAP |
| 2026-03-15 | W&B artifact saving: best mAP@50:95 only | Direct quality metric; eval_loss is noisy proxy for DETR; one artifact per improvement |
| 2026-03-15 | Replaced local `runs/` analysis with W&B API | Training on HF Jobs; no persistent local dir; W&B is source of truth |
| 2026-03-15 | Notebook numbering: 0x=data, 1x=train, 2x=eval, 3x=publish, 4x=export | Clear section-based organization |
| 2026-03-15 | Round 4: 10-run HP grid (LR, scheduler, weight decay) | Systematic exploration after r3 showed lr=1e-3 best; 200-epoch runs |
| 2026-03-16 | Round 5: LR × scheduler sweep, 500 epochs | Fixed augmentation (R4 had buggy albumentations); 6 runs to find best LR+scheduler combo |
| 2026-03-16 | Round 6: 1000 epochs, 4 runs, cosine_with_min_lr scheduler | R5 not plateaued at 500; plain cosine kills LR too early; cosine_with_min_lr (min=5e-5) keeps a floor |
| 2026-03-16 | Include lr=3e-4 in R6 grid | EMA-smoothed analysis showed r5_lr3e-4_cos had higher mean mAP than lr=4e-4 despite lower raw max |
| 2026-03-16 | Keep batch_size=32 (not 64) | 382 train images / 64 = 6 steps/epoch — too few for stable gradient estimation |
| 2026-03-16 | Create `scripts/analyze_runs.py` | Reusable W&B analysis with EMA smoothing and plateau detection; replaces ad-hoc notebook/scripts |
| 2026-03-22 | v3 data-centric strategy: Gemini style transfer from synthetic inputs | HP tuning saturated on 382 images; more diverse data is the main lever for mAP gains |
| 2026-03-22 | Gemini style transfer over web scraping | Better quality control, no licensing issues, annotations inherited from synthetic inputs (no labeling needed) |
| 2026-03-22 | v3 test/val identical to v2 | Direct metric comparison between v2 and v3; new images go to train only |
| 2026-03-22 | Drop synthetic dataset in v3 | v3 uses Gemini-generated photorealistic images instead; synthetic data only used as input for style transfer |
| 2026-03-22 | No manual annotation for generated images | Gemini preserves stone positions from synthetic input; annotations inherited directly |
| 2026-03-22 | v3 single-stage training over two-stage | v3 has enough real+generated data; simpler pipeline; no synthetic split to pre-train on |
| 2026-03-22 | Round 7: broad LR sweep {1e-4, 3e-4, 5e-4} + linear/cosine_with_min_lr, 500ep | First round on v3; need to find right LR regime for single-stage from COCO pretrained |
| 2026-03-23 | Fixed `__getitems__` batching bug | PyTorch ≥ 2.4 DataLoader + HF datasets set_transform → effective bs=1; all r5-r9 affected |
| 2026-03-23 | Round 10: calibrated scheduler, 80-100 epochs | Budget-matched epochs, warmup_ratio=0.05, LR 1e-4–3e-4, OS3/OS5 |
| 2026-03-23 | Selected `model-r10_os3_lr3e-4_cosmin100` as moku-v3 | Best test mAP@50 (0.862), corner_R4 (0.825); +14 pts over moku-v2 |
