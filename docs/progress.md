# Progress

## Current Status

**Phase**: v3 dataset expansion — scrape + pseudo-label + manual correction

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
- [ ] Round 6: long training sweep (4 runs, 1000 epochs) — in progress
- [ ] Push best model as `kaya-go/moku-v2` on HF Hub

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

## v3 Plan — Data-Centric Expansion

**Goal**: 3–5× the real training images via scraping + pseudo-labeling + manual correction. HP tuning is saturated on 382 images — more real data is the only remaining lever.

**v2 model** (`kaya-go/moku-v2`, best R6 artifact): used for pseudo-labeling.

### Phase 1 — Image Scraping (`src/moku/scraper.py`, `notebooks/05_Scrape_Images.ipynb`)

Collect as many real goban photos as possible from public sources:

| Source | Method | License | Target |
|--------|--------|---------|--------|
| Flickr | API search (`go board`, `goban`, `baduk`, `weiqi`, `围棋`, `碁`) | CC-BY / CC0 | 500–1000 |
| Reddit r/baduk | PRAW or pushshift API, image posts | Public | 200–500 |
| Wikimedia Commons | API search | CC / public domain | 50–100 |
| Google Images (optional) | Manual download or `icrawler` | Fair use / CC only | 100–300 |

**Implementation**:
- [ ] Create `src/moku/scraper.py`: async scraper using `httpx` for Flickr API + Reddit JSON API + Wikimedia API
  - Flickr: search by tags, download medium/large size, save metadata (license, author, URL)
  - Reddit: fetch r/baduk image posts (i.redd.it, imgur links)
  - Wikimedia: search "Go (game)" category
- [ ] Deduplicate by perceptual hash (dhash) — avoid near-duplicate images
- [ ] Filter out non-goban images (simple heuristic: aspect ratio ~1:1, min size 400×400)
- [ ] Save scraped images to `data/scrape/images/` with metadata JSON
- [ ] Create `notebooks/05_Scrape_Images.ipynb`: run scraper, preview results, push to staging

### Phase 2 — Pseudo-Labeling (`notebooks/06_Pseudo_Label.ipynb`)

Run v2 model inference on all scraped images to generate initial annotations:

- [ ] Load best v2 model from HF Hub (`kaya-go/moku-v2`)
- [ ] Run batch inference on scraped images (confidence threshold ~0.3 to keep borderline detections for human review)
- [ ] Convert model outputs to COCO annotation format (same as `images.json` for annotator)
- [ ] Flag images for review:
  - `no_board`: 0 corners detected → likely not a goban → candidate for exclusion
  - `partial_board`: 1–3 corners → keep but flag
  - `low_confidence`: any detection < 0.5 confidence
  - `high_stone_count`: > 200 stones (suspicious)
- [ ] Generate `data/scrape/images.json` in annotator format
- [ ] Preview pseudo-labels: grid of images with predicted bboxes overlaid

### Phase 3 — Manual Correction (`tools/annotator/`)

Use the existing HTML annotator to review and correct pseudo-labels:

- [ ] **Add exclude/skip functionality** to annotator:
  - New button "Exclude" (or `X` shortcut) to mark an image as excluded
  - Excluded images saved in `corrected.json` with `{"excluded": true}` flag
  - Excluded images shown dimmed in sidebar, filterable
  - New filter: "excluded" / "not excluded"
- [ ] **Add confidence display**: show model confidence scores on each bbox for review guidance
- [ ] Workflow:
  1. Start annotator: `python tools/annotator/server.py --data-dir data/scrape`
  2. Filter "flagged" images first (no_board, low_confidence) to quickly exclude non-goban images
  3. Review remaining images: correct stone/corner positions, exclude bad images
  4. Corrections saved to `data/scrape/corrected.json`

### Phase 4 — Dataset v3 Build (`notebooks/07_Build_Dataset_v3.ipynb`)

Combine existing v2 data + newly annotated scraped images:

- [ ] Load corrected scraped images (excluding flagged-as-excluded)
- [ ] Merge with existing v2 real data (382 images)
- [ ] Re-split all real images: 80/10/10 by base image name (stratified by source)
- [ ] Keep synthetic data unchanged (2500/500/500) for stage 1 pre-training
- [ ] Push as `kaya-go/moku-v3` on HF Hub (real + synthetic configs)
- [ ] Update `src/moku/dataset.py` with v3 build utilities

### Phase 5 — v3 Training

Two-stage training (same strategy as v2, re-tuned for larger dataset):

- [ ] Stage 1: synthetic pre-train (reuse v2 stage 1 checkpoint or re-train)
- [ ] Stage 2: real fine-tune on v3 dataset
- [ ] With 3–5× more real data, may need fewer epochs (200–500 vs 1000)
- [ ] LR re-sweep: start from v2 best (lr=3e-4 or 4e-4, linear/cosine_with_min_lr)
- [ ] Evaluate on v3 test set + v2 test set (backward compatibility check)

### v3 Milestones

| # | Milestone | Status |
|---|-----------|--------|
| 1 | Scraper code (`src/moku/scraper.py`) | not started |
| 2 | Scrape images notebook (`05_Scrape_Images.ipynb`) | not started |
| 3 | Pseudo-labeling notebook (`06_Pseudo_Label.ipynb`) | not started |
| 4 | Annotator: add exclude/skip feature | not started |
| 5 | Manual correction pass | not started |
| 6 | Build v3 dataset (`07_Build_Dataset_v3.ipynb`) | not started |
| 7 | v3 training round | not started |

---

## Files to Create / Modify

| File | Change |
|------|--------|
| `src/moku/model.py` | Model loading, eval transforms, collation utilities |
| `src/moku/runs.py` | W&B API utilities (fetch runs, artifacts, load models) |
| `src/moku/dataset.py` | Fix go-chess corners, dataset harmonization, v3 build |
| `src/moku/synthetic.py` | Synthetic goban generator |
| `src/moku/scraper.py` | **NEW** — Flickr / Reddit / Wikimedia image scraper |
| `scripts/train.py` | Two-stage training, W&B artifact saving |
| `tools/annotator/` | HTML/JS annotator app + exclude/skip feature |
| `notebooks/05_Scrape_Images.ipynb` | **NEW** — Run scraper, preview, deduplicate |
| `notebooks/06_Pseudo_Label.ipynb` | **NEW** — Pseudo-label scraped images with v2 model |
| `notebooks/07_Build_Dataset_v3.ipynb` | **NEW** — Merge v2 + scraped → v3 dataset |
| `notebooks/10_Train_Model.ipynb` | Two-stage training (local test + HF Jobs) |
| `notebooks/20_Analyze_Runs.ipynb` | W&B run analysis (loss/mAP curves) |
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
| 2026-03-22 | v3 data-centric strategy: scrape + pseudo-label + correct | HP tuning saturated on 382 images; more real data is the only remaining lever for mAP gains |
| 2026-03-22 | Scrape sources: Flickr CC, Reddit r/baduk, Wikimedia | Free, diverse, real-world images; CC/public domain licenses |
| 2026-03-22 | Add exclude/skip to annotator | Pseudo-labeled scraped images include non-goban images that must be filtered out during correction |
| 2026-03-22 | Keep synthetic pre-train for v3 | Two-stage training proved its value (corner AP +50%); reuse v2 stage 1 or re-train |
