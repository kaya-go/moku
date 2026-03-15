# Architecture

## Overview

Moku is an object detection pipeline that converts photos of Go boards (goban) into SGF format. The pipeline detects stones and the board itself, then uses geometry to map stone positions to grid coordinates.

## High-Level Flow

```
Photo → Object Detection (ONNX) → Stone Positions → Grid Mapping → SGF String
```

In the Kaya web app, the ONNX model runs in the browser via ONNX Runtime WebAssembly.

## Detection Strategy

We detect 3 object categories:

| ID | Name | Description |
|----|------|-------------|
| 0 | `black_stone` | Individual black stone |
| 1 | `white_stone` | Individual white stone |
| 2 | `board_corner` | Board corner point (small bbox at each corner) |

### Why board_corner instead of full board bbox?

Board corners enable **homography-based perspective correction**: given 4 corner points, we compute a homographic transform mapping image coordinates to a normalized [0,1]×[0,1] board space. This is more robust to angled/tilted photos than using a full board bounding box.

- A 4-point homography corrects for perspective distortion automatically.
- Works even with partial boards (1–3 visible corners).
- Avoids the need to estimate grid spacing from stone density.

### Why not detect empty intersections?

Source datasets annotate empty intersections (`empty`, `empty_edge`, `empty_corner`), but we intentionally drop these because:

- Empty intersections vastly outnumber stones, creating class imbalance.
- Detecting 361 empty intersections on a 19x19 board is wasteful when they can be inferred.
- Board corners + stone positions are sufficient to reconstruct the full board state.
- Fewer categories = simpler model, faster inference, smaller ONNX file.

### Partial Board Views

The model must handle photos where only part of the goban is visible (e.g., 1, 2, or 3 corners showing). This is common in real-world usage (close-up shots, angled photos):

- Only the visible corners and stones are annotated.
- `src/moku/grid.py` handles 1–4 visible corners via the homography solver.
- Dataset v2 should explicitly include partial board images to improve robustness.

### SGF Conversion (downstream, not in this repo)

After detection, the Kaya app will:

1. Use the detected `board_corner` points to compute a homography.
2. Transform all stone centers to normalized board space via homography.
3. Snap to nearest grid intersection based on board size (9, 13, or 19).
4. Generate an SGF string with all stone positions.

## Model Choice: RT-DETR r18vd

**Model**: `PekingU/rtdetr_r18vd` (Real-Time Detection Transformer with ResNet-18vd backbone)

### Why RT-DETR?

| Criterion | RT-DETR r18vd | YOLO variants | Original DETR |
|-----------|---------------|---------------|---------------|
| NMS required | No (end-to-end) | Yes | No |
| ONNX export simplicity | High (no NMS post-proc) | Medium | High |
| Model size | ~20MB | ~6-25MB | ~160MB |
| Browser inference speed | Fast | Fast | Slow |
| HF transformers support | Yes | No (needs ultralytics) | Yes |
| Vendor library needed | No | Yes (ultralytics) | No |

Key advantages:
- **No NMS**: Transformer decoder handles duplicate elimination internally, producing clean ONNX graphs.
- **Small backbone**: ResNet-18vd is lightweight enough for browser inference.
- **HF ecosystem**: Native support in `transformers` library — no vendor dependencies.
- **Good accuracy**: Competitive with YOLO on COCO benchmarks at similar model sizes.

### Training Approach

- Fine-tune from COCO-pretrained weights (`PekingU/rtdetr_r18vd`).
- Use HF `Trainer` API with standard object detection training loop.
- Training on M3 MacBook for small runs; HF Jobs for full training.

### v2 Two-Stage Training Strategy

Training follows a **synthetic pre-train → real fine-tune** approach:

1. **Stage 1 — Synthetic pre-training** (~1500 synthetic images, ~30 epochs, LR=1e-4):
   The model starts from COCO-pretrained weights and learns the general structure of Go boards: grid layout, stone shapes, and corner positions. Synthetic data provides perfect annotations and unlimited diversity (backgrounds, perspectives, lighting).

2. **Stage 2 — Real fine-tuning** (~320 real images, ~500 epochs, LR=2e-4 to 1e-3):
   The model adapts to the real-world domain: camera noise, natural lighting, real wood textures, etc. Multiple LR sweep runs have been performed to find the right balance between preserving stage 1 features and adapting to the real domain.

**Why not mix synthetic + real?** With a ~5:1 synthetic-to-real ratio, naive mixing risks the model over-fitting to synthetic appearance. Two-stage training cleanly separates domain learning from domain adaptation, which consistently outperforms mixing in sim-to-real transfer literature.

## ONNX Export

- Export via `torch.onnx.export` with dynamic axes for batch dimension.
- Target ONNX opset 16+.
- Verify with `onnxruntime` before publishing.
- Published to `kaya-go/moku-v1` on Hugging Face Hub.

## Dataset: kaya-go/moku-v1 → v2

See [dataset.md](dataset.md) for full details on sources, harmonization, and statistics.

## v2 Improvements Plan

The v1 model achieves mAP@50:95 of **0.39**. The bottleneck is data, not architecture. RT-DETR r18vd is retained for v2.

### Augmentation Overhaul

v1 uses only horizontal flip. v2 adds an aggressive albumentations pipeline:

| Type | Transforms |
|------|------------|
| Geometric | `Perspective`, `Rotate` (±15°), `HorizontalFlip`, `VerticalFlip`, `RandomCrop` |
| Photometric | `ColorJitter`, `RandomGamma`, `GaussianBlur`, `MotionBlur`, `RandomShadow` |

`Perspective` is the highest-priority augmentation for `board_corner` detection.

### Synthetic Data Generator (`src/moku/synthetic.py`)

Generates (PIL image, COCO annotations) pairs with perfect corner annotations:

- Wood-grain board texture (Perlin noise)
- Styled stones with specular highlights
- Random perspective distortion
- Lighting variations (vignette, gradient)
- Partial board crops (simulating 1–3 visible corners)

Target: 500 synthetic train / 100 val / 100 test images.

### Pseudo-labeled Real Images (`src/moku/scraper.py`)

Scrape real goban images from Flickr (CC license) and Reddit r/baduk → run v1 model → human review via HTML/JS annotator tool → add corrected samples to v2 training set.

### Corner Re-annotation (`tools/annotator/`)

HTML/JS tool (served via `python -m http.server`) with canvas magnifier for re-annotating suspicious board_corners in the v1 dataset.

### Optional: RT-DETR r34vd

RT-DETR r34vd (ResNet-34 backbone) doubles parameter count with the same ONNX export pipeline. Worth benchmarking after v2 data is assembled — but only if r18vd plateaus.

## Scripts

### `scripts/train.py`

Self-contained training script for HF Jobs. Supports two-stage training (synthetic pre-train, real fine-tune). See inline docstring for usage.

### `scripts/launch_grid.sh`

Shell launcher for HF Jobs. Launches stage 1 and/or stage 2 LR sweep jobs.

### `scripts/delete_runs.py`

Delete specific training runs from the Trackio experiment logs stored in `kaya-go/moku-experiment-logs`.

```bash
# List all runs:
pixi run python scripts/delete_runs.py

# Delete specific runs by name:
pixi run python scripts/delete_runs.py stage2_lr2e-4_run_3 stage2_lr5e-4_run_3

# Delete runs matching a regex pattern:
pixi run python scripts/delete_runs.py --pattern "run_3"

# Delete ALL runs (full reset):
pixi run python scripts/delete_runs.py --all
```
