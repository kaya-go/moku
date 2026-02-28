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

1. **board** — Bounding box of the full Go board. Used to define the playing area and for perspective/crop normalization.
2. **black_stone** — Individual black stones on the board.
3. **white_stone** — Individual white stones on the board.

### Why not detect empty intersections?

Source datasets annotate empty intersections (`empty`, `empty_edge`, `empty_corner`), but we intentionally drop these because:

- Empty intersections vastly outnumber stones, creating class imbalance.
- Detecting 361 empty intersections on a 19x19 board is wasteful when they can be inferred.
- The board bounding box + stone positions are sufficient to reconstruct the full board state.
- Fewer categories = simpler model, faster inference, smaller ONNX file.

### Partial Board Views

The model must handle photos where only part of the goban is visible (e.g., 1, 2, or 3 corners showing). This is common in real-world usage (close-up shots, angled photos). The current 3 categories already cover this:

- A partial board still has a `board` bounding box (covering the visible portion).
- Only the visible stones are annotated.
- The downstream SGF conversion must handle partial boards (e.g., infer grid from visible stones/edges).

Future data collection should include partial board photos to improve robustness.

### SGF Conversion (downstream, not in this repo)

After detection, the Kaya app will:

1. Use the board bounding box to identify the playing area.
2. Estimate the grid size (9x9, 13x13, or 19x19) from stone density and spacing.
3. Map each stone's center to the nearest grid intersection.
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

## ONNX Export

- Export via `torch.onnx.export` with dynamic axes for batch dimension.
- Target ONNX opset 16+.
- Verify with `onnxruntime` before publishing.
- Published to `kaya-go/moku-v1` on Hugging Face Hub.

## Dataset: kaya-go/moku-v1

See [dataset.md](dataset.md) for full details on sources, harmonization, and statistics.
