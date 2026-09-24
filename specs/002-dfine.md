# 002 — D-FINE

Status: in progress — plan in [#1](https://github.com/kaya-go/moku/issues/1) (section C)

## Why

Corner localization drives most board failures, and Kaya runs the model on one WASM thread, so
download size and latency matter. RT-DETR r18vd (today): 20M params, 81 MB ONNX, 654 ms
(ORT CPU, 1 thread, M3 Max).

D-FINE-S (`ustc-community/dfine-small-obj2coco`): 10M params, COCO AP ~50.7, Objects365 → COCO
pretraining, 41.5 MB ONNX, 318 ms, same processor and I/O contract (640, [0, 1], 300 queries,
`logits` / `pred_boxes`). Its fine-grained distribution refinement targets box-edge precision.
D-FINE-N (`ustc-community/dfine-nano-coco`): 4M params, 15.5 MB, 105 ms.

## Hypothesis / goal

On top of the fixed recipe (spec 001), D-FINE-S rebuilds at least as many boards as RT-DETR r18vd
at half the size and latency.

## Acceptance criteria

- **C0** (D-FINE-S, 2 seeds): paired Δ perfect boards vs B0 ≥ 0 on validation + test, no increase
  in corner failures.
- `moku export` of the best checkpoint: ONNX verified against PyTorch, Kaya contract unchanged.
- **C1** (D-FINE-N, 1 seed): measured, to decide whether a 15 MB model is good enough for mobile.

## Design

- Same loop and data as B0; `--model dfine-s` / `--model dfine-n`.
- Reference fine-tuning config (`dfine_hgnetv2_s_obj2coco.yml`, batch 32: lr 2.5e-4, backbone
  ×0.5, wd 1.25e-4) scaled linearly to batch 16: **lr 1.25e-4, backbone ×0.5**, wd 1e-4.
- Shipping a 41 MB model needs Kaya's `MIN_MOKU_MODEL_BYTES` (50 MB) lowered, the model URL moved
  to `moku-v4`, and the model's own stone threshold (see #1, section E).

## Tasks

- [x] Forward/backward with labels works in `transformers` 5.16 (`DFineForObjectDetection`, 3 classes).
- [ ] C0 × 2 seeds, C1 × 1.
- [ ] Evaluate vs B0 and moku-v3; export and benchmark the winner.

## Results

Inconclusive (2026-09-23). The first runs had a broken EMA (D-FINE ties its class/box heads;
fixed in `ModelEMA`). The fixed run (`c0-dfine-s-s0-v3`, with the generated images) learned
corners only from epoch ~17 and was stopped at epoch 23 (errors 73, corner failures 79%) to save
budget. DEIMv2-S (`c2-deimv2-s-s0`) had excellent calibration (TP ≈ 0.7) but no corners by
epoch 17. Both should be retried on the clean data (spec 004) if the budget allows.
