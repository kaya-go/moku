# Integrating moku-v4 in Kaya

Everything Kaya needs to switch from `kaya-go/moku-v3` to `kaya-go/moku-v4`. The work is in
`packages/board-recognition/src/moku-detector.ts` and `moku-postprocess.ts`. Background and
numbers are in [`specs/004-corners.md`](../specs/004-corners.md).

## What changes, in short

1. **Model URL**: `https://huggingface.co/kaya-go/moku-v4/resolve/main/model.onnx` (public).
2. **New ONNX output `corner_points`**: board corners come from a dedicated corner head. They are
   no longer taken from the DETR queries of class `board_corner`.
3. **Same stone logic, same default threshold (0.035)**: the calibration is baked into `logits`.

Nothing else changes: same input and preprocessing, same stone decoding, homography and snapping,
and the same corner selection code, now fed with different candidates.

## Why

On photos never used for any choice, the share of *perfect* boards (every intersection right and
every corner within half a cell) goes from 19% to 44% on the moku-v4 test set, and from 2% to 47%
on Gomrade (real game photos). Boards with a corner off by more than half a cell (a manual fix in
Kaya) drop from 63–79% to 20–26%. See the table at the end.

## Model

| | moku-v3 | moku-v4 |
|---|---|---|
| Architecture | RT-DETR r18vd | moku-v2 (RT-DETR r18vd), unchanged and frozen, + a corner head (443k params, 128 channels) |
| `model.onnx` size | ~80 MB | 82.6 MB (Kaya's 50 MB minimum in `scripts/copy-assets.ts` still holds) |
| Latency (1 thread, desktop CPU, ORT) | ~650 ms (measured on moku-v2, same detector) | ~660–700 ms (+2–9%) |

## ONNX contract

| Name | Shape | Type | Meaning |
|---|---|---|---|
| `pixel_values` (input) | `(batch, 3, 640, 640)` | float32 | Unchanged: RGB in [0, 1], squashed resize, no mean/std. |
| `logits` | `(batch, 300, 3)` | float32 | Unchanged layout. The stone calibration is baked in (+0.35 added to every class logit), so keep the 0.035 threshold. |
| `pred_boxes` | `(batch, 300, 4)` | float32 | Unchanged: normalized `(cx, cy, w, h)`. |
| `corner_points` | `(batch, 8, 3)` | float32 | **New.** The 8 best board-corner peaks, sorted by score (descending): `(x, y, score)`. |

`corner_points` details:

- `x`, `y` are normalized to [0, 1] in the 640×640 input, which is also the normalized original
  image because the resize is squashed: pixel = `x * origWidth`, `y * origHeight` (same as the
  `cx`, `cy` of `pred_boxes`).
- `score` is already a probability (sigmoid applied in the graph). Do **not** apply a sigmoid.
- The points are the local maxima (3×3) of an 80×80 heatmap with sub-cell offsets. There are
  always 8 of them, and the tail is low-score noise. They are class-agnostic: a corner is a
  corner, and `orderCorners` gives the TL/TR/BR/BL order as today.
- Only `batch_size` is a symbolic dim; the others are static (300, 3, 4, 8). The
  `freeDimensionOverrides` fallback in `moku-detector.ts` only needs `batch_size: 1`. The keys
  `Gatherlogits_dim_1` / `Gatherpred_boxes_dim_*` were moku-v3's names and are harmless but unused.

## Post-processing change

In `postprocess` (`moku-postprocess.ts`), the loop over the 300 queries fills `cornerCandidates`
with the queries whose best class is `board_corner` and whose score is ≥ 0.005 (around L170–205).
With moku-v4, **fill `cornerCandidates` from `corner_points` instead**, and ignore corner-class
queries. Everything after that stays exactly as it is: sort by score, drop a candidate within 5% of
the image diagonal of a better one, fall back to the inset image bounds below 2 candidates,
complete 2 or 3 corners geometrically, keep the top 4, check for degenerate or collapsed quads.

```ts
// moku-detector.ts, after session.run
const cornerPoints = 'corner_points' in results
  ? (results.corner_points.data as Float32Array) // (1, 8, 3)
  : null; // moku-v3 and older: no corner head
this.cachedCornerPoints = cornerPoints; // needed by refilter()'s full path (board size change)
const out = postprocess(logits, predBoxes, img, options.boardSize, threshold, outputSize, cornerPoints);

// moku-postprocess.ts, in postprocess(...)
const CORNER_MIN_THRESHOLD = 0.005; // unchanged
if (cornerPoints) {
  for (let k = 0; k < cornerPoints.length / 3; k++) {
    const score = cornerPoints[3 * k + 2];
    if (score < CORNER_MIN_THRESHOLD) continue;
    cornerCandidates.push({
      cx: cornerPoints[3 * k] * origImg.width,
      cy: cornerPoints[3 * k + 1] * origImg.height,
      score,
      classId: CLASS_BOARD_CORNER, // plus whatever else MokuRawDetection needs (w/h: use 0 or a small box)
    });
  }
}
// in the query loop: when cornerPoints is set, skip queries whose best class is board_corner
// (they are neither stones nor candidates). Stones: unchanged.
```

Keep the old path when `corner_points` is absent: a custom `modelUrl` or a cached moku-v3 model
must keep working. Read outputs by name (`results.corner_points`), never by index.

The refilter fast path (threshold change) reuses the cached corners and needs nothing new. The
full path (board size change) calls `postprocess` again and must pass the cached corner points.

### Threshold

The default threshold stays **0.035**. moku-v4's logits already carry the calibration fitted on
the validation set (a raw 0.025 threshold), so 0.035 is the best single value for this model. The
slider can stay, but users should rarely need it. On a model without calibration (v3), 0.035
still means what it meant.

### Reference implementation

The Python port of Kaya's pipeline is the reference, and it is what the numbers below come from:

- `src/moku/board.py`: `reconstruct_board(..., corner_method="head", corner_points=...)` builds
  the candidates from `corner_points` and then runs `select_corners`, the port of Kaya's selection.
  Stones go through `decode_queries` and `snap_to_grid`, both unchanged.
- `src/moku/corner_head.py`: `CornerHead.decode` is the in-graph decoding (for reference only;
  Kaya reads its output).

## Verifying the port

1. **Golden fixtures** (raw outputs → expected corners and grid):

   ```bash
   hf download kaya-go/moku-v4 model.onnx --local-dir artifacts/moku-v4
   pixi run moku predict photo1.jpg photo2.jpg --model artifacts/moku-v4/model.onnx \
       --board-size 19 --json fixtures/moku-v4.json
   ```

   Each entry has the image size, `board_size`, `threshold`, the raw `outputs` (`logits`,
   `pred_boxes`, `corner_points`), the expected `corners` (TL, TR, BR, BL in pixels) and the
   expected `grid` (`.` empty, `X` black, `O` white). A Kaya unit test can feed `outputs` to
   `postprocess` and expect the same corners (±1 px) and exactly the same grid. That tests the
   post-processing without running the model.
2. **End to end**: running the ONNX in Kaya on the same photos should give the same corners
   (`moku predict` runs the ONNX with `kaya_preprocess` in `src/moku/inference.py`, the Python
   port of Kaya's `preprocess`).
3. **Metrics** (moku side, already done): `pixi run moku eval artifacts/moku-v4/model.onnx
   --dataset kaya-go/moku-v4 -s test --corners head` gives 44% perfect boards [35–53].

## Expected results

Perfect boards, 90% CIs over photos (games for Gomrade). The ONNX was run with Kaya's
preprocessing and Kaya's pipeline, with moku-v4 corners taken from `corner_points`.

| | v4 test (113 boards) | Gomrade (104 boards) | Corner > ½ cell off (test / Gomrade) |
|---|---|---|---|
| moku-v3 (in Kaya today) | 19% [12–26] | 2% [0–5] | 63% / 79% |
| moku-v2 | 30% [22–37] | 28% [18–38] | 54% / 54% |
| **moku-v4** | **44% [35–53]** | **47% [37–58]** | **26% / 20%** |

## Known limits and follow-ups

- **Partial boards** (1–3 visible corners): the corner head learns every annotated (visible)
  corner, but the validation and test sets have almost no partial boards, so their behavior is
  not measured. The same Kaya
  selection runs on the head's points, and true and false peaks overlap in score (true median
  0.69; 95% of false peaks below 0.38), so no score floor separates them cleanly.
- **Stones** are moku-v2's, unchanged. The remaining failures are split between corners (~30% of
  boards), the single threshold (a per-image oracle threshold would reach ~50%) and stones missed
  at any threshold (~22%).
- **Not ported on purpose**: the corner selection by stone fit (`moku eval --corners fit`,
  `src/moku/corner_fit.py`). It helps DETR corners but brings nothing on top of the corner head.
