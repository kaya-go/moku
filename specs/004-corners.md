# 004 — Corners

Status: in progress

## Why

One wrong corner among the 4 Kaya keeps breaks the homography and the whole board. With
ground-truth corners, moku-v3's test errors drop from 50 to 6.6 per board. Diagnostic on v4
validation (a GT corner counts as found within half a grid cell):

| | moku-v2 | moku-v3 |
|---|---|---|
| any query near the true corner | 86% | 88% |
| … among Kaya's 4 kept corners | 68% | 72% |
| … among the 8 best corner candidates | 72% | 80% |
| corner-failed boards with all 4 true corners in the top 8 | 10 / 62 | 31 / 73 |

So a better selection among candidates can save some boards, but 20–30% of true corners are not
proposed well enough: that needs better data or a better model. Corner boxes are not the issue
(1–1.5× a stone). The recipe alone (spec 001) did not fix corners, and the Gemini-generated
images seem to hurt them (B1, real photos only, beats B0 on corner failures).

## Levers (no second model in Kaya)

1. **Data**: train without the generated images, add the Roboflow training photos, and correct
   misplaced corner labels automatically from the stones (the annotated stones define the grid).
2. **Selection and refinement in post-processing**: among the corner candidates, pick the quad
   that best explains the detected stones, then refine the homography on the stones. Prototype in
   `moku.board`, measure on saved predictions, port to Kaya with v4.
3. **Query budget**: checked above — true corners are usually proposed by *some* query (86–88%);
   raising the 300 queries is not the first fix.
4. **Corner head** (same model): a small head on the encoder features regresses the 4 ordered
   corners + a visibility per corner, trained on the annotated (visible) corners. Hybrid use: each
   regressed corner snaps to the nearest DETR corner candidate within a cell, else stays as is.
   Exported as an extra ONNX output (`board_corners`); Kaya reads outputs by name, so it can adopt
   it later without breaking.

A 4th "board" class (bbox of the grid) was considered: a bbox cannot give a perspective quad, it
only filters false corners outside the board, and it changes Kaya's 3-class contract. Not now.

## Acceptance criteria

- Corner label correction: counts of corrected images per source, before/after audit residuals,
  and a few rendered examples checked by eye.
- Each lever reports corner failures and perfect boards (strict) on v4 validation/test and
  Gomrade, paired against the previous best, with 90% CIs.

## Tasks

- [x] Diagnostic of corner candidates (table above).
- [x] `refine_corners` (`moku.annotations`): least-squares homography from the snapped stones,
  applied when it lowers the snap residual and moves no corner by more than 1 cell. Applied to
  every real image of moku-v4 (all splits; generated images left as is): 591 corrected
  (go_game_v10 138, go_chess 181, my_go_detection 272), median shift 0.25 cell, p90 0.41, max 0.99.
  **The v2/v3 baselines of spec 003 predate this correction and must be recomputed.**
- [ ] B2 (launched 2026-09-23, `b2-rtdetr-real-rf-s0`): RT-DETR, real photos only (v3 real +
  Roboflow train), corrected corners. Compare with B1 (`b1-rtdetr-real-s0`, no Roboflow, old
  labels) and B0 (`b0-rtdetr-s0/1`, with generated images).
- [x] Post-processing selection/refinement prototype: `moku.corner_fit`, `moku eval --corners fit`.
  Every convex quad of 4 (or 3 completed) among the 8 best deduplicated candidates is scored by the
  mean clipped distance of the detected stones to its grid (+ collisions, + a small rank prior);
  it replaces Kaya's quad when better by 0.02 cell, then a least-squares homography from the
  inlier stones refines it (≤ 1 cell per corner, kept only if the fit improves). Needs ≥ 8 stones.
- [x] Corner head (`moku.corner_head`, `--corner-head`): CenterNet-style heatmap + sub-cell
  offset on the stride-8 encoder map (conv 256→64→64→3, ~1.2 GFLOPs), penalty-reduced focal loss
  (σ = 1 cell of the heatmap) + L1 offset, weight 1. Decoded in-graph into the 8 best local maxima,
  ONNX output `corner_points` `(batch, 8, 3)` = x, y in [0, 1], score. `moku eval --corners head`
  (Kaya's selection on the head points) and `head+fit`. Checkpoints still selected on Kaya's pipeline.
- [x] H1 (`h1-rtdetr-real-rf-head-s0`): B2 + corner head, 72 epochs.
- [x] F1 (`f1-v2-frozen-head`): corner head alone on frozen moku-v2 (`--freeze-detector`), 24 epochs.
- [x] G1 (`g1-v2-ft-head`): F1 fine-tuned end to end (lr 2e-5, 20 epochs, selected on `head`).
- [x] F2 (`f2-v2-frozen-head128`): F1 with a 128-channel head, 100 epochs.
- [x] Stone threshold calibrated on validation with the head's corners (`moku calibrate --corners head`):
  0.035 → 0.025 (offset +0.35); ONNX exported with it, verified, published privately as `kaya-go/moku-v4`.
- [ ] Port to Kaya: read `corner_points`, run Kaya's corner selection on those points, switch to moku-v4.

## Results

### 2026-09-23 night: data, stone fit, corner head

Perfect boards (strict), 90% CI, paired Δ vs moku-v2 (Kaya pipeline). Corner fail = a corner more
than half a cell off. v4 labels with corrected corners. `best` = checkpoint selected on validation.

| Model | Corners | Val (129) | Test (113) | Gomrade (104) | Corner fail val / test / Gomrade |
|---|---|---|---|---|---|
| moku-v2 | kaya | 29% | 30% | 28% | 52 / 54 / 54% |
| moku-v2 | fit | 29% (+1) | 32% (+2 [0, +5]) | 33% (+5 [−5, +14]) | 44 / 43 / 39% |
| moku-v3 | kaya | 15% (−14) | 19% (−12) | 2% (−26) | 55 / 63 / 79% |
| B0 `b0-rtdetr-s0` | kaya | 15% (−14) | | | 74% (val) |
| B1 `b1-rtdetr-real-s0` | kaya | 24% (−5) | | | 66% (val) |
| B2 `b2-rtdetr-real-rf-s0` best (ep 20) | kaya | 33% (+4) | 19% (−11 [−19, −2]) | 19% (−9) | 58 / 70 / 62% |
| B2 last (ep 72) | kaya | 8% | 6% | 4% | 68 / 81 / 56% |
| H1 best (ep 12) | head | 42% (+13 [+3, +22]) | 41% (+11 [+2, +19]) | 24% (−4 [−16, +9]) | 23 / 28 / 16% |
| F1 (v2 frozen + head) | head | 34% (+5 [+2, +11]) | 34% (+4 [0, +8]) | 37% (+9 [+3, +15]) | 29 / 30 / 33% |
| F1 | head+fit | 34% (+5) | 38% (+8 [+3, +14]) | 34% (+6) | 25 / 22 / 27% |
| G1 (F1 fine-tuned) | head | 40% (+12) | 34% (+4 [−5, +14]) | 38% (+11 [+5, +17]) | 29 / 27 / 24% |
| F2 (v2 frozen + head 128) | head | 34% (+5 [+2, +11]) | 36% (+6 [+1, +12]) | 38% (+11 [+5, +18]) | 24 / 25 / 23% |
| **moku-v4** = F2, threshold 0.025, ONNX | head | 42% | 44% [35, 53] | 47% [37, 58] | 23 / 26 / 20% |

`head+fit` above is the first version (selection among the head's quads); selection is now skipped
for the head (it broke 18 correct quads on H1's test and fixed 1), which leaves head+fit ≈ head.
moku-v4's validation number is optimistic (the offset is fitted on it); test and Gomrade are held out.
F2 is chosen over H1 (better on v4 test, 41%, but 24% on Gomrade: its stones do not generalize)
and G1 (no better than F2, and it moves v2's stones): v2's stones are untouched, so no stone
regression is possible. ONNX latency (1 thread, desktop CPU) 705 ms vs 649 ms for v2 (+9%), 82.6 MB.

Where the remaining boards are lost (F2, head corners, test / Gomrade): corners ≈ 30%; the threshold
≈ 14 points (a per-image oracle threshold reaches 48% / 53%; a per-image Otsu threshold on log scores
gave +4 on v4 but −5 on Gomrade, not kept); stones missed or misclassified at any threshold ≈ 22%.

- Data lever: removing the generated images and adding Roboflow (B0 → B1 → B2) helps on
  validation, but B2's 33% is selection bias: 19% on test, and every run from the PekingU base
  peaks around epoch 12–20 then degrades (B2 last: 8%). moku-v2 stays the best stone detector.
- Stone fit (`--corners fit`): consistently fewer corner failures (−8 to −17 points) and errors,
  small perfect-board gains; on the corner head's points it sometimes hurts (H1 head+fit < head).
- Corner head: the largest lever. H1's head halves corner failures (16–28% vs 54% for v2), and does
  not degrade with training like the DETR corners. On frozen v2 (F1) the head is weaker (loss 0.9 vs
  0.18) but keeps v2's stones: +4 to +9 points everywhere.
- Stone threshold (B2, per source): 0.035 is best on every source but the peak is sharp
  (go_chess 54% at 0.035, 8% at 0.05): scores are low (TP ≈ 0.06–0.1), which is why users tune the slider.

Resume point (2026-09-23, before G1/F2) — previous resume point:

1. `moku runs list` for B0, B1, B2 (all finished by then); `moku eval` their `best/` and `last/`
   with v2/v3 on moku-v4 validation/test (new labels) and Gomrade; `moku calibrate` the winner.
2. If real-only + Roboflow + corrected corners wins (expected from B1): train the corner head
   (lever 4) and prototype the post-processing selection (lever 2) on that data.
3. DEIMv2-S (best calibration, TP ≈ 0.7) and D-FINE-S deserve a rerun on the clean data only if
   the budget allows (~$2.5 per A100 run; ~$30 of HF credit left, hard cap).
