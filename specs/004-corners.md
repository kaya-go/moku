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
- [ ] Post-processing selection/refinement prototype.
- [ ] Corner head.

## Results

_Pending._ Resume point (2026-09-23, paused for budget):

1. `moku runs list` for B0, B1, B2 (all finished by then); `moku eval` their `best/` and `last/`
   with v2/v3 on moku-v4 validation/test (new labels) and Gomrade; `moku calibrate` the winner.
2. If real-only + Roboflow + corrected corners wins (expected from B1): train the corner head
   (lever 4) and prototype the post-processing selection (lever 2) on that data.
3. DEIMv2-S (best calibration, TP ≈ 0.7) and D-FINE-S deserve a rerun on the clean data only if
   the budget allows (~$2.5 per A100 run; ~$30 of HF credit left, hard cap).
