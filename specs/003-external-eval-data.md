# 003 — External evaluation data

Status: in progress — plan in [#1](https://github.com/kaya-go/moku/issues/1) (section D)

## Why

Validation and test hold ~50 images each but only ~26 and ~30 distinct photos, all ≤640 px, so a
perfect-board rate carries a ±12–15 point CI and does not look like Kaya's input (phone photos up
to 1600 px). No new photos will be taken by hand, so any extension must come from public data.

## Goal

A second, independent evaluation set of real board photos with known positions, on which moku-v2,
moku-v3 and every new model are evaluated the same way (none of them has seen it, so comparisons
stay fair). The current test split stays as the regression check.

## Candidates (survey of 2026-09-23)

Nothing public offers a large, permissively licensed set of phone photos with positions.

| Source | Content | Use | License |
|---|---|---|---|
| [Gomrade](https://www.kaggle.com/datasets/davids1992/gomrade-dataset-go-baduk-images-with-labels) (Kaggle) | ~2,180 frames from ~60 games, 1000–1900 px, 19×19 positions as text + 4 corner clicks per folder | **eval** (sample 1–3 frames per game, bootstrap over games) | CC BY-NC-ND 4.0: private evaluation only, never redistributed nor trained on |
| [Tengen-Go test images](https://github.com/OliverBenz/Tengen-Go) | ~80 images, 9×9 and 13×13, angled, hard lighting | eval, if its labels hold positions | AGPL-3.0 |
| Roboflow `serjs-workspace/my-go-detection` | 367 photos, same 3 classes as moku | training (check the corner definition) | CC BY 4.0 |
| [goban_data_set](https://github.com/irglbriz/goban_data_set) | 494 phone photos, 1000², grid-corner keypoints, no stones | corner-only eval | MIT |

Stone-only Roboflow sets are left out of training: their unlabelled corners would be taught as
background.

## Decision

- **Gomrade → evaluation only**, private dataset `kaya-go/moku-gomrade` (split `test`): the 40 games
  of real boards (6 folders of rendered lesson diagrams dropped), 3 frames per game spread by stone
  count (empty boards skipped) = 104 boards. Boxes are synthesized from the position and the clicked
  grid corners. Bootstrap clusters = games (`source_dataset = gomrade/<game>`).
- **Roboflow `my-go-detection` → mix**, in `kaya-go/moku-v4` (private for now): moku-v3 unchanged
  plus the 367 Roboflow images split by photo (same file name or same position) 50/25/25 into
  train / validation / test (226 / 77 / 64 images). Its corners are grid corners (median stone snap
  residual 0.16 cell; 24 images flagged by `moku dataset audit`, like v3's rate). Training copies are
  downscaled to 1280 px; validation/test keep the phone resolution (up to 4096 px), as Kaya gets it.
  It brings the first 9×9 (33) and 13×13 (29) boards.
- The first comparison runs (specs 001/002) train on v3's train split (`--train-exclude
  my_go_detection`) and select checkpoints on the larger v4 validation, so the recipe and the
  architecture are measured without a data change; the Roboflow training images are a later run.

## Acceptance criteria

- `moku eval --dataset <external>` reports board metrics on the external set for v2, v3 and new models.
- The external set is private (license) and documented in `docs/dataset.md`.

## Tasks

- [x] Get Gomrade and the Roboflow `my-go-detection` COCO export.
- [x] `moku.external` converters, `moku dataset build-gomrade | build-v4`; both datasets pushed (private).
- [x] Evaluate v2 / v3 on Gomrade.
- [ ] Evaluate v2 / v3 on the v4 validation/test Roboflow photos; new models on everything.
- [ ] Dataset cards (license, attribution); decide whether moku-v4 goes public.

## Results

**Gomrade (104 boards, 40 games), stone threshold 0.035** — v3's regression is confirmed on
independent real photos:

| Model | Perfect boards | Errors / board | Corner fail | stone cdAP |
|---|---|---|---|---|
| moku-v2 | 29% | 62.4 | 54% | 0.823 |
| moku-v3 | 2% | 83.9 | 79% | 0.761 |

Paired Δ v3 − v2: −27 points of perfect boards (CI [−35, −19]; CIs here were computed over
positions, before the per-game clustering fix, so they are too narrow). mAP@50 is meaningless on
Gomrade (synthesized box sizes); use cdAP and board metrics.
