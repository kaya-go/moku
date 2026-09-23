# 003 — External evaluation data

Status: draft — plan in [#1](https://github.com/kaya-go/moku/issues/1) (section D)

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

## Acceptance criteria

- `moku eval --dataset <external>` reports board metrics on the external set for v2, v3 and new models.
- The external set is private (license) and documented in `docs/dataset.md`.

## Tasks

- [ ] Get Gomrade (needs a Kaggle account/token) and Roboflow `my-go-detection` (needs a Roboflow export).
- [ ] Converter to the moku format (image, positions, corners), sampling a few frames per game.
- [ ] Evaluate v2 / v3 / new models on it.

## Results

_Pending._
