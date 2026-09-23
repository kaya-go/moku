# 001 — Training loop and run tracking

Status: in progress — plan in [#1](https://github.com/kaya-go/moku/issues/1) (section B)

## Why

moku-v3 (W&B run `r10_os3_lr3e-4_cosmin100`) beats v2 on mAP@50 but rebuilds fewer boards
(validation perfect boards 17% vs 44%). The loop that trained it had real defects:

- augmentation silently misconfigured under albumentations 2.x (σ≈51–112 noise on 30% of images);
- `dataloader_num_workers=0`: augmentation in the main process, ~17 img/s on an A10G;
- no weight EMA, and the same lr (3e-4) on the backbone and the head;
- checkpoint chosen by the max of a noisy validation mAP@50, a metric that does not track boards;
- true stones and corners score ~0.07, so Kaya's thresholds sit on a knife edge.

W&B tracking is also gone (expired key): runs must be followed without it.

## Hypothesis / goal

The reference RT-DETR/D-FINE recipe alone (same model, same data) fixes calibration
(true-positive scores well above Kaya's thresholds) and most corner failures.

## Acceptance criteria

- `moku train launch` starts a run on HF Jobs and prints its URL; the run writes config, metrics,
  log and checkpoints to the bucket `hadim/moku-runs`; `moku runs list|show|pull` read them back.
- Throughput ≥ 3× r10 (> 50 img/s on `a10g-large`).
- **B0** (RT-DETR r18vd, fixed recipe, 2 seeds): validation perfect boards ≥ moku-v2 (44%), fewer
  corner failures than v2 and v3, median TP scores > 0.3. Compared with paired CIs vs moku-v3 on
  validation + test (`moku eval`).

## Design

- **Training code in the package** (`src/moku/training/`), so it is linted, testable and shares
  `moku.evaluation` with `moku eval`. `scripts/train.py` is a thin PEP 723 entry point; on HF Jobs
  `src/` is mounted at `/moku-src` (`hf jobs uv run -v ./src:/moku-src`).
- **Plain PyTorch loop** instead of HF `Trainer`: param groups, EMA, per-iteration schedule, the
  augmentation switch and board-metric selection are each a few lines here and fights with `Trainer`.
- **Recipe** (RT-DETR / D-FINE reference configs, scaled to batch 16):
  AdamW, lr 1e-4, backbone ×0.1 (RT-DETR) or ×0.5 (D-FINE), wd 1e-4 but none on 1-D tensors,
  grad clip 0.1, bf16 autocast; 500-iteration warm-up, flat to 50%, cosine to 5%; EMA (decay 0.999,
  warm-up τ = 500 iterations), restarted when strong augmentation stops; last 8 of 72 epochs with
  flips only; data v3 with real images ×3.
- **Selection**: EMA weights evaluated every epoch on validation with `moku.evaluation.evaluate`
  (the exact `moku eval` code); `best/` = most perfect boards, then fewest errors. `last/` is kept
  too. The test split is never looked at during training.
- **Tracking without W&B**: the run directory is a mounted HF bucket; `metrics.jsonl` gets one
  record per 25 iterations and per epoch. The HF Jobs page gives the live log.

## Tasks

- [x] `moku.training.data`: fixed augmentation, Kaya-like squash resize, oversampling, previews.
- [x] `moku.training.engine`: loop, param groups, schedule, EMA, board-metric selection, run dir.
- [x] TP-score calibration metric in `moku.evaluation` (also shown by `moku eval`).
- [x] `moku train launch|preview-aug`, `moku runs list|show|pull`; `hf://buckets/...` model sources.
- [x] W&B removed (dependency, `wandb:` sources, `scripts/analyze_runs.py`).
- [x] Local smoke test (MPS, 2 epochs).
- [ ] GPU smoke test on HF Jobs (throughput, bucket writes).
- [ ] B0 × 2 seeds.
- [ ] Evaluate B0 vs moku-v2 / moku-v3 on validation + test; record below.

## Results

_Pending._
