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
  `moku.evaluation` with `moku eval`. `scripts/train.py` is a thin entry point.
- **Jobs run the locked pixi environment**: `moku train launch` stages `pixi.toml`, `pixi.lock`,
  `src/` and `scripts/`, and runs `pixi run --frozen -e cuda python scripts/train.py` in the pixi
  Docker image. Local and remote runs share one dependency set (no PEP 723 header to keep in sync).
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

Partial (2026-09-23), validation v4, EMA weights:

- Throughput: ~60 img/s (RT-DETR) and ~52 img/s (D-FINE-S) on an A100, CPU-bound (GPU busy 43%
  of the step; ~8.7k small kernels per step). The transformers matcher was quadratic in the batch
  size; replaced by a per-image matcher (`moku.training.matcher`).
- B0 (RT-DETR, v3 train incl. generated images): corners learned from epoch ~10 but validation
  got worse after epoch ~25 (errors 61 → 70+, corner failures 70% → 80–90%) while the training
  loss kept falling. Best val (old perfect definition) 20–23% at epochs 17–25.
- The recipe did **not** fix calibration: true-positive scores stay ~0.05–0.08, even on training
  images (max score per image ~0.3). Likely one-to-one matching ambiguity on dense identical
  stones. DEIMv2's MAL loss gives TP ≈ 0.7 (run C2, stopped for slow corners).
- B1 (RT-DETR, real photos only): better corners than B0 (60–64% failures vs 70–90%).
- A per-model stone threshold baked into the ONNX as a logit offset (`moku calibrate`) gives
  +3 to +8 points of perfect boards on v2/v3 for free.
