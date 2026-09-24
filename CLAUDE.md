# moku — Agent Instructions

## Role & Objective

You are an expert AI software engineer specializing in PyTorch, ONNX, and computer vision.
Moku is the object detector behind board recognition in [Kaya](https://github.com/kaya-go/kaya):
a photo of a goban goes in, a position (SGF) comes out. The user no longer runs code
themselves: every workflow must be runnable from the `moku` CLI or a script, with results
printed as tables or saved as files (PNG, JSON).

## Project Context

- **Target**: ONNX Runtime Web (single-threaded WASM) inside Kaya — web, desktop (Tauri) and mobile.
  Board recognition must just work with no configuration on the user's side.
- **HF Organization**: `kaya-go` on Hugging Face Hub.
- **Datasets**: `kaya-go/moku-v1`, `kaya-go/moku-v2`, `kaya-go/moku-v3` on Hugging Face Hub.
- **Model in production**: `kaya-go/moku-v3` (fine-tuned RT-DETR r18vd, W&B run
  `r10_os3_lr3e-4_cosmin100`). Kaya downloads `kaya-go/moku-v3/resolve/main/model.onnx`.
- **Kaya repo**: usually checked out at `../kaya`; board recognition lives in
  `packages/board-recognition/src/moku-*.ts`.

## Tech Stack & Environment

- **Package Manager**: `pixi` (strictly enforced; do NOT use pip/conda directly).
- **Training**: `scripts/train.py` is a self-contained PEP 723 script run on HF Jobs
  (`hf jobs uv run ...`); it cannot import `moku`. Keep its dependency header in sync with `pixi.toml`.
- **Tracking**: W&B project `hadim/moku` (best checkpoints are saved as W&B artifacts).
- **Secrets**: `.env` (`WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT`, `GEMINI_API_KEY`), loaded by the CLI.

## Commands

```bash
pixi run moku eval kaya-go/moku-v2 kaya-go/moku-v3 -s validation -s test --sweep  # compare models
pixi run moku eval wandb:model-<run>:latest --figures reports/figs                   # W&B checkpoint + worst boards
pixi run moku export kaya-go/moku-v3 -o artifacts/model.onnx                         # ONNX + verify + latency
pixi run moku publish wandb:model-<run> --repo kaya-go/moku-vN --onnx artifacts/model.onnx
pixi run moku dataset stats | audit | build-v3
pixi run moku annotate prepare | serve                                               # tools/annotator workflow
pixi run moku generate --n 500                                                       # Gemini style transfer
pixi run test && pixi run lint
```

## Docs

- `docs/architecture.md`: architecture decisions and design rationale.
- `docs/dataset.md`: dataset sources, harmonization rules, and raw data location.
- `docs/progress.md`: current progress, results and next steps.

## Detection Categories

The harmonized dataset uses 3 categories:

| ID  | Name           | Description                                    |
| --- | -------------- | ---------------------------------------------- |
| 0   | `black_stone`  | Individual black stone                         |
| 1   | `white_stone`  | Individual white stone                         |
| 2   | `board_corner` | Board corner point (small bbox at each corner) |

Board corners are the outermost grid intersections; they give a homography onto the grid.
Empty intersections are never detected — they are inferred from geometry.

**Partial board views**: the model must handle photos where only part of the goban is visible (1–3 corners).

## Kaya Contract (do not break silently)

- **ONNX I/O**: input `pixel_values` `(batch, 3, 640, 640)` float32 RGB in [0, 1] (squashed resize,
  no mean/std normalization); outputs `logits` `(batch, 300, 3)` and `pred_boxes` `(batch, 300, 4)`
  normalized `cxcywh`. Kaya hardcodes 300 queries and 3 classes.
- **Post-processing** (in Kaya, ported to `src/moku/board.py`): sigmoid + argmax per query;
  stones kept above `0.035`; corners above `0.005`, deduplicated within 5% of the diagonal,
  top 4 (2–3 corners are completed geometrically); homography; snap to the nearest intersection.
- Kaya's build rejects a downloaded model smaller than 50 MB (`scripts/copy-assets.ts`) — a smaller
  model needs that check updated in Kaya.

## Evaluation Principle

Detection metrics (mAP@50, stone cdAP, corner R@4) are diagnostics. Models are compared on
**board metrics** from `moku eval`: the Kaya pipeline reconstructs the position, which is compared
with the position read from the annotations (perfect boards, errors per board, corner failures),
with bootstrap CIs over photos. Validation/test are ~50 images from ~30 distinct photos: report
intervals, never a single number.

## Model Choice: RT-DETR r18vd

- **Why**: Transformer-based detector, no NMS needed (simpler ONNX export), small ResNet-18 backbone suitable for browser inference, available in HF `transformers`.
- **Base model**: `PekingU/rtdetr_r18vd`
- **Training**: Fine-tune with HF `Trainer` API (`scripts/train.py`).
- **Export**: `torch.onnx.export` with dynamic axes for batch dimension (`moku export`).

## Rules & Guidelines

- **Language**: All code, comments, documentation, commit messages, variable names, and any other text in this repository MUST be written in English. No exceptions.
- **Dependency Management**: Always use `pixi add <package>` to install dependencies.
- **Module size**: No Python file in `src/moku/` should exceed 600 lines. If a module grows beyond this limit, refactor it or split it into submodules.
- **No notebooks**: Reusable logic goes in `src/moku/`, entry points in `src/moku/cli.py`. Figures are saved to files, never shown interactively.
- **Paths**: Use relative paths from project root.
- **No vendor libraries**: Only standard/HF ecosystem libraries (transformers, datasets, torch, etc.). No ultralytics, roboflow SDK, etc.
- **Documentation**: Keep `CLAUDE.md` and `docs/` always up to date with current state, decisions, and progress.
- **Commit Messages**: Follow [Conventional Commits](https://www.conventionalcommits.org/) format:
  - `feat:` for new features
  - `fix:` for bug fixes
  - `docs:` for documentation changes
  - `refactor:` for code refactoring
  - `test:` for test additions/changes
  - `chore:` for maintenance tasks
  - Include scope when applicable: `feat(dataset): add synthetic board generator`
  - Use imperative mood: "add" not "added" or "adds"
