# AI Agent Instructions

## Role & Objective

You are an expert AI software engineer specializing in PyTorch, ONNX, and computer vision.
Your goal is to assist in building an object detection model that converts goban (Go board) photos to SGF files for use in the Kaya project.

## Project Context

- **Repository**: `moku`
- **Purpose**: Object detection model for goban image recognition and SGF conversion.
- **Target**: WebAssembly (WASM) via ONNX Runtime in the Kaya web app.
- **HF Organization**: `kaya-go` on Hugging Face Hub.
- **Dataset**: `kaya-go/moku-v1`, `kaya-go/moku-v2`, `kaya-go/moku-v3` on Hugging Face Hub.
- **Model**: `kaya-go/moku-v2` (fine-tuned RT-DETR) on Hugging Face Hub.

## Tech Stack & Environment

- **Package Manager**: `pixi` (Strictly enforced. Do NOT use pip/conda directly).
- **Languages**: Python.
- **Key Libraries**:
  - `pytorch` (Model training & handling)
  - `transformers` (Model loading, training, image processing)
  - `datasets` (HF dataset creation, loading, pushing)
  - `onnx`, `onnxruntime` (Export & verification)
  - `httpx` (Data fetching)
  - `huggingface_hub` (HF Hub interactions)
- **Linting**: `ruff`

## Project Architecture

```
moku/
├── AGENTS.md                  # This file - project instructions for AI agents
├── docs/
│   ├── architecture.md        # Architecture decisions and design
│   ├── dataset.md             # Dataset details, sources, harmonization
│   └── progress.md            # Current progress and next steps
├── notebooks/
│   ├── 01_Build_Dataset.ipynb # Dataset v1 creation and upload to HF
│   ├── 02_Build_Dataset_v2.ipynb # Dataset v2: corrected real + synthetic
│   ├── 03_Annotate.ipynb      # Corner annotation workflow
│   ├── 04_Synthetic_Preview.ipynb # Synthetic data preview
│   ├── 05_Generate_Images.ipynb # Gemini image generation
│   ├── 06_Annotate_Generated.ipynb # Pseudo-label + correct generated images
│   ├── 07_Build_Dataset_v3.ipynb # Dataset v3: v2 + generated images
│   ├── 21_Evaluate.ipynb      # Model evaluation (mAP, center-distance)
│   ├── 30_Publish_Model.ipynb # Select W&B artifact → push to HF Hub
│   └── 40_Export_ONNX.ipynb   # ONNX export for browser inference
├── scripts/
│   ├── train.py               # Training script for HF Jobs (two-stage, W&B)
│   ├── analyze_runs.py        # Reusable W&B run analysis (EMA, plateau)
│   ├── launch_grid_r4.sh      # HP grid search launcher (round 4)
│   ├── launch_grid_r5.sh      # HP grid search launcher (round 5)
│   └── launch_grid_r6.sh      # HP grid search launcher (round 6)
├── src/moku/
│   ├── __init__.py
│   ├── annotator.py           # Annotator server utilities
│   ├── cli.py                 # CLI entry point
│   ├── dataset.py             # Dataset loading, harmonization, v3 build utilities
│   ├── evaluation.py          # Evaluation utilities (mAP, center-distance, sweep)
│   ├── generate.py            # Gemini image generation for goban photos
│   ├── grid.py                # HP grid search helpers
│   ├── model.py               # Model loading, eval transforms, collation utilities
│   ├── runs.py                # W&B run fetching, artifact management
│   ├── synthetic.py           # Synthetic goban image generator
│   └── viz/                   # Visualization utilities
├── pixi.toml
└── pyproject.toml
```

## Pipeline Overview

1. **Dataset** (`01–07` notebooks): Harmonize raw COCO datasets, generate synthetic/Gemini images, annotate, and build HF datasets (`kaya-go/moku-v1`, `v2`, `v3`).
2. **Train** (`scripts/train.py`): Fine-tune RT-DETR r18vd via HF Jobs. Two-stage: synthetic pre-train → real fine-tune. Best weights saved as W&B artifacts.
3. **Analyze** (`scripts/analyze_runs.py`): Fetch W&B run metrics, compare key metrics (mAP@50, corner_R, stone_F1), plateau analysis.
4. **Evaluate** (`21_Evaluate.ipynb`): Validate model with mAP and center-distance metrics on test set. Supports loading from HF Hub branches or W&B artifacts.
5. **Publish** (`30_Publish_Model.ipynb`): Select best W&B artifact and push to HF Hub as official model.
6. **Export** (`40_Export_ONNX.ipynb`): Convert to ONNX with dynamic axes for batch size. Upload to HF Hub.

## Detection Categories

The harmonized dataset uses 3 categories:

| ID  | Name           | Description                                    |
| --- | -------------- | ---------------------------------------------- |
| 0   | `black_stone`  | Individual black stone                         |
| 1   | `white_stone`  | Individual white stone                         |
| 2   | `board_corner` | Board corner point (small bbox at each corner) |

Board corners enable perspective-corrected grid inference via homography.
For go_game_v10, real `board_corner` annotations are used directly.
For go_chess, synthetic 20×20 corner bboxes are generated from board segmentation polygons.

Categories like `empty`, `empty_edge`, `empty_corner`, `board` (full bbox) from source datasets are **dropped** — empty intersections are inferred from board geometry and stone positions during SGF conversion.

**Partial board views**: The model must handle photos where only part of the goban is visible (1–3 corners). Future datasets should include partial board photos.

## Raw Data Sources

Raw datasets are stored at `/Users/hadim/Data/moku/raw/` (not committed to repo):

| Dataset               | Dir Name                         | Images | Kept Categories                         | Notes                                |
| --------------------- | -------------------------------- | ------ | --------------------------------------- | ------------------------------------ |
| Go Game detection v10 | `Go Game detection.v10i.coco`    | 256    | board, black_stone, white_stone         | Primary dataset. Superset of v1.     |
| Go game detection v1  | `Go game detection.v1i.coco`     | 243    | —                                       | **Skipped**: fully contained in v10. |
| go-chess 2 v3         | `go-chess 2.v3-go-chess.v1.coco` | 236    | goboard→board, black_stone, white_stone | Different source images.             |

## Model Choice: RT-DETR r18vd

- **Why**: Transformer-based detector, no NMS needed (simpler ONNX export), small ResNet-18 backbone suitable for browser inference, available in HF `transformers`.
- **Base model**: `PekingU/rtdetr_r18vd`
- **Training**: Fine-tune with HF `Trainer` API.
- **Export**: `torch.onnx.export` with dynamic axes for batch dimension.

## Rules & Guidelines

- **Language**: All code, comments, documentation, commit messages, variable names, and any other text in this repository MUST be written in English. No exceptions.
- **Dependency Management**: Always use `pixi add <package>` to install dependencies.
- **Code Style**: Adhere to `ruff` defaults.
- **Module size**: No Python file in `src/moku/` should exceed 600 lines. If a module grows beyond this limit, refactor it or split it into submodules.
- **Notebooks**: Maintain clean cells; use Markdown for documentation. Keep notebooks executable on an M3 MacBook. For long training jobs, use HF Jobs. Prefer `pandas.DataFrame` + `display()` over loops of `print()` for summarizing data.
- **Library code**: Reusable or long functions go in `src/moku/`. Notebooks import from `moku`.
- **Paths**: Use relative paths from project root.
- **No vendor libraries**: Only standard/HF ecosystem libraries (transformers, datasets, torch, etc.). No ultralytics, roboflow SDK, etc.
- **Documentation**: Keep `AGENTS.md` and `docs/` always up to date with current state, decisions, and progress.
- **Commit Messages**: Follow [Conventional Commits](https://www.conventionalcommits.org/) format:
  - `feat:` for new features
  - `fix:` for bug fixes
  - `docs:` for documentation changes
  - `refactor:` for code refactoring
  - `test:` for test additions/changes
  - `chore:` for maintenance tasks
  - Include scope when applicable: `feat(dataset): add synthetic board generator`
  - Use imperative mood: "add" not "added" or "adds"
