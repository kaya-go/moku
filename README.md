# Moku

Object detector that turns photos of a Go board (goban) into a position. Moku detects black
stones, white stones and the four board corners; the [Kaya](https://github.com/kaya-go/kaya) app
runs the exported ONNX model in the browser (ONNX Runtime Web), maps the stones onto the grid
through the corners' homography and produces an SGF.

| Resource    | Link                                                                         |
| ----------- | ---------------------------------------------------------------------------- |
| Model       | [kaya-go/moku-v3](https://huggingface.co/kaya-go/moku-v3) (`model.onnx` for Kaya) |
| Dataset     | [kaya-go/moku-v3](https://huggingface.co/datasets/kaya-go/moku-v3)           |

## Pipeline

```
Photo → detector (ONNX, 640×640) → stones + corners → homography → grid → SGF
```

| ID  | Category       | Description                                |
| --- | -------------- | ------------------------------------------ |
| 0   | `black_stone`  | Individual black stone                     |
| 1   | `white_stone`  | Individual white stone                     |
| 2   | `board_corner` | Outermost grid intersection at each corner |

The model is a fine-tuned [RT-DETR](https://arxiv.org/abs/2304.08069) (ResNet-18vd backbone,
from [`PekingU/rtdetr_r18vd`](https://huggingface.co/PekingU/rtdetr_r18vd)): no NMS, so the ONNX
graph is plain and Kaya only needs a sigmoid, a threshold and some geometry.

## Usage

```bash
pixi install

# Compare models end to end (detection metrics + positions rebuilt with Kaya's post-processing)
pixi run moku eval kaya-go/moku-v2 kaya-go/moku-v3 --split validation --split test --sweep

# Export for Kaya, check ONNX Runtime against PyTorch, measure single-thread latency
pixi run moku export kaya-go/moku-v3 --output artifacts/model.onnx

# Datasets and annotation
pixi run moku dataset stats
pixi run moku dataset audit            # annotations whose corners do not fit their stones
pixi run moku annotate prepare && pixi run moku annotate serve

pixi run test
```

Training runs on Hugging Face Jobs with the self-contained [`scripts/train.py`](scripts/train.py)
(see its docstring and [`scripts/launch_grid_r10.sh`](scripts/launch_grid_r10.sh), the recipe of
the production model).

## Documentation

- [docs/architecture.md](docs/architecture.md) — design decisions
- [docs/dataset.md](docs/dataset.md) — dataset sources, harmonization and splits
- [docs/progress.md](docs/progress.md) — results and next steps

## License

AGPL-3.0
