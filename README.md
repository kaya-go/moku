# Moku

Object detection model for converting goban (Go board) photos and screenshots to SGF files. Moku detects stones and board geometry in goban images, enabling automatic game transcription. The exported ONNX model powers the [Kaya](https://github.com/kaya-go/kaya) web app, running directly in the browser via ONNX Runtime WebAssembly.

| Resource      | Link                                                                       |
| ------------- | -------------------------------------------------------------------------- |
| Trained Model | [kaya-go/moku-v1](https://huggingface.co/kaya-go/moku-v1)                  |
| Dataset       | [kaya-go/moku-v1](https://huggingface.co/datasets/kaya-go/moku-v1)         |
| ONNX Model    | [kaya-go/moku-v1](https://huggingface.co/kaya-go/moku-v1) (`model.onnx`) |

## Pipeline

```bash
Photo → Object Detection (ONNX) → Stone Positions → Grid Mapping → SGF
```

## Dataset

The [`kaya-go/moku-v1`](https://huggingface.co/datasets/kaya-go/moku-v1) dataset is a harmonized object detection dataset combining two COCO-format sources from Roboflow (~492 images total). It detects 3 categories:

| ID  | Category      | Description                |
| --- | ------------- | -------------------------- |
| 0   | `board`       | Full Go board bounding box |
| 1   | `black_stone` | Individual black stone     |
| 2   | `white_stone` | Individual white stone     |

Empty intersections are not detected — they are inferred from board geometry and stone positions during SGF conversion. Images are re-split into train/validation/test sets with base-image grouping to avoid data leakage from Roboflow augmentations.

## Model

The model is a fine-tuned [RT-DETR](https://arxiv.org/abs/2304.08069) with a ResNet-18vd backbone, starting from the COCO-pretrained [`PekingU/rtdetr_r18vd`](https://huggingface.co/PekingU/rtdetr_r18vd) checkpoint. RT-DETR was chosen because:

- **No NMS**: the transformer decoder eliminates duplicates internally, producing clean ONNX graphs.
- **Small**: ResNet-18vd backbone (~20MB) is lightweight enough for browser inference.
- **HF native**: first-class support in the `transformers` library with no vendor dependencies.

Fine-tuning is done with the HF `Trainer` API on the harmonized dataset. The trained model is published to [`kaya-go/moku-v1`](https://huggingface.co/kaya-go/moku-v1) on Hugging Face Hub.

## ONNX Export

The trained model is exported to ONNX via `torch.onnx.export` (opset 16+, dynamic batch axis) and verified with `onnxruntime`. The ONNX file is available in the [`kaya-go/moku-v1`](https://huggingface.co/kaya-go/moku-v1) model repository as `model.onnx`.

## Installation

```bash
pixi install
```

## License

AGPL-3.0
