# Moku

Object detection model for converting goban (Go board) photos and screenshots to SGF files.

| Resource | Link |
|----------|------|
| Model | [kaya-go/moku-v1](https://huggingface.co/kaya-go/moku-v1) |
| Dataset | [kaya-go/moku-v1](https://huggingface.co/datasets/kaya-go/moku-v1) |

These ONNX models power the [Kaya](https://github.com/kaya-go/kaya) app, enabling automatic game transcription from board images.

## Features

- **Dataset Building**: Generate and augment training data for goban detection
- **Training**: Train object detection models to recognize board positions
- **Evaluation**: Validate model accuracy on test datasets
- **Export**: Convert trained models to ONNX format for web deployment
- **Publishing**: Upload models to Hugging Face Hub

## Installation

```bash
pixi install
```

## License

AGPL-3.0
