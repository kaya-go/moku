"""Run a detector over images and keep its raw outputs.

Every metric is computed from the raw per-query outputs — class logits and
normalized ``(cx, cy, w, h)`` boxes, i.e. exactly what the ONNX model hands to
Kaya — so a single inference pass feeds detection metrics, board metrics and
threshold sweeps alike.

Model sources accepted by :func:`load_detector`:

- ``path/to/model.onnx`` — run with onnxruntime and Kaya's own preprocessing;
- ``hf://buckets/<ns>/<bucket>/<run>/<best|last>`` — a checkpoint written by a training run;
- a local directory or a Hugging Face Hub repo id (``org/name[@revision]``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from PIL import Image

INPUT_SIZE = 640


@dataclass
class RawPrediction:
    """Raw detector outputs for one image."""

    logits: np.ndarray  # (Q, C) class logits
    boxes: np.ndarray  # (Q, 4) normalized cxcywh
    width: int  # original image size, in pixels
    height: int


class Detector(Protocol):
    name: str

    def predict(self, images: list[Image.Image]) -> list[RawPrediction]: ...


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def kaya_preprocess(image: Image.Image, size: int = INPUT_SIZE) -> np.ndarray:
    """Replicate Kaya's ``preprocess``: bilinear squash to ``size²``, scale to [0, 1], CHW.

    Mirrors ``moku-postprocess.ts`` sample for sample (including its border
    handling), so ONNX evaluations see the exact tensor the app feeds the model.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = rgb.shape[:2]

    def axis(n_src: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        src = (np.arange(size) + 0.5) * (n_src / size) - 0.5
        lo = np.maximum(0, np.floor(src)).astype(int)
        hi = np.minimum(lo + 1, n_src - 1)
        return lo, hi, (src - np.floor(src)).astype(np.float32)

    x0, x1, fx = axis(width)
    y0, y1, fy = axis(height)
    fx, fy = fx[None, :, None], fy[:, None, None]
    top = rgb[y0][:, x0] * (1 - fx) + rgb[y0][:, x1] * fx
    bottom = rgb[y1][:, x0] * (1 - fx) + rgb[y1][:, x1] * fx
    out = (top * (1 - fy) + bottom * fy) / 255.0
    return out.transpose(2, 0, 1)[None].astype(np.float32)


class TorchDetector:
    """A ``transformers`` object-detection model with its image processor."""

    def __init__(self, model, image_processor, name: str, device: str | None = None):
        self.device = device or default_device()
        self.model = model.to(self.device).eval()
        self.image_processor = image_processor
        self.name = name

    @classmethod
    def from_pretrained(cls, source: str, device: str | None = None, name: str | None = None) -> TorchDetector:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        repo, _, revision = source.partition("@")
        processor = AutoImageProcessor.from_pretrained(repo, revision=revision or None)
        model = AutoModelForObjectDetection.from_pretrained(repo, revision=revision or None)
        return cls(model, processor, name=name or source, device=device)

    @torch.no_grad()
    def predict(self, images: list[Image.Image]) -> list[RawPrediction]:
        images = [im.convert("RGB") for im in images]
        pixel_values = self.image_processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)
        out = self.model(pixel_values=pixel_values)
        logits = out.logits.float().cpu().numpy()
        boxes = out.pred_boxes.float().cpu().numpy()
        return [RawPrediction(lg, bx, im.width, im.height) for lg, bx, im in zip(logits, boxes, images)]


class OnnxDetector:
    """An exported ``model.onnx`` fed with Kaya's preprocessing."""

    def __init__(self, path: str | Path, threads: int | None = None):
        import onnxruntime as ort

        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.name = str(path)

    def predict(self, images: list[Image.Image]) -> list[RawPrediction]:
        preds = []
        for im in images:
            logits, boxes = self.session.run(["logits", "pred_boxes"], {"pixel_values": kaya_preprocess(im)})
            preds.append(RawPrediction(logits[0], boxes[0], im.width, im.height))
        return preds


def resolve_source(source: str) -> str:
    """Local directory for a bucket checkpoint (downloaded under ``runs/``); anything else is passed through."""
    from moku.runs import BUCKET_PREFIX, pull_checkpoint

    return str(pull_checkpoint(source)) if source.startswith(BUCKET_PREFIX) else source


def load_detector(source: str, device: str | None = None) -> Detector:
    """Load a detector from an ONNX file, a bucket checkpoint, a local dir or a Hub repo."""
    if source.endswith(".onnx"):
        return OnnxDetector(source)
    return TorchDetector.from_pretrained(resolve_source(source), device, name=source)


def run_detector(detector: Detector, images, batch_size: int = 8) -> list[RawPrediction]:
    """Run ``detector`` over an iterable of PIL images, in batches."""
    preds: list[RawPrediction] = []
    batch: list[Image.Image] = []
    for image in images:
        batch.append(image)
        if len(batch) == batch_size:
            preds.extend(detector.predict(batch))
            batch = []
    if batch:
        preds.extend(detector.predict(batch))
    return preds
