"""ONNX export for Kaya.

The exported graph is the contract with the Kaya app
(``packages/board-recognition/src/moku-detector.ts``):

- input ``pixel_values``: ``(batch, 3, 640, 640)`` float32, RGB scaled to [0, 1]
  (no mean/std normalization);
- outputs ``logits``: ``(batch, 300, 3)`` raw class logits (Kaya applies the
  sigmoid) and ``pred_boxes``: ``(batch, 300, 4)`` normalized ``(cx, cy, w, h)``.

Post-processing (thresholds, corner selection, grid mapping) stays in Kaya.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

OPSET = 18


class _OnnxWrapper(torch.nn.Module):
    """Strip the HF output dataclass down to the ``(logits, pred_boxes)`` tuple."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor):
        out = self.model(pixel_values=pixel_values)
        return out.logits, out.pred_boxes


def load_for_export(source: str) -> tuple[torch.nn.Module, int]:
    """Load a detector on CPU with eager attention; returns ``(wrapper, input_size)``."""
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    repo, _, revision = source.partition("@")
    processor = AutoImageProcessor.from_pretrained(repo, revision=revision or None)
    model = AutoModelForObjectDetection.from_pretrained(repo, revision=revision or None, attn_implementation="eager")
    if getattr(processor, "do_normalize", False):
        raise ValueError("Kaya feeds [0, 1] pixels without mean/std normalization; this processor normalizes.")
    size = processor.size["height"]
    return _OnnxWrapper(model.eval().cpu()).eval(), size


def export_onnx(source: str, output: Path, opset: int = OPSET) -> Path:
    """Export ``source`` (Hub repo id, ``repo@revision`` or local dir) to ``output``."""
    wrapper, size = load_for_export(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (torch.randn(1, 3, size, size),),
        str(output),
        opset_version=opset,
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes"],
        dynamic_axes={name: {0: "batch_size"} for name in ("pixel_values", "logits", "pred_boxes")},
        do_constant_folding=True,
        dynamo=False,
    )
    return output


def verify_onnx(source: str, onnx_path: Path, atol: float = 5e-2, seed: int = 0) -> dict[str, float]:
    """Compare PyTorch and ONNX Runtime outputs on a random input.

    Queries can come out permuted (top-k ties, attention numerics), so each
    PyTorch query is matched to the nearest ONNX query by box L1 distance.
    """
    import onnxruntime as ort

    wrapper, size = load_for_export(source)
    x = torch.rand(1, 3, size, size, generator=torch.Generator().manual_seed(seed))
    with torch.no_grad():
        pt_logits, pt_boxes = (t[0].numpy() for t in wrapper(x))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_logits, ort_boxes = (t[0] for t in session.run(["logits", "pred_boxes"], {"pixel_values": x.numpy()}))
    match = np.abs(pt_boxes[:, None, :] - ort_boxes[None, :, :]).sum(-1).argmin(axis=1)
    diffs = {
        "logits_max_abs_diff": float(np.abs(pt_logits - ort_logits[match]).max()),
        "boxes_max_abs_diff": float(np.abs(pt_boxes - ort_boxes[match]).max()),
    }
    if max(diffs.values()) > atol:
        raise AssertionError(f"ONNX outputs differ from PyTorch beyond {atol}: {diffs}")
    return diffs


def benchmark_onnx(onnx_path: Path, threads: int = 1, runs: int = 10) -> dict[str, float]:
    """Latency of one 640×640 image on ONNX Runtime CPU.

    Kaya runs the model in single-threaded WebAssembly, so ``threads=1`` on a
    desktop CPU is an optimistic lower bound of what users see.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).random((1, 3, 640, 640), dtype=np.float32)
    session.run(None, {"pixel_values": x})
    t0 = time.perf_counter()
    for _ in range(runs):
        session.run(None, {"pixel_values": x})
    return {
        "latency_ms": (time.perf_counter() - t0) / runs * 1000,
        "size_mb": onnx_path.stat().st_size / 1e6,
        "threads": threads,
    }
