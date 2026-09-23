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

from moku.corner_head import corner_points, load_corner_head

OPSET = 18
OUTPUTS = ("logits", "pred_boxes", "corner_points")  # corner_points only with the corner head


class _OnnxWrapper(torch.nn.Module):
    """Kaya's contract around a HF detector: [0, 1] pixels in, ``(logits, pred_boxes)`` out.

    Models that expect mean/std-normalized pixels get the normalization inside the graph, and
    ``logit_offset`` (see ``moku.evaluation.stone_offset``) is added to every class logit.
    """

    def __init__(self, model: torch.nn.Module, mean=None, std=None, logit_offset: float = 0.0):
        super().__init__()
        self.model = model
        self.normalize = mean is not None
        if self.normalize:
            self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))
        self.logit_offset = float(logit_offset)

    def forward(self, pixel_values: torch.Tensor):
        if self.normalize:
            pixel_values = (pixel_values - self.mean) / self.std
        out = self.model(pixel_values=pixel_values)
        points = corner_points(self.model, out)
        if points is None:
            return out.logits + self.logit_offset, out.pred_boxes
        return out.logits + self.logit_offset, out.pred_boxes, points


def load_for_export(source: str, logit_offset: float = 0.0) -> tuple[torch.nn.Module, int]:
    """Load a detector on CPU with eager attention; returns ``(wrapper, input_size)``."""
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    repo, _, revision = source.partition("@")
    processor = AutoImageProcessor.from_pretrained(repo, revision=revision or None)
    model = AutoModelForObjectDetection.from_pretrained(repo, revision=revision or None, attn_implementation="eager")
    load_corner_head(model, repo, revision or None)
    mean, std = (
        (processor.image_mean, processor.image_std) if getattr(processor, "do_normalize", False) else (None, None)
    )
    size = processor.size["height"]
    return _OnnxWrapper(model.eval().cpu(), mean, std, logit_offset).eval(), size


def export_onnx(source: str, output: Path, opset: int = OPSET, logit_offset: float = 0.0) -> Path:
    """Export ``source`` (Hub repo id, ``repo@revision`` or local dir) to ``output``."""
    wrapper, size = load_for_export(source, logit_offset)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (torch.randn(1, 3, size, size),),
        str(output),
        opset_version=opset,
        input_names=["pixel_values"],
        output_names=list(OUTPUTS[: 3 if hasattr(wrapper.model, "corner_head") else 2]),
        dynamic_axes={name: {0: "batch_size"} for name in ("pixel_values", *OUTPUTS)},
        do_constant_folding=True,
        dynamo=False,
    )
    return output


def verify_onnx(
    source: str, onnx_path: Path, atol: float = 5e-2, seed: int = 0, logit_offset: float = 0.0
) -> dict[str, float]:
    """Compare PyTorch and ONNX Runtime outputs on a random input.

    Queries can come out permuted (top-k ties, attention numerics), so each
    PyTorch query is matched to the nearest ONNX query by box L1 distance.
    """
    import onnxruntime as ort

    wrapper, size = load_for_export(source, logit_offset)
    x = torch.rand(1, 3, size, size, generator=torch.Generator().manual_seed(seed))
    with torch.no_grad():
        pt = [t[0].numpy() for t in wrapper(x)]
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = [t[0] for t in session.run(OUTPUTS[: len(pt)], {"pixel_values": x.numpy()})]
    (pt_logits, pt_boxes), (ort_logits, ort_boxes) = pt[:2], ort_out[:2]
    match = np.abs(pt_boxes[:, None, :] - ort_boxes[None, :, :]).sum(-1).argmin(axis=1)
    diffs = {
        "logits_max_abs_diff": float(np.abs(pt_logits - ort_logits[match]).max()),
        "boxes_max_abs_diff": float(np.abs(pt_boxes - ort_boxes[match]).max()),
    }
    if len(pt) == 3:
        diffs["corner_points_max_abs_diff"] = float(np.abs(pt[2] - ort_out[2]).max())
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
