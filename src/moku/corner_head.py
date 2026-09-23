"""Dense corner head (spec 004, lever 4): a heatmap of board corners on the detector's own features.

DETR queries miss 20–30% of the true corners. This small CenterNet-style head reads the
stride-8 map of the hybrid encoder and predicts, per cell, a corner score and the sub-cell
offset of the corner. It is decoded into the ``N_POINTS`` best local maxima, ``(x, y, score)``
in normalized image coordinates: class-agnostic, so partial boards need no special case.

The head is a ``corner_head`` submodule of the ``transformers`` model: EMA and ``save_pretrained``
carry its weights (``corner_head.*`` in ``model.safetensors``), ``from_pretrained`` drops them and
:func:`load_corner_head` puts them back. The ONNX export adds a ``corner_points`` output.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from moku.dataset import CATEGORIES

N_POINTS = 8
SIGMA = 1.0  # Gaussian radius of a target peak, in heatmap cells (8 px at 640)
PRIOR = 0.01
PREFIX = "corner_head."


def encoder_features(outputs) -> torch.Tensor:
    """The highest-resolution (stride 8) map of the hybrid encoder."""
    return outputs.encoder_last_hidden_state[0]


class CornerHead(nn.Module):
    def __init__(self, in_channels: int = 256, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, 1),  # corner logit, x / y offset in the cell
        )
        nn.init.constant_(self.net[-1].bias, 0.0)
        nn.init.constant_(self.net[-1].bias[:1], -torch.log(torch.tensor((1 - PRIOR) / PRIOR)).item())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)

    def loss(self, raw: torch.Tensor, labels: list[dict]) -> dict[str, torch.Tensor]:
        """Penalty-reduced focal loss on the heatmap + L1 on the offsets at the true corners."""
        raw = raw.float()
        b, _, h, w = raw.shape
        heat = torch.zeros(b, h, w, device=raw.device)
        pos, offsets = [], []
        ys, xs = torch.meshgrid(torch.arange(h, device=raw.device), torch.arange(w, device=raw.device), indexing="ij")
        for i, lab in enumerate(labels):
            corners = lab["boxes"][lab["class_labels"] == CATEGORIES["board_corner"], :2].float()
            for cx, cy in corners * torch.tensor([w, h], device=raw.device):
                px, py = cx.clamp(0, w - 1e-3).floor(), cy.clamp(0, h - 1e-3).floor()
                g = torch.exp(-((xs - px) ** 2 + (ys - py) ** 2) / (2 * SIGMA**2))
                heat[i] = torch.maximum(heat[i], g)
                pos.append((i, int(py), int(px)))
                offsets.append(torch.stack([cx - px, cy - py]))
        p = raw[:, 0].sigmoid().clamp(1e-4, 1 - 1e-4)
        is_pos = torch.zeros_like(heat, dtype=torch.bool)
        for i, y, x in pos:
            is_pos[i, y, x] = True
        pos_loss = (torch.log(p) * (1 - p) ** 2)[is_pos].sum()
        neg_loss = (torch.log(1 - p) * p**2 * (1 - heat) ** 4)[~is_pos].sum()
        n = max(len(pos), 1)
        losses = {"corner_heatmap": -(pos_loss + neg_loss) / n}
        if pos:
            idx = torch.tensor(pos, device=raw.device)
            pred = raw[idx[:, 0], 1:3, idx[:, 1], idx[:, 2]].sigmoid()
            losses["corner_offset"] = F.l1_loss(pred, torch.stack(offsets), reduction="sum") / n
        else:
            losses["corner_offset"] = raw[:, 1:3].sum() * 0.0
        return losses

    @staticmethod
    def decode(raw: torch.Tensor, k: int = N_POINTS) -> torch.Tensor:
        """``(B, k, 3)``: ``(x, y, score)`` of the ``k`` best local maxima, x and y in [0, 1]."""
        _, _, h, w = raw.shape
        heat = raw[:, 0:1].sigmoid()
        heat = heat * (F.max_pool2d(heat, 3, stride=1, padding=1) == heat).to(heat.dtype)
        scores, idx = heat.flatten(1).topk(k)
        offset = raw[:, 1:3].sigmoid().flatten(2)
        x = ((idx % w).to(raw.dtype) + offset[:, 0].gather(1, idx)) / w
        y = (torch.div(idx, w, rounding_mode="floor").to(raw.dtype) + offset[:, 1].gather(1, idx)) / h
        return torch.stack([x, y, scores], dim=-1)


def attach_corner_head(model: nn.Module) -> CornerHead:
    model.corner_head = CornerHead(getattr(model.config, "encoder_hidden_dim", 256))
    return model.corner_head


def corner_points(model: nn.Module, outputs) -> torch.Tensor | None:
    """Decoded corner points of a forward pass, or ``None`` for a model without the head."""
    head = getattr(model, "corner_head", None)
    return None if head is None else head.decode(head(encoder_features(outputs)))


def load_corner_head(model: nn.Module, repo: str, revision: str | None = None) -> bool:
    """Attach the head when the checkpoint has ``corner_head.*`` weights. Returns whether it did."""
    from safetensors import safe_open

    path = Path(repo) / "model.safetensors"
    if not path.exists():
        if Path(repo).exists():
            return False
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(repo, "model.safetensors", revision=revision))
    with safe_open(str(path), "pt") as f:
        state = {k[len(PREFIX) :]: f.get_tensor(k) for k in f.keys() if k.startswith(PREFIX)}
    if not state:
        return False
    head = attach_corner_head(model)
    head.load_state_dict(state)
    head.to(next(model.parameters()).device)
    return True
