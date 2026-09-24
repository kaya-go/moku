"""Per-image Hungarian matching for RT-DETR and D-FINE (``transformers``' ``RTDetrHungarianMatcher``).

The stock matcher builds one cost matrix between *all* queries and *all* targets of the batch,
copies it to the CPU and only uses its per-image diagonal blocks. With goban images (up to ~300
objects each) that matrix grows with the square of the batch size and dominated the step time
(0.36 s per batch of 16 on an A100; 4× slower at batch 32). This version computes the same costs
image by image, so matching scales linearly with the batch.
"""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment

_EMPTY = torch.zeros(0, dtype=torch.int64)


def _corners(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def _generalized_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise GIoU of ``(N, 4)`` and ``(M, 4)`` xyxy boxes."""
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    inter = (torch.min(a[:, None, 2:], b[None, :, 2:]) - torch.max(a[:, None, :2], b[None, :, :2])).clamp(min=0)
    inter = inter[..., 0] * inter[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    iou = inter / union
    hull = (torch.max(a[:, None, 2:], b[None, :, 2:]) - torch.min(a[:, None, :2], b[None, :, :2])).clamp(min=0)
    hull = hull[..., 0] * hull[..., 1]
    return iou - (hull - union) / hull


@torch.no_grad()
def blockwise_forward(self, outputs: dict, targets: list[dict]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Drop-in ``RTDetrHungarianMatcher.forward``: same costs, one image at a time."""
    logits, pred_boxes = outputs["logits"].float(), outputs["pred_boxes"].float()
    costs = []
    for b, target in enumerate(targets):
        ids, boxes = target["class_labels"], target["boxes"].float()
        if len(ids) == 0:
            costs.append(None)
            continue
        if self.use_focal_loss:
            prob = logits[b].sigmoid()[:, ids]
            neg = (1 - self.alpha) * (prob**self.gamma) * (-(1 - prob + 1e-8).log())
            pos = self.alpha * ((1 - prob) ** self.gamma) * (-(prob + 1e-8).log())
            class_cost = pos - neg
        else:
            class_cost = -logits[b].softmax(-1)[:, ids]
        bbox_cost = torch.cdist(pred_boxes[b], boxes, p=1)
        giou_cost = -_generalized_iou(_corners(pred_boxes[b]), _corners(boxes))
        costs.append(self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost)
    # One device-to-host copy for the whole batch.
    flat = torch.cat([c.flatten() for c in costs if c is not None]).cpu() if any(c is not None for c in costs) else None
    indices, offset = [], 0
    for cost in costs:
        if cost is None:
            indices.append((_EMPTY, _EMPTY))
            continue
        block = flat[offset : offset + cost.numel()].view(cost.shape).numpy()
        offset += cost.numel()
        rows, cols = linear_sum_assignment(block)
        indices.append((torch.as_tensor(rows, dtype=torch.int64), torch.as_tensor(cols, dtype=torch.int64)))
    return indices


def install() -> None:
    """Patch ``transformers``' RT-DETR matcher (also used by D-FINE) with :func:`blockwise_forward`."""
    from transformers.loss.loss_rt_detr import RTDetrHungarianMatcher

    RTDetrHungarianMatcher.forward = blockwise_forward
