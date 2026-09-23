"""Training building blocks: schedule, parameter groups, EMA and targets."""

import numpy as np
import pytest
import torch

from moku.training.data import clip_boxes, to_target, train_indices
from moku.training.engine import ModelEMA, lr_factor, param_groups


def test_lr_schedule_warmup_flat_cosine():
    total, warmup = 1000, 100
    assert lr_factor(0, total, warmup, 0.5, 0.05) == pytest.approx(0.01)
    assert lr_factor(99, total, warmup, 0.5, 0.05) == pytest.approx(1.0)
    assert lr_factor(499, total, warmup, 0.5, 0.05) == 1.0
    assert lr_factor(750, total, warmup, 0.5, 0.05) == pytest.approx(0.525)
    assert lr_factor(1000, total, warmup, 0.5, 0.05) == pytest.approx(0.05)


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.BatchNorm2d(4))
        self.head = torch.nn.Linear(4, 2)


def test_param_groups_split_backbone_and_decay():
    model = torch.nn.Module()
    model.model = _Tiny()  # names look like HF's `model.backbone.*`
    groups = {g["name"]: g for g in param_groups(model, 1e-4, 0.1, 1e-4)}
    assert groups["backbone"]["lr"] == pytest.approx(1e-5)
    assert groups["backbone_no_decay"]["weight_decay"] == 0.0
    assert groups["head"]["weight_decay"] == 1e-4
    assert sum(len(g["params"]) for g in groups.values()) == len(list(model.parameters()))
    assert all(p.ndim > 1 for p in groups["head"]["params"])


def test_ema_follows_then_averages():
    model = torch.nn.Linear(2, 1)
    ema = ModelEMA(model, decay=0.9, tau=1.0)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema.update(model)  # decay ramps from 0: the first update nearly copies
    first = ema.module.weight.clone()
    with torch.no_grad():
        model.weight.fill_(2.0)
    for _ in range(50):
        ema.update(model)
    assert torch.all(ema.module.weight > first)
    assert torch.allclose(ema.module.weight, torch.full_like(first, 2.0), atol=1e-2)
    ema.restart()
    assert ema.current_decay() == 0.0


def test_targets_are_normalized_cxcywh():
    boxes, cats = clip_boxes([[-5, 10, 20, 20], [630, 630, 20, 20], [100, 100, 0, 5]], [0, 1, 2], 640, 640)
    assert cats == [0, 1]  # the zero-width box is dropped
    assert boxes[0] == [0.0, 10.0, 15.0, 20.0]
    target = to_target(boxes, cats)
    np.testing.assert_allclose(target["boxes"][0].numpy(), np.array([7.5, 20, 15, 20]) / 640, rtol=1e-6)
    assert target["class_labels"].dtype == torch.int64


def test_train_indices_oversample_real():
    split = {"source_dataset": ["go_chess", "generated", "go_game_v10", "generated"]}
    assert sorted(train_indices(split, 3)) == [0, 0, 0, 1, 2, 2, 2, 3]
    assert train_indices(split, 1, "generated") == [1, 3]
