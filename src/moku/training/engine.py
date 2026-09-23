"""Training loop for moku detectors (RT-DETR, D-FINE) — the recipe of the reference repos.

- AdamW with a lower learning rate for the backbone and no weight decay on norms and biases;
- linear warm-up, flat, then cosine decay, per iteration;
- weight EMA, evaluated and saved instead of the raw weights;
- strong augmentation, then a few final epochs with flips only (D-FINE's "stop augmentation");
- checkpoint selection on validation **board** metrics (Kaya's pipeline), not mAP.

Everything a run produces lands in ``<output_dir>/<run_name>/``: ``config.json``,
``metrics.jsonl`` (one JSON record per log step and per evaluation), ``train.log``,
``best/`` and ``last/`` (``save_pretrained`` checkpoints with their ``eval.json``)
and ``summary.json`` once the run is over. On HF Jobs that directory is a
mounted bucket, so progress can be followed while the run is going.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from moku.dataset import CATEGORIES, ID_TO_CATEGORY
from moku.training.data import DetectionDataset, collate, train_augmentation, train_indices

MODELS = {
    "rtdetr-r18": "PekingU/rtdetr_r18vd",
    "dfine-s": "ustc-community/dfine-small-obj2coco",
    "dfine-n": "ustc-community/dfine-nano-coco",
}


@dataclass
class TrainConfig:
    run_name: str
    model: str = "rtdetr-r18"  # a key of MODELS or a Hub repo id
    dataset: str = "kaya-go/moku-v4"
    train_exclude: str | None = None  # comma-separated source datasets left out of training
    output_dir: str = "runs"
    epochs: int = 72
    no_aug_epochs: int = 8
    batch_size: int = 16
    lr: float = 1e-4
    backbone_lr_mult: float = 0.1
    weight_decay: float = 1e-4
    max_grad_norm: float = 0.1
    warmup_iters: int = 500
    flat_fraction: float = 0.5  # share of the iterations at peak lr before the cosine decay
    min_lr_ratio: float = 0.05
    ema_decay: float = 0.999
    ema_tau: float = 500.0  # EMA warm-up, in iterations
    oversample_real: int = 3
    source: str | None = None  # "real" or "generated" to train on one source only
    seed: int = 0
    workers: int | None = None  # default: available CPUs minus 2
    amp: bool = True  # bf16 autocast on CUDA
    eval_every: int = 1
    log_every: int = 25
    max_hours: float | None = None  # stop cleanly (and save) past this budget
    limit_train: int | None = None  # smoke tests: truncate the training set
    limit_eval: int | None = None
    device: str | None = None

    @property
    def base_model(self) -> str:
        return MODELS.get(self.model, self.model)


# ---------------------------------------------------------------------------
# Optimizer, schedule, EMA
# ---------------------------------------------------------------------------


def param_groups(model: torch.nn.Module, lr: float, backbone_lr_mult: float, weight_decay: float) -> list[dict]:
    """Backbone vs rest × with vs without weight decay (1-D tensors: norms, biases, scales)."""
    groups = {(bb, decay): [] for bb in (True, False) for decay in (True, False)}
    for name, param in model.named_parameters():
        if param.requires_grad:
            groups[(".backbone." in name, param.ndim > 1)].append(param)
    return [
        {
            "params": params,
            "lr": lr * (backbone_lr_mult if bb else 1.0),
            "weight_decay": weight_decay if decay else 0.0,
            "name": f"{'backbone' if bb else 'head'}{'' if decay else '_no_decay'}",
        }
        for (bb, decay), params in groups.items()
        if params
    ]


def lr_factor(it: int, total: int, warmup: int, flat_fraction: float, min_ratio: float) -> float:
    """Linear warm-up → flat → cosine decay down to ``min_ratio`` at ``total``."""
    if it < warmup:
        return (it + 1) / warmup
    flat_end = max(warmup, int(total * flat_fraction))
    if it < flat_end:
        return 1.0
    progress = min(1.0, (it - flat_end) / max(1, total - flat_end))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


class ModelEMA:
    """Exponential moving average of the weights, with the RT-DETR/D-FINE warm-up.

    The decay ramps up as ``decay · (1 − exp(−updates / tau))``, so the average
    follows the model closely at first; :meth:`restart` resets the ramp.
    """

    def __init__(self, model: torch.nn.Module, decay: float, tau: float):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.tau, self.updates = decay, tau, 0
        self._pairs = None

    def current_decay(self) -> float:
        return self.decay * (1 - math.exp(-self.updates / self.tau))

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        if self._pairs is None:  # state_dict tensors share storage with the modules: resolve them once
            ema_state, state = self.module.state_dict(), model.state_dict()
            names = [n for n, v in ema_state.items() if v.dtype.is_floating_point]
            self._pairs = ([ema_state[n] for n in names], [state[n] for n in names])
        self.updates += 1
        d = self.current_decay()
        ema_tensors, model_tensors = self._pairs
        torch._foreach_mul_(ema_tensors, d)
        torch._foreach_add_(ema_tensors, model_tensors, alpha=1 - d)

    def restart(self) -> None:
        self.updates = 0


# ---------------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------------


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self) -> None:
        for s in self.streams:
            s.flush()

    def isatty(self) -> bool:
        return False


class RunDir:
    """``<output_dir>/<run_name>/``: JSONL metrics, stdout copy and checkpoints."""

    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self._log = open(root / "train.log", "a", buffering=1)
        sys.stdout = _Tee(sys.__stdout__, self._log)

    def write_json(self, name: str, data: dict) -> None:
        (self.root / name).write_text(json.dumps(data, indent=2, default=str))

    def log(self, record: dict) -> None:
        with open(self.root / "metrics.jsonl", "a") as f:
            f.write(json.dumps(record, default=float) + "\n")

    def save_checkpoint(self, name: str, model, processor, info: dict) -> None:
        """``save_pretrained`` into a local temp dir, then copy (the root may be a bucket mount)."""
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            processor.save_pretrained(tmp)
            Path(tmp, "eval.json").write_text(json.dumps(info, indent=2, default=float))
            target = self.root / name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(tmp, target)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_model(model, processor, split, device: str) -> dict[str, float]:
    """Validation metrics of ``model``, computed exactly like ``moku eval``."""
    from moku.evaluation import evaluate
    from moku.inference import TorchDetector

    was_training = model.training
    result = evaluate(TorchDetector(model, processor, name="ema", device=device), split, "validation", batch_size=16)
    model.train(was_training)
    b = result.board
    return {
        "perfect": b["perfect"],
        "errors": b["errors"],
        "le2_errors": b["le2_errors"],
        "corner_fail": b["corner_fail"],
        **{k: v for k, v in result.detection.items() if not k.startswith("AP/")},
    }


def selection_key(metrics: dict[str, float]) -> tuple[float, float]:
    """Best checkpoint: most perfect boards, then fewest wrong intersections."""
    return metrics["perfect"], -metrics["errors"]


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def available_cpus() -> int:
    """CPUs this container may use (cgroup quota when set, else affinity)."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _worker_init(worker_id: int) -> None:
    import cv2

    # One thread per worker: OpenCV otherwise spawns a pool per worker, and the oversubscribed
    # CPUs starve the main process (loss matching, kernel launches).
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def _loader(dataset, cfg: TrainConfig, workers: int, seed: int) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
        worker_init_fn=_worker_init,
        generator=torch.Generator().manual_seed(seed),
    )


def _default_device() -> str:
    from moku.inference import default_device

    return default_device()


def train(cfg: TrainConfig, extra_config: dict | None = None) -> dict:
    """Train, evaluate every ``eval_every`` epochs and keep the best and last EMA checkpoints."""
    from datasets import load_dataset
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    start = time.time()
    run = RunDir(Path(cfg.output_dir) / cfg.run_name)
    seed_everything(cfg.seed)
    device = cfg.device or _default_device()
    workers = cfg.workers if cfg.workers is not None else max(1, available_cpus() - 2)  # 2 CPUs for the main process
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ds = load_dataset(cfg.dataset)
    exclude = tuple(cfg.train_exclude.split(",")) if cfg.train_exclude else ()
    indices = train_indices(ds["train"], cfg.oversample_real, cfg.source, exclude)
    if cfg.limit_train:
        indices = random.Random(cfg.seed).sample(indices, min(cfg.limit_train, len(indices)))
    val = ds["validation"]
    if cfg.limit_eval:
        val = val.select(range(min(cfg.limit_eval, len(val))))
    strong = DetectionDataset(ds["train"], indices, train_augmentation(strong=True))
    light = DetectionDataset(ds["train"], indices, train_augmentation(strong=False))

    processor = AutoImageProcessor.from_pretrained(cfg.base_model)
    model = AutoModelForObjectDetection.from_pretrained(
        cfg.base_model,
        num_labels=len(CATEGORIES),
        id2label=ID_TO_CATEGORY,
        label2id=CATEGORIES,
        ignore_mismatched_sizes=True,
    ).to(device)
    ema = ModelEMA(model, cfg.ema_decay, cfg.ema_tau)
    optimizer = torch.optim.AdamW(
        param_groups(model, cfg.lr, cfg.backbone_lr_mult, cfg.weight_decay), lr=cfg.lr, betas=(0.9, 0.999)
    )
    iters_per_epoch = len(indices) // cfg.batch_size
    total_iters = iters_per_epoch * cfg.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda it: lr_factor(it, total_iters, cfg.warmup_iters, cfg.flat_fraction, cfg.min_lr_ratio)
    )
    amp = cfg.amp and device == "cuda"

    run.write_json(
        "config.json",
        {
            **asdict(cfg),
            "base_model": cfg.base_model,
            "device": device,
            "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
            "workers": workers,
            "train_images": len(indices),
            "train_unique_images": len(set(indices)),
            "val_images": len(val),
            "iters_per_epoch": iters_per_epoch,
            "total_iters": total_iters,
            "params": sum(p.numel() for p in model.parameters()),
            "param_groups": {g["name"]: sum(p.numel() for p in g["params"]) for g in optimizer.param_groups},
            "versions": _versions(),
            **(extra_config or {}),
        },
    )
    print(f"Run {cfg.run_name}: {cfg.base_model} on {device}, {len(indices)} train images, {workers} workers")
    print(f"{iters_per_epoch} iters/epoch × {cfg.epochs} epochs = {total_iters} iters")

    best: dict | None = None
    it = 0
    stop_epoch = cfg.epochs - cfg.no_aug_epochs
    loader = _loader(strong, cfg, workers, cfg.seed)
    stopped_early = False
    for epoch in range(1, cfg.epochs + 1):
        if epoch == stop_epoch + 1 and cfg.no_aug_epochs > 0:
            print(f"Epoch {epoch}: strong augmentation off, EMA restarted")
            loader = _loader(light, cfg, workers, cfg.seed + epoch)
            ema.restart()
        model.train()
        t_epoch, t_data, n_seen, skipped = time.time(), 0.0, 0, 0
        running: dict[str, float] = {}
        t0 = time.time()
        for images, labels in loader:
            t_data += time.time() - t0
            images = images.to(device, non_blocking=True)
            labels = [{k: v.to(device, non_blocking=True) for k, v in lab.items()} for lab in labels]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                out = model(pixel_values=images, labels=labels)
            loss = out.loss
            if not torch.isfinite(loss):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                t0 = time.time()
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            ema.update(model)
            it += 1
            n_seen += images.shape[0]
            running["loss"] = running.get("loss", 0.0) + loss.item()
            running["grad_norm"] = running.get("grad_norm", 0.0) + grad_norm.item()
            running["n"] = running.get("n", 0) + 1
            if it % cfg.log_every == 0:
                n = running.pop("n")
                record = {
                    "type": "train",
                    "iter": it,
                    "epoch": epoch,
                    **{k: v / n for k, v in running.items()},
                    "lr_head": scheduler.get_last_lr()[_group_index(optimizer, "head")],
                    "ema_decay": ema.current_decay(),
                    "img_per_s": n_seen / (time.time() - t_epoch),
                    "data_wait": t_data / (time.time() - t_epoch),
                    **{f"loss/{k}": v.item() for k, v in (out.loss_dict or {}).items() if "_aux_" not in k},
                }
                run.log(record)
                print(
                    f"ep {epoch} it {it}/{total_iters} loss {record['loss']:.3f} lr {record['lr_head']:.2e} "
                    f"{record['img_per_s']:.0f} img/s (data wait {record['data_wait']:.0%})"
                )
                running = {}
            t0 = time.time()

        elapsed_h = (time.time() - start) / 3600
        over_budget = cfg.max_hours is not None and elapsed_h * (epoch + 1) / epoch > cfg.max_hours
        last_epoch = epoch == cfg.epochs or over_budget
        epoch_record = {
            "type": "epoch",
            "epoch": epoch,
            "iter": it,
            "epoch_s": time.time() - t_epoch,
            "img_per_s": n_seen / (time.time() - t_epoch),
            "skipped_steps": skipped,
            "elapsed_h": elapsed_h,
        }
        if epoch % cfg.eval_every == 0 or last_epoch:
            t_eval = time.time()
            metrics = evaluate_model(ema.module, processor, val, device)
            epoch_record.update({f"val/{k}": v for k, v in metrics.items()}, eval_s=time.time() - t_eval)
            print(
                f"== epoch {epoch}: val perfect {metrics['perfect']:.0%}, errors {metrics['errors']:.1f}, "
                f"corner fail {metrics['corner_fail']:.0%}, mAP@50 {metrics['mAP@50']:.3f}, "
                f"TP score stone {metrics['stone_tp_score']:.2f} / corner {metrics['corner_tp_score']:.2f}"
            )
            if best is None or selection_key(metrics) > selection_key(best["metrics"]):
                best = {"epoch": epoch, "iter": it, "metrics": metrics}
                run.save_checkpoint("best", ema.module, processor, best)
                print(f"   ★ new best (epoch {epoch})")
        run.log(epoch_record)
        if over_budget and epoch < cfg.epochs:
            print(f"Time budget of {cfg.max_hours} h reached after epoch {epoch}: stopping.")
            stopped_early = True
            break

    last = {"epoch": epoch, "iter": it, "metrics": metrics}
    run.save_checkpoint("last", ema.module, processor, last)
    summary = {
        "status": "stopped_early" if stopped_early else "done",
        "hours": (time.time() - start) / 3600,
        "best": best,
        "last": last,
    }
    run.write_json("summary.json", summary)
    print(f"Done in {summary['hours']:.2f} h. Best epoch {best['epoch']}: {best['metrics']}")
    return summary


def _group_index(optimizer, name: str) -> int:
    return next(i for i, g in enumerate(optimizer.param_groups) if g["name"] == name)


def _versions() -> dict[str, str]:
    import albumentations
    import transformers

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "albumentations": albumentations.__version__,
    }
