# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "torch",
#     "torchvision",
#     "transformers>=5.2.0",
#     "datasets>=4.6.0",
#     "accelerate>=1.12.0",
#     "wandb",
#     "pycocotools>=2.0.11",
#     "torchmetrics>=1.7.0",
#     "faster-coco-eval>=1.7.0",
#     "huggingface_hub>=1.5.0",
#     "albumentations>=1.4.20",
#     "numpy",
# ]
# ///
"""Fine-tune RT-DETR r18vd on kaya-go/moku dataset.

Self-contained training script for HF Jobs. Supports:
  - Single-stage training on v3 (--stage 1 with default v3 dataset, no config)
  - Two-stage training on v2: synthetic pre-train (--stage 1) + real fine-tune (--stage 2)

Usage (single-stage on v3, local CPU test):
    uv run scripts/train.py --stage 1 --num-epochs 2 --use-cpu

Usage (single-stage on v3, HF Jobs):
    hf jobs uv run --detach --flavor a10g-large --timeout 2h \\
        --secrets HF_TOKEN --secrets WANDB_API_KEY \\
        scripts/train.py --stage 1 --run-name r7_lr3e-4 --lr 3e-4 --num-epochs 500

Usage (two-stage on v2):
    hf jobs uv run --detach --flavor a10g-small --timeout 3h \\
        --secrets HF_TOKEN --secrets WANDB_API_KEY \\
        scripts/train.py --stage 1 --dataset kaya-go/moku-v2 --run-name v2_stage1

Resume interrupted training:
    uv run scripts/train.py --stage 1 --run-name v2_stage1 --resume
"""

from __future__ import annotations

import argparse
import os
import sys

import albumentations as A
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from transformers import (
    RTDetrForObjectDetection,
    RTDetrImageProcessor,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_MODEL = "PekingU/rtdetr_r18vd"
HF_DATASET = "kaya-go/moku-v3"
HF_MODEL = "kaya-go/moku-v2"
STAGE1_REVISION = "stage1"

CATEGORIES = {"black_stone": 0, "white_stone": 1, "board_corner": 2}
ID_TO_CATEGORY = {v: k for k, v in CATEGORIES.items()}
NUM_LABELS = len(CATEGORIES)


def _compute_cd_ap_inline(
    raw: list[dict],
    distance_threshold: float = 0.02,
) -> dict[str, float]:
    """Compute per-class center-distance AP at a single distance threshold.

    Lightweight version inlined in train.py to avoid importing moku.evaluation
    (which is not available on HF Jobs). Returns {class_name: cdAP_value}.
    """
    results: dict[str, float] = {}
    for cls_id, cls_name in ID_TO_CATEGORY.items():
        all_detections: list[tuple[float, bool]] = []
        total_gt = 0

        for img in raw:
            gt_mask = img["gt_labels"] == cls_id
            pred_mask = img["pred_labels"] == cls_id

            g_centers = img["gt_centers"][gt_mask] if gt_mask.any() else torch.zeros(0, 2)
            p_centers = img["pred_centers"][pred_mask] if pred_mask.any() else torch.zeros(0, 2)
            p_scores = img["pred_scores"][pred_mask] if pred_mask.any() else torch.zeros(0)
            img_diag = img["img_diag"]

            total_gt += g_centers.shape[0]

            if p_centers.shape[0] == 0:
                continue
            if g_centers.shape[0] == 0:
                for s in p_scores.tolist():
                    all_detections.append((s, False))
                continue

            sorted_idx = torch.argsort(p_scores, descending=True)
            p_centers = p_centers[sorted_idx]
            p_scores = p_scores[sorted_idx]

            dists = torch.cdist(p_centers.float(), g_centers.float())
            matched_gt: set[int] = set()

            for pi in range(p_centers.shape[0]):
                score = float(p_scores[pi])
                best_gi = -1
                best_dist = float("inf")
                for gi in range(g_centers.shape[0]):
                    if gi in matched_gt:
                        continue
                    d = float(dists[pi, gi])
                    if d < best_dist:
                        best_dist = d
                        best_gi = gi

                if best_gi >= 0 and best_dist / img_diag <= distance_threshold:
                    all_detections.append((score, True))
                    matched_gt.add(best_gi)
                else:
                    all_detections.append((score, False))

        if total_gt == 0:
            results[cls_name] = 0.0
            continue

        all_detections.sort(key=lambda x: -x[0])
        tp_cum = 0
        fp_cum = 0
        precisions = np.empty(len(all_detections))
        recalls = np.empty(len(all_detections))

        for i, (_, is_tp) in enumerate(all_detections):
            if is_tp:
                tp_cum += 1
            else:
                fp_cum += 1
            precisions[i] = tp_cum / (tp_cum + fp_cum)
            recalls[i] = tp_cum / total_gt

        ap = 0.0
        for t in np.linspace(0, 1, 101):
            mask = recalls >= t
            if mask.any():
                ap += precisions[mask].max()
        ap /= 101.0

        results[cls_name] = round(float(ap), 4)

    return results


def _compute_corner_recall_at_k_inline(
    raw: list[dict],
    k: int = 4,
    dist_threshold: float = 0.02,
) -> float:
    """Corner recall using top-k predictions per image (inline version).

    Mirrors inference where only the top-4 corners are kept.
    Returns mean recall across images with GT corners.
    """
    CORNER_CLS_ID = 2
    recalls: list[float] = []

    for img in raw:
        gt_mask = img["gt_labels"] == CORNER_CLS_ID
        pred_mask = img["pred_labels"] == CORNER_CLS_ID

        g_centers = img["gt_centers"][gt_mask] if gt_mask.any() else torch.zeros(0, 2)
        n_gt = g_centers.shape[0]
        if n_gt == 0:
            continue

        p_centers = img["pred_centers"][pred_mask] if pred_mask.any() else torch.zeros(0, 2)
        p_scores = img["pred_scores"][pred_mask] if pred_mask.any() else torch.zeros(0)
        img_diag = img["img_diag"]

        if p_centers.shape[0] == 0:
            recalls.append(0.0)
            continue

        topk = min(k, p_centers.shape[0])
        topk_idx = torch.argsort(p_scores, descending=True)[:topk]
        p_centers = p_centers[topk_idx]

        dists = torch.cdist(p_centers.float(), g_centers.float())
        matched_gt: set[int] = set()
        n_matched = 0

        for pi in range(p_centers.shape[0]):
            best_gi = -1
            best_dist = float("inf")
            for gi in range(n_gt):
                if gi in matched_gt:
                    continue
                d = float(dists[pi, gi])
                if d < best_dist:
                    best_dist = d
                    best_gi = gi

            if best_gi >= 0 and best_dist / img_diag <= dist_threshold:
                matched_gt.add(best_gi)
                n_matched += 1

        recalls.append(n_matched / n_gt)

    return round(float(np.mean(recalls)) if recalls else 0.0, 4)


class MAPEvalCallback(TrainerCallback):
    """Compute COCO mAP, center-distance AP, and corner recall on eval set.

    All metrics are threshold-free:
    - mAP@50, mAP@50:95 (COCO standard, all predictions kept)
    - stone_cdAP@2% (center-distance AP at 2% of image diagonal)
    - corner_R4 (top-4 corner recall — matches inference behaviour)

    When ``save_best_artifact`` is True, saves the best model (by mAP@50)
    as a W&B artifact whenever a new best is reached.
    """

    def __init__(
        self,
        eval_dataset,
        image_processor: RTDetrImageProcessor,
        save_best_artifact: bool = False,
        eval_batch_size: int = 64,
    ):
        self.eval_dataset = eval_dataset
        self.image_processor = image_processor
        self.save_best_artifact = save_best_artifact
        self.eval_batch_size = eval_batch_size
        self.best_map_50: float = -1.0
        self.trainer: Trainer | None = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch) + 1 if state.epoch is not None else 1
        total = int(args.num_train_epochs)
        print(f"\n{'=' * 60}")
        print(f"  Epoch {epoch}/{total} — Training")
        print(f"{'=' * 60}")

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        epoch = int(state.epoch) if state.epoch else 0
        print(f"\n{'-' * 60}")
        print(f"  Epoch {epoch} — Evaluation")
        print(f"{'-' * 60}")

        if model is None:
            return
        from torchmetrics.detection import MeanAveragePrecision

        device = next(model.parameters()).device
        model.eval()

        metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            max_detection_thresholds=[1, 10, 400],
            backend="faster_coco_eval",
        )

        # Collect raw predictions for cdAP computation
        raw_predictions: list[dict] = []

        dataloader = torch.utils.data.DataLoader(
            self.eval_dataset, batch_size=self.eval_batch_size, collate_fn=collate_fn, shuffle=False
        )

        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"]

            with torch.no_grad():
                outputs = model(pixel_values=pixel_values)

            orig_sizes = torch.stack([lab["orig_size"] for lab in labels]).to(device)
            # threshold=0 keeps all predictions for proper mAP and cdAP computation
            results = self.image_processor.post_process_object_detection(
                outputs, target_sizes=orig_sizes, threshold=0.0
            )

            preds = [
                {"boxes": r["boxes"].cpu(), "scores": r["scores"].cpu(), "labels": r["labels"].cpu()} for r in results
            ]

            targets = []
            for lab in labels:
                boxes_cxcywh = lab["boxes"]
                orig_h, orig_w = lab["orig_size"]
                cx, cy, w, h = boxes_cxcywh.unbind(-1)
                abs_boxes = torch.stack(
                    [(cx - w / 2) * orig_w, (cy - h / 2) * orig_h, (cx + w / 2) * orig_w, (cy + h / 2) * orig_h],
                    dim=-1,
                )
                targets.append({"boxes": abs_boxes.cpu(), "labels": lab["class_labels"].cpu()})

            metric.update(preds, targets)

            # Collect raw data for cdAP
            for res, lab in zip(results, labels):
                orig_h, orig_w = lab["orig_size"]
                img_diag = float((orig_h**2 + orig_w**2) ** 0.5)
                pred_boxes = res["boxes"].cpu()
                pred_scores = res["scores"].cpu()
                pred_labels = res["labels"].cpu()
                pred_cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
                pred_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
                pred_centers = torch.stack([pred_cx, pred_cy], dim=-1) if pred_boxes.numel() > 0 else torch.zeros(0, 2)

                boxes_cxcywh = lab["boxes"]
                if boxes_cxcywh.numel() > 0:
                    cx, cy, w, h = boxes_cxcywh.unbind(-1)
                    gt_centers = torch.stack([cx * orig_w, cy * orig_h], dim=-1)
                else:
                    gt_centers = torch.zeros(0, 2)

                raw_predictions.append(
                    {
                        "pred_labels": pred_labels,
                        "pred_scores": pred_scores,
                        "pred_centers": pred_centers,
                        "gt_labels": lab["class_labels"].cpu(),
                        "gt_centers": gt_centers,
                        "img_diag": img_diag,
                    }
                )

        result = metric.compute()

        # Compute center-distance AP (threshold-free) at 2% for stones
        cd_ap = _compute_cd_ap_inline(raw_predictions, distance_threshold=0.02)
        # Compute top-4 corner recall (matches inference behaviour)
        corner_r4 = _compute_corner_recall_at_k_inline(raw_predictions, k=4)

        # Log all metrics with eval_ prefix for W&B
        eval_metrics = {
            "eval_map": float(result.get("map", 0)),
            "eval_map_50": float(result.get("map_50", 0)),
            "eval_map_75": float(result.get("map_75", 0)),
            "eval_mar_400": float(result.get("mar_400", 0)),
            "eval_corner_R4": corner_r4,
            "eval_stone_cdAP": round(float(np.mean([cd_ap["black_stone"], cd_ap["white_stone"]])), 4),
        }
        if state.log_history:
            state.log_history[-1].update(eval_metrics)
        if metrics is not None:
            metrics.update(eval_metrics)
        if self.trainer is not None:
            self.trainer.log(eval_metrics)

        print(
            f"  mAP@50={eval_metrics['eval_map_50']:.4f}"
            f"  corner_R4={eval_metrics['eval_corner_R4']:.4f}"
            f"  stone_cdAP={eval_metrics['eval_stone_cdAP']:.4f}"
            f"  (mAP={eval_metrics['eval_map']:.4f})"
        )

        # Save best model as W&B artifact when mAP@50 improves
        if self.save_best_artifact and eval_metrics["eval_map_50"] > self.best_map_50:
            self.best_map_50 = eval_metrics["eval_map_50"]
            print(f"  ★ New best mAP@50={self.best_map_50:.4f} — saving W&B artifact...")
            self._save_artifact(model, epoch, eval_metrics)

    def _save_artifact(self, model, epoch: int, map_metrics: dict) -> None:
        """Save model + image processor as a W&B artifact.

        Only keeps the latest version — previous versions are deleted to
        avoid quota bloat.
        """
        import tempfile

        import wandb

        if wandb.run is None:
            return

        artifact_name = f"model-{wandb.run.name}"

        # Delete previous versions before logging a new one
        api = wandb.Api()
        try:
            collection = api.artifact(
                f"{wandb.run.entity}/{wandb.run.project}/{artifact_name}:latest",
            )
            for v in collection.collection.artifacts():
                if v.state == "COMMITTED":
                    v.delete(delete_aliases=True)
        except wandb.errors.CommError:
            pass  # No previous versions exist yet

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            self.image_processor.save_pretrained(tmpdir)

            artifact = wandb.Artifact(
                name=artifact_name,
                type="model",
                metadata={
                    "epoch": epoch,
                    **{k: round(v, 6) for k, v in map_metrics.items()},
                },
            )
            artifact.add_dir(tmpdir)
            wandb.log_artifact(artifact)


class HubPushCallback(TrainerCallback):
    """Push model & processor to HF Hub (with revision) periodically during training.

    Only pushes every ``push_every_n_epochs`` saves to avoid hitting the
    HF Hub rate limit (128 commits/hour).  Defaults to every 10 epochs.
    """

    def __init__(
        self,
        hub_model_id: str,
        revision: str,
        image_processor: RTDetrImageProcessor,
        push_every_n_epochs: int = 50,
    ):
        self.hub_model_id = hub_model_id
        self.revision = revision
        self.image_processor = image_processor
        self.push_every_n_epochs = push_every_n_epochs

    def on_save(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        epoch = int(state.epoch)
        if epoch % self.push_every_n_epochs != 0:
            return
        print(f"Pushing checkpoint (epoch {epoch}) to {self.hub_model_id} (revision={self.revision})...")
        model.push_to_hub(self.hub_model_id, revision=self.revision)
        self.image_processor.push_to_hub(self.hub_model_id, revision=self.revision)


# ---------------------------------------------------------------------------
# Data helpers (inlined from moku.model / moku.dataset)
# ---------------------------------------------------------------------------
def _build_coco_target(example: dict) -> dict:
    return {
        "image_id": example["image_id"],
        "annotations": [
            {
                "id": int(ann_id),
                "category_id": int(cat),
                "bbox": [float(x) for x in bbox],
                "area": float(area),
                "iscrowd": int(iscrowd),
            }
            for ann_id, cat, bbox, area, iscrowd in zip(
                example["objects"]["id"],
                example["objects"]["category"],
                example["objects"]["bbox"],
                example["objects"]["area"],
                example["objects"]["iscrowd"],
            )
        ],
    }


def _unbatch_example(example: dict) -> dict:
    if isinstance(example.get("image"), list):
        return {k: v[0] if isinstance(v, list) else v for k, v in example.items()}
    return example


# ---------------------------------------------------------------------------
# Albumentations augmentation pipeline
# ---------------------------------------------------------------------------
def _build_train_augmentation() -> A.Compose:
    """Aggressive augmentation for small dataset: phone angles, lighting, noise."""
    return A.Compose(
        [
            # ── Geometric (phone angles, partial views) ──
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Perspective(scale=(0.03, 0.12), p=0.6),
            A.Rotate(limit=30, border_mode=0, p=0.5),
            A.RandomResizedCrop(
                size=(640, 640),
                scale=(0.5, 1.0),
                ratio=(0.75, 1.33),
                p=0.5,
            ),
            # ── Photometric (variable lighting) ──
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.08, p=0.6),
            A.RandomGamma(gamma_limit=(70, 130), p=0.3),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.2),
            # ── Blur & noise (mobile camera) ──
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=(3, 9), p=1.0),
                ],
                p=0.4,
            ),
            A.GaussNoise(var_limit=(26.0, 416.0), p=0.3),
            # ── Shadows & occlusion (real lighting) ──
            A.RandomShadow(
                num_shadows_lower=1,
                num_shadows_upper=3,
                shadow_dimension=5,
                shadow_roi=(0, 0, 1, 1),
                p=0.3,
            ),
            # ── Compression artifacts (JPEG from phones) ──
            A.ImageCompression(quality_range=(40, 95), p=0.3),
            A.Downscale(scale_range=(0.5, 0.9), p=0.2),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["category_ids"],
            min_area=1.0,
            min_visibility=0.3,
        ),
    )


def _apply_augmentation(
    image: Image.Image,
    annotations: list[dict],
    aug: A.Compose,
) -> tuple[Image.Image, list[dict]]:
    """Apply albumentations augmentation to image and COCO annotations."""
    img_np = np.array(image)
    h, w = img_np.shape[:2]

    # Clip COCO bboxes to image bounds to avoid albumentations validation errors
    # (source annotations can have coords slightly outside [0, w] / [0, h]).
    bboxes = []
    category_ids = []
    for ann in annotations:
        x, y, bw, bh = ann["bbox"]
        x = max(0.0, min(float(x), w))
        y = max(0.0, min(float(y), h))
        bw = max(0.0, min(float(bw), w - x))
        bh = max(0.0, min(float(bh), h - y))
        if bw > 0 and bh > 0:
            bboxes.append([x, y, bw, bh])
            category_ids.append(ann["category_id"])

    result = aug(image=img_np, bboxes=bboxes, category_ids=category_ids)

    aug_image = Image.fromarray(result["image"])

    aug_annotations = []
    for bbox, cat_id in zip(result["bboxes"], result["category_ids"]):
        x, y, w, h = bbox
        aug_annotations.append(
            {
                "id": 0,
                "category_id": int(cat_id),
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": 0,
            }
        )

    return aug_image, aug_annotations


def make_train_transform(image_processor: RTDetrImageProcessor):
    aug = _build_train_augmentation()

    def transform(example: dict) -> dict:
        example = _unbatch_example(example)
        image = example["image"].convert("RGB")
        target = _build_coco_target(example)
        image, target["annotations"] = _apply_augmentation(image, target["annotations"], aug)
        return image_processor(images=[image], annotations=[target], return_tensors="pt")

    return transform


def make_eval_transform(image_processor: RTDetrImageProcessor):
    def transform(example: dict) -> dict:
        example = _unbatch_example(example)
        image = example["image"].convert("RGB")
        target = _build_coco_target(example)
        return image_processor(images=[image], annotations=[target], return_tensors="pt")

    return transform


def collate_fn(batch: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": [x["labels"] for x in batch],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune RT-DETR on moku dataset (two-stage)")
    p.add_argument(
        "--stage", type=int, choices=[1, 2], default=1, help="Training stage: 1=synthetic/single, 2=real (fine-tune)"
    )
    p.add_argument("--resume-from", type=str, default=None, help="HF model ID or local path to resume from (stage 2)")
    p.add_argument("--resume-revision", type=str, default=None, help="HF Hub revision for --resume-from model")
    p.add_argument(
        "--resume-from-wandb",
        type=str,
        default=None,
        help="W&B artifact path to resume from, e.g. 'model-r10_s1:latest' (stage 2)",
    )
    p.add_argument("--resume", action="store_true", help="Resume interrupted training from last checkpoint")
    p.add_argument("--run-name", default="baseline", help="Name of the training run")
    p.add_argument("--dataset", type=str, default=None, help="HF dataset ID (default: HF_DATASET constant)")
    p.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Dataset config name (default: stage-based for v2, None for v3)",
    )
    p.add_argument("--num-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument(
        "--lr-scheduler",
        default="cosine",
        choices=[
            "cosine",
            "linear",
            "constant",
            "constant_with_warmup",
            "cosine_with_restarts",
            "cosine_with_min_lr",
        ],
    )
    p.add_argument(
        "--lr-scheduler-kwargs",
        type=str,
        default=None,
        help="JSON string of extra scheduler kwargs, e.g. '{\"num_cycles\": 4}'",
    )
    p.add_argument("--max-grad-norm", type=float, default=0.1, help="Max gradient norm for clipping")
    p.add_argument("--use-cpu", action="store_true", help="Force CPU (for local testing)")
    p.add_argument("--push-to-hub", action="store_true", help="Push model to HF Hub after training")
    p.add_argument(
        "--hub-revision",
        type=str,
        default=None,
        help="HF Hub branch to push to (default: stage1 for stage 1, main for stage 2)",
    )
    p.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    p.add_argument("--round", type=str, default=None, help="Round label for W&B grouping (e.g. r4)")
    p.add_argument(
        "--no-save-best-artifact",
        action="store_true",
        help="Disable saving best model as W&B artifact (enabled by default when W&B is active)",
    )
    p.add_argument(
        "--filter-source",
        type=str,
        default=None,
        choices=["real", "generated"],
        help="Filter train set to only include images from this source (v3 only)",
    )
    p.add_argument(
        "--oversample-real",
        type=int,
        default=1,
        help="Repeat real images N times in train set to rebalance real:generated ratio",
    )
    return p.parse_args()


def _download_wandb_artifact(artifact_path: str) -> str:
    """Download a W&B model artifact and return its local directory path.

    Args:
        artifact_path: Artifact name with optional version/alias, e.g.
            ``"model-r10_s1:latest"`` or full path
            ``"hadim/moku/model-r10_s1:v5"``.
    """
    import wandb

    api = wandb.Api()
    entity = os.environ.get("WANDB_ENTITY", "hadim")
    project = os.environ.get("WANDB_PROJECT", "moku")

    if artifact_path.count("/") < 2:
        artifact_path = f"{entity}/{project}/{artifact_path}"

    print(f"Downloading W&B artifact: {artifact_path}")
    artifact = api.artifact(artifact_path, type="model")
    artifact_dir = artifact.download()
    print(f"  → {artifact_dir}")
    return artifact_dir


def _load_dotenv():
    """Load .env file if present (for local dev). No-op on HF Jobs."""
    env_path = os.path.join(os.path.dirname(__file__), os.pardir, ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def main():
    args = parse_args()
    _load_dotenv()

    # --- W&B setup ---
    if not args.no_wandb:
        if not os.environ.get("WANDB_API_KEY"):
            print("ERROR: WANDB_API_KEY environment variable is not set.")
            print("Set it via .env file locally or as an HF secret for HF Jobs.")
            sys.exit(1)
        os.environ.setdefault("WANDB_ENTITY", "hadim")
        os.environ.setdefault("WANDB_PROJECT", "moku")

    # --- Determine dataset, config, and model source ---
    hf_dataset = args.dataset or HF_DATASET

    if args.stage == 1:
        # Stage 1: train from COCO pretrained base model (fresh detection head)
        # For v2: config="synthetic". For v3: no config (all data in default split).
        if args.dataset_config is not None:
            config_name = args.dataset_config or None  # empty string → None
        elif "moku-v2" in hf_dataset:
            config_name = "synthetic"
        else:
            config_name = None
        model_source = BASE_MODEL
        model_revision = None
        hub_revision = args.hub_revision or STAGE1_REVISION
        default_lr = 1e-4
        default_epochs = 30
    else:
        # Stage 2: fine-tune from an existing checkpoint
        if args.dataset_config is not None:
            config_name = args.dataset_config or None
        elif "moku-v2" in hf_dataset:
            config_name = "real"
        else:
            config_name = None
        if args.resume_from_wandb:
            model_source = _download_wandb_artifact(args.resume_from_wandb)
            model_revision = None
        elif args.resume_from:
            model_source = args.resume_from
            model_revision = args.resume_revision
        else:
            model_source = HF_MODEL
            model_revision = STAGE1_REVISION
        hub_revision = args.hub_revision or "main"
        default_lr = 2e-5
        default_epochs = 50

    # Use provided values or stage-specific defaults
    lr = args.lr if args.lr != 1e-4 or args.stage == 1 else default_lr
    num_epochs = args.num_epochs if args.num_epochs != 50 or args.stage == 2 else default_epochs

    print(f"=== Stage {args.stage} Training ===")
    print(f"Dataset: {hf_dataset} (config={config_name})")
    print(f"Model source: {model_source} (revision={model_revision})")
    print(f"Hub target: {HF_MODEL} (revision={hub_revision})")
    print(f"LR: {lr}, Epochs: {num_epochs}")

    # --- Load dataset ---
    print(f"\nLoading dataset: {hf_dataset} (config={config_name})")
    dataset = load_dataset(hf_dataset, config_name)
    print(dataset)

    # --- Filter train set by source ---
    if args.filter_source:
        train_ds = dataset["train"]
        sources = train_ds["source_dataset"]
        if args.filter_source == "generated":
            indices = [i for i, s in enumerate(sources) if s == "generated"]
        else:
            indices = [i for i, s in enumerate(sources) if s != "generated"]
        dataset["train"] = train_ds.select(indices)
        print(f"\nFiltered train set to source={args.filter_source}: {len(dataset['train'])} samples")

    # --- Oversample real images ---
    if args.oversample_real > 1:
        from datasets import concatenate_datasets

        train_ds = dataset["train"]
        sources = train_ds["source_dataset"]
        real_idx = [i for i, s in enumerate(sources) if s != "generated"]
        gen_idx = [i for i, s in enumerate(sources) if s == "generated"]
        real_ds = train_ds.select(real_idx)
        gen_ds = train_ds.select(gen_idx) if gen_idx else None
        parts = [real_ds] * args.oversample_real
        if gen_ds is not None:
            parts.append(gen_ds)
        dataset["train"] = concatenate_datasets(parts)
        print(
            f"\nOversampled real {args.oversample_real}×: {len(real_idx)} real × {args.oversample_real} + {len(gen_idx)} gen = {len(dataset['train'])} samples"
        )

    # --- Load model & image processor ---
    print(f"\nLoading model: {model_source} (revision={model_revision})")
    image_processor = RTDetrImageProcessor.from_pretrained(
        model_source if args.stage == 2 else BASE_MODEL,
        revision=model_revision,
    )

    if args.stage == 1:
        # Stage 1: fresh model from COCO pretrained
        model = RTDetrForObjectDetection.from_pretrained(
            model_source,
            num_labels=NUM_LABELS,
            id2label=ID_TO_CATEGORY,
            label2id=CATEGORIES,
            ignore_mismatched_sizes=True,
        )
    else:
        # Stage 2: load from stage 1 checkpoint (already has correct head)
        model = RTDetrForObjectDetection.from_pretrained(
            model_source,
            revision=model_revision,
            num_labels=NUM_LABELS,
            id2label=ID_TO_CATEGORY,
            label2id=CATEGORIES,
        )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total:,} total, {trainable:,} trainable")

    # --- Apply transforms ---
    dataset["train"].set_transform(make_train_transform(image_processor))
    dataset["validation"].set_transform(make_eval_transform(image_processor))

    # Disable __getitems__ to prevent DataLoader from batching the transform
    # call.  With __getitems__ enabled (PyTorch ≥ 2.4), the DataLoader calls
    # dataset.__getitems__([i, j, ...]) which passes a *batch* of examples to
    # the transform — and _unbatch_example keeps only the first element,
    # silently discarding the rest.  Setting __getitems__ to None makes
    # DataLoader fall back to per-item __getitem__, which is correct.
    for split in ("train", "validation"):
        dataset[split].__getitems__ = None  # type: ignore[assignment]

    # --- Training arguments ---
    report_to = "none" if args.no_wandb else "wandb"

    # Parse lr_scheduler_kwargs if provided
    lr_scheduler_kwargs = None
    if args.lr_scheduler_kwargs:
        import json

        lr_scheduler_kwargs = json.loads(args.lr_scheduler_kwargs)

    training_args = TrainingArguments(
        output_dir=f"runs/{args.run_name}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=lr,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_map_50",
        greater_is_better=True,
        logging_steps=10,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        fp16=False,
        bf16=torch.cuda.is_available() and not args.use_cpu,
        max_grad_norm=args.max_grad_norm,
        use_cpu=args.use_cpu,
        push_to_hub=False,
        report_to=report_to,
        run_name=args.run_name,
    )

    # --- W&B metadata (group + tags) ---
    if report_to == "wandb":
        import wandb

        tags = [f"stage{args.stage}"]
        if args.round:
            tags.append(args.round)
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "moku"),
            entity=os.environ.get("WANDB_ENTITY", "hadim"),
            name=args.run_name,
            group=args.round,
            tags=tags,
            config={
                "stage": args.stage,
                "round": args.round,
                "dataset": hf_dataset,
                "dataset_config": config_name,
                "learning_rate": lr,
                "lr_scheduler": args.lr_scheduler,
                "lr_scheduler_kwargs": lr_scheduler_kwargs,
                "num_epochs": num_epochs,
                "batch_size": args.batch_size,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "max_grad_norm": args.max_grad_norm,
                "model_source": model_source,
                "hub_revision": hub_revision,
            },
            resume="allow",
        )

    map_callback = MAPEvalCallback(
        dataset["validation"],
        image_processor,
        save_best_artifact=(report_to == "wandb" and not args.no_save_best_artifact),
        eval_batch_size=args.eval_batch_size,
    )
    callbacks = [map_callback]
    if args.push_to_hub:
        callbacks.append(HubPushCallback(HF_MODEL, hub_revision, image_processor))

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        callbacks=callbacks,
    )
    map_callback.trainer = trainer

    print("Starting training...")
    if args.resume:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    print("Training complete.")

    # --- Final push to Hub (best model) ---
    if args.push_to_hub:
        print(f"Pushing best model to {HF_MODEL} (revision={hub_revision})...")
        trainer.model.push_to_hub(HF_MODEL, revision=hub_revision)
        image_processor.push_to_hub(HF_MODEL, revision=hub_revision)
        print(f"Model pushed to https://huggingface.co/{HF_MODEL} (branch: {hub_revision})")


if __name__ == "__main__":
    main()
