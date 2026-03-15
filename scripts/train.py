# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "torch",
#     "torchvision",
#     "transformers>=5.2.0",
#     "datasets>=4.6.0",
#     "accelerate>=1.12.0",
#     "trackio>=0.15.0",
#     "pycocotools>=2.0.11",
#     "torchmetrics>=1.7.0",
#     "faster-coco-eval>=1.7.0",
#     "scipy",
#     "huggingface_hub>=1.5.0",
# ]
# ///
"""Fine-tune RT-DETR r18vd on kaya-go/moku-v2 dataset.

Self-contained training script for HF Jobs. Supports two-stage training:
  - Stage 1: Pre-train on synthetic data (--stage 1)
  - Stage 2: Fine-tune on real data from a stage 1 checkpoint (--stage 2 --resume-from <model>)

Usage (local CPU test, stage 1):
    uv run scripts/train.py --stage 1 --num-epochs 2 --use-cpu

Usage (HF Jobs, stage 1):
    hf jobs uv run --detach --flavor a10g-small --timeout 3h --secrets HF_TOKEN \\
        scripts/train.py --stage 1 --run-name v2_stage1 --num-epochs 30 --push-to-hub

Usage (HF Jobs, stage 2 from stage 1 branch):
    hf jobs uv run --detach --flavor a10g-small --timeout 3h --secrets HF_TOKEN \\
        scripts/train.py --stage 2 --run-name v2_stage2_lr2e-5 --lr 2e-5 --num-epochs 50

Resume interrupted training:
    uv run scripts/train.py --stage 1 --run-name v2_stage1 --resume
"""

from __future__ import annotations

import argparse
import random

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
HF_DATASET = "kaya-go/moku-v2"
HF_MODEL = "kaya-go/moku-v2"
STAGE1_REVISION = "stage1"

CATEGORIES = {"black_stone": 0, "white_stone": 1, "board_corner": 2}
ID_TO_CATEGORY = {v: k for k, v in CATEGORIES.items()}
NUM_LABELS = len(CATEGORIES)


class MAPEvalCallback(TrainerCallback):
    """Compute COCO mAP on eval set after each evaluation and log to Trackio."""

    def __init__(self, eval_dataset, image_processor: RTDetrImageProcessor, threshold: float = 0.01):
        self.eval_dataset = eval_dataset
        self.image_processor = image_processor
        self.threshold = threshold
        self.trainer: Trainer | None = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch) + 1 if state.epoch is not None else 1
        total = int(args.num_train_epochs)
        print(f"\n{'=' * 60}")
        print(f"  Epoch {epoch}/{total} — Training")
        print(f"{'=' * 60}")

    def on_evaluate(self, args, state, control, model=None, **kwargs):
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

        dataloader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=8, collate_fn=collate_fn, shuffle=False)

        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"]

            with torch.no_grad():
                outputs = model(pixel_values=pixel_values)

            orig_sizes = torch.stack([lab["orig_size"] for lab in labels]).to(device)
            results = self.image_processor.post_process_object_detection(
                outputs, target_sizes=orig_sizes, threshold=self.threshold
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

        result = metric.compute()

        # Log mAP metrics so they appear in Trackio
        map_metrics = {
            "eval_map": float(result.get("map", 0)),
            "eval_map_50": float(result.get("map_50", 0)),
            "eval_map_75": float(result.get("map_75", 0)),
            "eval_mar_400": float(result.get("mar_400", 0)),
        }
        if state.log_history:
            state.log_history[-1].update(map_metrics)
        if self.trainer is not None:
            self.trainer.log(map_metrics)

        print(
            f"  mAP@50:95={map_metrics['eval_map']:.4f}  mAP@50={map_metrics['eval_map_50']:.4f}  mAP@75={map_metrics['eval_map_75']:.4f}  mAR@400={map_metrics['eval_mar_400']:.4f}"
        )


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
# Data helpers (inlined from moku.training / moku.dataset)
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


def _flip_horizontal(image: Image.Image, annotations: list[dict]) -> tuple[Image.Image, list[dict]]:
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    w = image.width
    for ann in annotations:
        x, y, bw, bh = ann["bbox"]
        ann["bbox"] = [w - x - bw, y, bw, bh]
    return image, annotations


def make_train_transform(image_processor: RTDetrImageProcessor, flip_p: float = 0.5):
    def transform(example: dict) -> dict:
        example = _unbatch_example(example)
        image = example["image"].convert("RGB")
        target = _build_coco_target(example)
        if random.random() < flip_p:
            image, target["annotations"] = _flip_horizontal(image, target["annotations"])
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
    p.add_argument("--stage", type=int, choices=[1, 2], default=1, help="Training stage: 1=synthetic, 2=real")
    p.add_argument("--resume-from", type=str, default=None, help="HF model ID or local path to resume from (stage 2)")
    p.add_argument("--resume", action="store_true", help="Resume interrupted training from last checkpoint")
    p.add_argument("--run-name", default="baseline", help="Name of the training run")
    p.add_argument("--num-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=8)
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
    p.add_argument("--no-trackio", action="store_true", help="Disable Trackio logging")
    return p.parse_args()


def main():
    args = parse_args()

    # --- Determine dataset config and model source ---
    if args.stage == 1:
        config_name = "synthetic"
        model_source = BASE_MODEL
        model_revision = None
        hub_revision = args.hub_revision or STAGE1_REVISION
        default_lr = 1e-4
        default_epochs = 30
    else:
        config_name = "real"
        if args.resume_from:
            model_source = args.resume_from
            model_revision = None
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
    print(f"Dataset config: {config_name}")
    print(f"Model source: {model_source} (revision={model_revision})")
    print(f"Hub target: {HF_MODEL} (revision={hub_revision})")
    print(f"LR: {lr}, Epochs: {num_epochs}")

    # --- Load dataset ---
    print(f"\nLoading dataset: {HF_DATASET} ({config_name})")
    dataset = load_dataset(HF_DATASET, config_name)
    print(dataset)

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

    # --- Training arguments ---
    report_to = "none" if args.no_trackio else "trackio"

    # Parse lr_scheduler_kwargs if provided
    lr_scheduler_kwargs = None
    if args.lr_scheduler_kwargs:
        import json

        lr_scheduler_kwargs = json.loads(args.lr_scheduler_kwargs)

    training_args = TrainingArguments(
        project="moku",
        output_dir=f"runs/{args.run_name}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=lr,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        fp16=False,
        bf16=torch.cuda.is_available() and not args.use_cpu,
        max_grad_norm=args.max_grad_norm,
        use_cpu=args.use_cpu,
        push_to_hub=False,
        report_to=report_to,
        trackio_space_id="kaya-go/moku-experiments" if report_to == "trackio" else None,
        run_name=args.run_name,
    )

    map_callback = MAPEvalCallback(dataset["validation"], image_processor)
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
