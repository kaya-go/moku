# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "torchvision",
#     "transformers>=5.2.0",
#     "datasets>=4.6.0",
#     "accelerate>=1.12.0",
#     "trackio>=0.15.0",
#     "pycocotools>=2.0.11",
#     "scipy",
#     "huggingface_hub>=1.5.0",
# ]
# ///
"""Fine-tune RT-DETR r18vd on kaya-go/moku-v1 dataset.

Self-contained training script for HF Jobs.

Usage (local CPU test):
    uv run scripts/train.py --num-epochs 2 --use-cpu

Usage (HF Jobs with GPU):
    hf jobs uv run \
        --flavor a10g-small \
        --timeout 3h \
        --secrets HF_TOKEN \
        scripts/train.py
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
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_MODEL = "PekingU/rtdetr_r18vd"
HF_DATASET = "kaya-go/moku-v1"
HF_MODEL = "kaya-go/moku-v1"

CATEGORIES = {"black_stone": 0, "white_stone": 1, "board_corner": 2}
ID_TO_CATEGORY = {v: k for k, v in CATEGORIES.items()}
NUM_LABELS = len(CATEGORIES)


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
    p = argparse.ArgumentParser(description="Fine-tune RT-DETR on moku dataset")
    p.add_argument("--run-name", default="baseline", help="Name of the training run")
    p.add_argument("--num-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--lr-scheduler", default="cosine", choices=["cosine", "linear", "constant"])
    p.add_argument("--use-cpu", action="store_true", help="Force CPU (for local testing)")
    p.add_argument("--push-to-hub", action="store_true", help="Push model to HF Hub after training")
    p.add_argument("--no-trackio", action="store_true", help="Disable Trackio logging")
    return p.parse_args()


def main():
    args = parse_args()

    # --- Load dataset ---
    print(f"Loading dataset: {HF_DATASET}")
    dataset = load_dataset(HF_DATASET)
    print(dataset)

    # --- Load model & image processor ---
    print(f"Loading model: {BASE_MODEL}")
    image_processor = RTDetrImageProcessor.from_pretrained(BASE_MODEL)
    model = RTDetrForObjectDetection.from_pretrained(
        BASE_MODEL,
        num_labels=NUM_LABELS,
        id2label=ID_TO_CATEGORY,
        label2id=CATEGORIES,
        ignore_mismatched_sizes=True,
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total:,} total, {trainable:,} trainable")

    # --- Apply transforms ---
    dataset["train"].set_transform(make_train_transform(image_processor))
    dataset["validation"].set_transform(make_eval_transform(image_processor))
    dataset["test"].set_transform(make_eval_transform(image_processor))

    # --- Training arguments ---
    report_to = "none" if args.no_trackio else "trackio"

    training_args = TrainingArguments(
        project="moku",
        output_dir=f"runs/{args.run_name}",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler,
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
        fp16=torch.cuda.is_available() and not args.use_cpu,
        use_cpu=args.use_cpu,
        push_to_hub=args.push_to_hub,
        hub_model_id=HF_MODEL if args.push_to_hub else None,
        report_to=report_to,
        trackio_space_id="kaya-go/moku-training" if report_to == "trackio" else None,
        run_name=args.run_name,
    )

    # --- Train ---
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    print("Starting training...")
    trainer.train()
    print("Training complete.")

    # --- Push to Hub ---
    if args.push_to_hub:
        print(f"Pushing model to {HF_MODEL}...")
        trainer.model.push_to_hub(HF_MODEL)
        image_processor.push_to_hub(HF_MODEL)
        print(f"Model pushed to https://huggingface.co/{HF_MODEL}")


if __name__ == "__main__":
    main()
