"""Gradio-based corner annotator for moku.

Interactive tool for correcting board_corner bounding boxes in the
kaya-go/moku-v1 dataset. Displays images with annotation overlays and
lets users click to place, move, or delete corner points.

Usage (from project root):
    pixi run python -m moku.annotator
    pixi run python -m moku.annotator --split train
    pixi run python -m moku.annotator --flagged-only
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import numpy as np
from datasets import DatasetDict, load_dataset
from PIL import Image, ImageDraw, ImageFont

from moku.dataset import CATEGORIES, ID_TO_CATEGORY, audit_corners

# ── Constants ──────────────────────────────────────────────────────────────

CORNER_CAT = CATEGORIES["board_corner"]
CORNER_SIZE = 20  # default bbox side for new corners
MAX_CORNERS = 4

# Colors per category (RGB)
CAT_COLORS = {
    0: (231, 41, 138),  # black_stone — magenta
    1: (27, 158, 119),  # white_stone — teal
    2: (217, 95, 2),  # board_corner — orange
}

CORRECTIONS_FILE = Path("data/annotate/corrected.json")

# ── State ──────────────────────────────────────────────────────────────────


class AnnotatorState:
    """Holds dataset + corrections state for the Gradio app."""

    def __init__(
        self,
        dataset: DatasetDict,
        flagged_only: bool = False,
        split_filter: str | None = None,
    ):
        self.dataset = dataset
        self.corrections: dict[str, dict] = {}
        self._load_corrections()

        # Build index: list of (split, row_idx, image_id, source, flagged)
        flagged_df = audit_corners(dataset, expected_count=4)
        flagged_keys = set(zip(flagged_df["split"], flagged_df["image_id"].astype(str)))

        self.index: list[dict] = []
        for split_name, ds in dataset.items():
            if split_filter and split_name != split_filter:
                continue
            for row_idx in range(len(ds)):
                sample = ds[row_idx]
                iid = str(sample["image_id"])
                source = sample.get("source_dataset", "")
                is_flagged = (split_name, iid) in flagged_keys
                if flagged_only and not is_flagged:
                    continue
                self.index.append(
                    {
                        "split": split_name,
                        "row_idx": row_idx,
                        "image_id": iid,
                        "source": source,
                        "flagged": is_flagged,
                    }
                )

    def _load_corrections(self):
        """Load existing corrections from disk."""
        if CORRECTIONS_FILE.exists():
            with open(CORRECTIONS_FILE) as f:
                self.corrections = json.load(f)

    def _save_corrections(self):
        """Persist corrections to disk."""
        CORRECTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CORRECTIONS_FILE, "w") as f:
            json.dump(self.corrections, f, indent=2)

    @property
    def n_images(self) -> int:
        return len(self.index)

    def unique_key(self, idx: int) -> str:
        """Unique key for an image: split__rowIdx."""
        entry = self.index[idx]
        return f"{entry['split']}__{entry['row_idx']}"

    def get_sample(self, idx: int) -> dict:
        entry = self.index[idx]
        return self.dataset[entry["split"]][entry["row_idx"]]

    def get_corners(self, idx: int) -> list[dict]:
        """Get current corner boxes (corrected if available, else original)."""
        key = self.unique_key(idx)
        if key in self.corrections:
            return [b for b in self.corrections[key]["boxes"] if b["category"] == CORNER_CAT]

        sample = self.get_sample(idx)
        boxes = []
        for ann_id, bbox, cat in zip(
            sample["objects"]["id"],
            sample["objects"]["bbox"],
            sample["objects"]["category"],
        ):
            if cat == CORNER_CAT:
                x, y, w, h = bbox
                boxes.append(
                    {
                        "id": int(ann_id),
                        "x": float(x),
                        "y": float(y),
                        "w": float(w),
                        "h": float(h),
                        "category": CORNER_CAT,
                    }
                )
        return boxes

    def get_all_boxes(self, idx: int) -> list[dict]:
        """Get all annotation boxes for an image."""
        sample = self.get_sample(idx)
        key = self.unique_key(idx)

        # Non-corner boxes always come from original
        boxes = []
        for ann_id, bbox, cat in zip(
            sample["objects"]["id"],
            sample["objects"]["bbox"],
            sample["objects"]["category"],
        ):
            if cat != CORNER_CAT:
                x, y, w, h = bbox
                boxes.append(
                    {
                        "id": int(ann_id),
                        "x": float(x),
                        "y": float(y),
                        "w": float(w),
                        "h": float(h),
                        "category": int(cat),
                    }
                )

        # Corner boxes: corrected if available
        boxes.extend(self.get_corners(idx))
        return boxes

    def save_corners(self, idx: int, corners: list[dict]):
        """Save corrected corner boxes for an image."""
        key = self.unique_key(idx)
        entry = self.index[idx]
        self.corrections[key] = {
            "boxes": corners,
            "image_id": entry["image_id"],
            "split": entry["split"],
            "row_idx": entry["row_idx"],
        }
        self._save_corrections()

    def is_corrected(self, idx: int) -> bool:
        return self.unique_key(idx) in self.corrections


# ── Rendering ──────────────────────────────────────────────────────────────


def render_image(state: AnnotatorState, idx: int) -> Image.Image:
    """Render image with all annotation overlays."""
    sample = state.get_sample(idx)
    img = sample["image"].copy().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    boxes = state.get_all_boxes(idx)

    for box in boxes:
        cat = box["category"]
        color = CAT_COLORS.get(cat, (255, 255, 255))
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        x1, y1, x2, y2 = x, y, x + w, y + h

        if cat == CORNER_CAT:
            # Draw corner as a filled circle + outline
            cx, cy = x + w / 2, y + h / 2
            r = max(w, h) / 2 + 2
            draw.ellipse(
                [cx - r, cy - r, cx + r, cy + r],
                fill=(*color, 80),
                outline=color,
                width=3,
            )
            # Draw crosshair through center
            cr = r + 4
            draw.line([(cx - cr, cy), (cx + cr, cy)], fill=color, width=2)
            draw.line([(cx, cy - cr), (cx, cy + cr)], fill=color, width=2)
        else:
            # Draw stone bbox as rectangle
            draw.rectangle([x1, y1, x2, y2], outline=(*color, 180), width=2)
            label = ID_TO_CATEGORY.get(cat, f"cls{cat}")
            draw.text((x1 + 2, y1 - 12), label, fill=color)

    return img


def render_corners_table(corners: list[dict]) -> list[list]:
    """Build a table of corner positions for display."""
    if not corners:
        return []
    # Label corners by quadrant
    if len(corners) >= 2:
        cx = np.mean([b["x"] + b["w"] / 2 for b in corners])
        cy = np.mean([b["y"] + b["h"] / 2 for b in corners])
    else:
        cx = cy = 0
    rows = []
    for b in corners:
        bcx = b["x"] + b["w"] / 2
        bcy = b["y"] + b["h"] / 2
        if len(corners) >= 2:
            label = ("T" if bcy < cy else "B") + ("L" if bcx < cx else "R")
        else:
            label = "?"
        rows.append([label, f"{bcx:.0f}", f"{bcy:.0f}", f"{b['w']:.0f}×{b['h']:.0f}"])
    return rows


# ── Gradio App ─────────────────────────────────────────────────────────────


def build_app(state: AnnotatorState) -> gr.Blocks:
    """Build and return the Gradio Blocks app."""

    def get_image_choices():
        """Build dropdown choices: idx → display label."""
        choices = []
        for i, entry in enumerate(state.index):
            flags = []
            if entry["flagged"]:
                flags.append("⚠")
            if state.is_corrected(i):
                flags.append("✓")
            label = f"{i + 1}. {entry['split']}/{entry['image_id']} ({entry['source']})"
            if flags:
                label = " ".join(flags) + " " + label
            choices.append((label, i))
        return choices

    def load_view(idx: int):
        """Load image view and info for index idx."""
        idx = int(idx)
        if idx < 0 or idx >= state.n_images:
            return None, "No image", [], ""
        entry = state.index[idx]
        img = render_image(state, idx)
        sample = state.get_sample(idx)

        corners = state.get_corners(idx)
        table = render_corners_table(corners)

        status_parts = [
            f"**Image {idx + 1} / {state.n_images}**",
            f"Split: `{entry['split']}` | ID: `{entry['image_id']}` | Source: `{entry['source']}`",
            f"Size: {sample['width']}×{sample['height']}",
            f"Corners: {len(corners)}/4",
        ]
        if entry["flagged"]:
            status_parts.append("⚠️ **Flagged**")
        if state.is_corrected(idx):
            status_parts.append("✅ **Corrected**")
        status = "  \n".join(status_parts)

        return img, status, table, idx

    def on_image_click(idx: int, evt: gr.SelectData):
        """Handle click on the image to place a corner."""
        idx = int(idx)
        click_x, click_y = evt.index  # pixel coordinates in original image

        corners = state.get_corners(idx)

        # Check if clicking near an existing corner (within 2× bbox size) → remove it
        for c in corners:
            cx = c["x"] + c["w"] / 2
            cy = c["y"] + c["h"] / 2
            dist = ((click_x - cx) ** 2 + (click_y - cy) ** 2) ** 0.5
            if dist < max(c["w"], c["h"]) * 1.5:
                corners.remove(c)
                state.save_corners(idx, corners)
                return load_view(idx)

        # Add new corner if < MAX_CORNERS
        if len(corners) >= MAX_CORNERS:
            # Replace the nearest corner instead
            dists = []
            for c in corners:
                cx = c["x"] + c["w"] / 2
                cy = c["y"] + c["h"] / 2
                dists.append(((click_x - cx) ** 2 + (click_y - cy) ** 2) ** 0.5)
            nearest_idx = int(np.argmin(dists))
            corners[nearest_idx]["x"] = click_x - CORNER_SIZE / 2
            corners[nearest_idx]["y"] = click_y - CORNER_SIZE / 2
        else:
            import time

            corners.append(
                {
                    "id": int(time.time() * 1000),
                    "x": click_x - CORNER_SIZE / 2,
                    "y": click_y - CORNER_SIZE / 2,
                    "w": CORNER_SIZE,
                    "h": CORNER_SIZE,
                    "category": CORNER_CAT,
                }
            )

        state.save_corners(idx, corners)
        return load_view(idx)

    def on_clear_corners(idx: int):
        """Remove all corners for the current image."""
        idx = int(idx)
        state.save_corners(idx, [])
        return load_view(idx)

    def on_reset_corners(idx: int):
        """Reset to original (uncorrected) corners."""
        idx = int(idx)
        key = state.unique_key(idx)
        if key in state.corrections:
            del state.corrections[key]
            state._save_corrections()
        return load_view(idx)

    def on_navigate(idx: int, direction: int):
        idx = int(idx) + direction
        idx = max(0, min(idx, state.n_images - 1))
        return (idx, *load_view(idx))

    def on_select_image(choice):
        idx = int(choice)
        return (idx, *load_view(idx))

    # ── Layout ─────────────────────────────────────────────────────────

    with gr.Blocks(
        title="Moku Corner Annotator",
        theme=gr.themes.Soft(primary_hue="orange", secondary_hue="blue"),
        css="""
        .corner-table { font-size: 14px; }
        .status-box { font-size: 14px; line-height: 1.6; }
        """,
    ) as app:
        # Hidden state for current index
        current_idx = gr.State(value=0)

        gr.Markdown("# 🔲 Moku Corner Annotator")
        gr.Markdown(
            "Click on the image to **place a corner** (up to 4). "
            "Click **near an existing corner** to remove it. "
            "If 4 corners exist, clicking will **move the nearest** one. "
            "All changes are auto-saved."
        )

        with gr.Row():
            # ── Left: navigation ──
            with gr.Column(scale=1, min_width=280):
                image_dropdown = gr.Dropdown(
                    choices=get_image_choices(),
                    label="Image",
                    type="value",
                    interactive=True,
                )
                with gr.Row():
                    prev_btn = gr.Button("← Prev", size="sm")
                    next_btn = gr.Button("Next →", size="sm")

                status_md = gr.Markdown("", elem_classes=["status-box"])

                corner_table = gr.Dataframe(
                    headers=["Pos", "X", "Y", "Size"],
                    label="Corners",
                    interactive=False,
                    elem_classes=["corner-table"],
                )

                with gr.Row():
                    clear_btn = gr.Button("🗑 Clear corners", variant="stop", size="sm")
                    reset_btn = gr.Button("↩ Reset to original", size="sm")

            # ── Right: image ──
            with gr.Column(scale=4):
                image_display = gr.Image(
                    label="Image",
                    type="pil",
                    interactive=False,
                    height=700,
                )

        # ── Events ─────────────────────────────────────────────────────

        load_outputs = [image_display, status_md, corner_table, current_idx]
        nav_outputs = [current_idx, *load_outputs]

        # Dropdown selection
        image_dropdown.change(
            on_select_image,
            inputs=[image_dropdown],
            outputs=nav_outputs,
        )

        # Navigation buttons
        prev_btn.click(
            lambda idx: on_navigate(idx, -1),
            inputs=[current_idx],
            outputs=nav_outputs,
        )
        next_btn.click(
            lambda idx: on_navigate(idx, +1),
            inputs=[current_idx],
            outputs=nav_outputs,
        )

        # Image click → place/remove corner
        image_display.select(
            on_image_click,
            inputs=[current_idx],
            outputs=load_outputs,
        )

        # Clear / Reset buttons
        clear_btn.click(on_clear_corners, inputs=[current_idx], outputs=load_outputs)
        reset_btn.click(on_reset_corners, inputs=[current_idx], outputs=load_outputs)

        # Initial load
        app.load(
            lambda: (0, *load_view(0)),
            outputs=nav_outputs,
        )

    return app


# ── CLI entry point ────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Moku Corner Annotator (Gradio)")
    parser.add_argument("--dataset", default="kaya-go/moku-v1", help="HF dataset name")
    parser.add_argument("--split", default=None, help="Filter to a single split")
    parser.add_argument("--flagged-only", action="store_true", help="Show only flagged images")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset} ...")
    dataset = load_dataset(args.dataset)
    print(f"Building annotator state ...")
    anno_state = AnnotatorState(dataset, flagged_only=args.flagged_only, split_filter=args.split)
    print(f"  {anno_state.n_images} images indexed")

    app = build_app(anno_state)
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
