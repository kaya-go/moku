"""AI image generation for goban training data.

Provides synthetic → photorealistic style transfer using Gemini.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import random
import re
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

# ---------------------------------------------------------------------------
# Style-transfer prompt building blocks for synthetic → photorealistic
# ---------------------------------------------------------------------------

# (material, weight) — ~50% wood, rest is diverse
BOARD_MATERIALS = [
    ("natural wood with visible grain texture", 3),
    ("light bamboo with fine grain", 2),
    ("dark walnut wood with rich grain", 3),
    ("maple wood with subtle grain", 2),
    ("plastic board with a matte finish", 1),
    ("thick glass board with a smooth surface", 1),
    ("aluminum board with a brushed metal finish", 1),
    ("lacquered wood with a glossy surface", 2),
    ("aged kaya wood with deep golden tones", 3),
]
_BOARD_MATERIAL_NAMES = [m for m, _ in BOARD_MATERIALS]
_BOARD_MATERIAL_WEIGHTS = [w for _, w in BOARD_MATERIALS]

STONE_MATERIALS = [
    "smooth, round, and slightly convex shell and slate stones",
    "polished glass stones with subtle translucency",
    "yunzi stones with a warm jade-like sheen",
    "plastic stones with a matte finish",
    "ceramic stones with a smooth glaze",
]

STYLE_SURFACES = [
    "a wooden table",
    "a tatami mat",
    "a dark tablecloth",
    "a clean desk",
    "a stone countertop",
    "a bamboo mat",
    "a green felt surface",
    "a leather desk pad",
    "a marble surface",
]

STYLE_LIGHTINGS = [
    "Warm natural window light from the side.",
    "Soft diffused overhead light.",
    "Bright even daylight.",
    "Evening warm lamp light casting gentle shadows.",
    "Cool neutral overhead light with no harsh shadows.",
    "Dramatic side lighting with subtle shadows.",
    "Soft golden hour light from a nearby window.",
]

_STYLE_PROMPT_TEMPLATE = """\
Transform this synthetic Go board image into a photorealistic photograph of \
the EXACT same board position.

CRITICAL — you MUST follow ALL of these rules:
1. Every stone must remain on the EXACT SAME grid intersection as in the \
input image. Do NOT move, add, or remove any stone.
2. Preserve the color of every stone exactly (black stays black, white stays white).
3. Keep the same board size ({board_size}×{board_size} grid lines).
4. Keep the EXACT same camera angle, zoom level, and perspective as the \
input. Do NOT tilt, rotate, or change the viewpoint.
5. The board must remain at the EXACT same position and scale within the \
frame as in the input. Do NOT move or resize the board.

Visual style:
- The board is made of {board_material}.
- The stones are {stone_material}.
- The board sits on {surface}.
- {lighting}
- {context}
- Sharp focus, high resolution photograph. No text, watermarks, or overlays.

This is a STYLE TRANSFER task: change ONLY the textures and lighting to look \
photorealistic while keeping the board geometry, camera angle, and stone \
layout PIXEL-PERFECT identical to the input.
"""

_CONTEXT_OPTIONS = [
    "The area around the board is empty table surface, nothing else.",
    "A wooden bowl of captured black stones sits near one edge of the board.",
    "Two stone bowls (one with black, one with white stones) are partially visible at the edges.",
    "A tea cup sits near a corner of the board.",
    "The area around the board shows only the table surface.",
    "A small clock and a bowl of stones are at the edge of the frame.",
    "The area around the board is empty table surface, nothing else.",
    "The area around the board is empty table surface, nothing else.",
]


# ---------------------------------------------------------------------------
# Synthetic → photorealistic style transfer
# ---------------------------------------------------------------------------


def make_style_transfer_prompt(board_size: int) -> str:
    """Build a randomized style-transfer prompt for synthetic → real.

    Parameters
    ----------
    board_size : int
        Board grid size (9, 13, or 19).
    """
    prompt = _STYLE_PROMPT_TEMPLATE.format(
        board_size=board_size,
        board_material=random.choices(_BOARD_MATERIAL_NAMES, weights=_BOARD_MATERIAL_WEIGHTS, k=1)[0],
        stone_material=random.choice(STONE_MATERIALS),
        surface=random.choice(STYLE_SURFACES),
        lighting=random.choice(STYLE_LIGHTINGS),
        context=random.choice(_CONTEXT_OPTIONS),
    )
    return prompt


def synthetic_to_real(
    client: genai.Client,
    synth_image: Image.Image,
    prompt: str,
    *,
    model: str = "gemini-3.1-flash-image-preview",
) -> Image.Image | None:
    """Send a synthetic goban image to Gemini for style transfer.

    Returns a photorealistic PIL Image or None on failure.
    """
    response = client.models.generate_content(
        model=model,
        contents=[synth_image, prompt],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio="1:1",
            ),
        ),
    )
    if response.parts:
        for part in response.parts:
            if part.inline_data is not None:
                raw = part.as_image()
                return Image.open(io.BytesIO(raw.image_bytes))
    return None


async def synthetic_to_real_async(
    client: genai.Client,
    synth_image: Image.Image,
    prompt: str,
    *,
    model: str = "gemini-3.1-flash-image-preview",
) -> Image.Image | None:
    """Async wrapper around synthetic_to_real using a thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: synthetic_to_real(client, synth_image, prompt, model=model),
    )


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------


async def generate_conditioned(
    n: int,
    out_dir: Path,
    model: str,
    prefix: str,
    workers: int,
    image_size: int,
) -> None:
    """Generate ``n`` Gemini images conditioned on synthetic boards (resumable).

    Each sample is a synthetic goban with exact annotations, restyled into a
    photo by Gemini; the annotations carry over since positions are preserved.
    Writes ``<out_dir>/images/<prefix>_NNNN.{png,json}`` and ``<out_dir>/images.json``.
    """
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    from moku.synthetic import generate_synthetic_sample

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set (environment or .env)")

    client = genai.Client(api_key=api_key)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Find already generated files to resume (need both .png and .json)
    existing = set()
    png_pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.png$")
    for p in images_dir.iterdir():
        m = png_pattern.match(p.name)
        if m:
            idx_val = int(m.group(1))
            json_path = images_dir / f"{prefix}_{idx_val:04d}.json"
            if json_path.exists():
                existing.add(idx_val)

    if existing:
        print(f"Resuming — {len(existing)} images already exist, skipping them.")

    # Build list of indices to generate
    indices: list[int] = []
    idx = 0
    while len(indices) < n - len(existing):
        if idx not in existing:
            indices.append(idx)
        idx += 1

    if not indices:
        print(f"All {n} images already exist. Nothing to do.")
        return

    semaphore = asyncio.Semaphore(workers)
    generated = len(existing)
    total_failures = 0

    board_sizes = [9, 13, 19]
    board_weights = [0.15, 0.15, 0.7]

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    async def _worker(target_idx: int) -> bool:
        nonlocal generated, total_failures

        board_size = random.choices(board_sizes, weights=board_weights, k=1)[0]
        max_retries = 3

        for attempt in range(max_retries):
            # Generate synthetic image + annotations
            # ~40% of the time use a larger margin to give Gemini room
            # for context objects around the board
            margin_frac: float | None = None
            if random.random() < 0.4:
                margin_frac = random.uniform(0.05, 0.15)
            synth_image, annotation = generate_synthetic_sample(
                board_size=board_size,
                image_size=image_size,
                margin_frac=margin_frac,
            )

            prompt = make_style_transfer_prompt(board_size)

            async with semaphore:
                try:
                    real_image = await synthetic_to_real_async(
                        client,
                        synth_image,
                        prompt,
                        model=model,
                    )
                except Exception as e:
                    progress.console.print(f"  [red]Error idx={target_idx:04d}: {e}[/red]")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2**attempt)
                        continue
                    total_failures += 1
                    return False

            if real_image is not None:
                filename = f"{prefix}_{target_idx:04d}.png"
                path = images_dir / filename

                # Resize Gemini output to match image_size
                real_image = real_image.resize((image_size, image_size))
                real_image.save(path)

                # Save per-image annotation JSON (atomic per worker)
                objects = annotation["objects"]
                boxes_list = []
                for ann_id, obj in enumerate(
                    zip(
                        objects["id"],
                        objects["bbox"],
                        objects["category"],
                        objects["area"],
                    )
                ):
                    obj_id, bbox, category, area = obj
                    boxes_list.append(
                        {
                            "id": ann_id,
                            "x": round(bbox[0], 2),
                            "y": round(bbox[1], 2),
                            "w": round(bbox[2], 2),
                            "h": round(bbox[3], 2),
                            "category": int(category),
                        }
                    )

                ann_data = {
                    "image": {
                        "id": filename,
                        "filename": filename,
                        "width": image_size,
                        "height": image_size,
                        "source": "synthetic_conditioned",
                        "board_size": board_size,
                    },
                    "boxes": boxes_list,
                }
                json_path = images_dir / f"{prefix}_{target_idx:04d}.json"
                with open(json_path, "w") as jf:
                    json.dump(ann_data, jf, indent=2)

                generated += 1
                progress.update(task, completed=generated)
                return True

            # No image returned — retry
            if attempt < max_retries - 1:
                await asyncio.sleep(1)

        progress.console.print(f"  [yellow]No image for idx={target_idx:04d} after {max_retries} attempts[/yellow]")
        total_failures += 1
        return False

    with progress:
        task = progress.add_task(f"Generating conditioned (×{workers})", total=n, completed=generated)
        tasks = [asyncio.create_task(_worker(i)) for i in indices]
        await asyncio.gather(*tasks)

    # Rebuild images.json from all per-image JSON files
    images_json_path = out_dir / "images.json"
    images_meta: list[dict] = []
    annotations: dict[str, dict] = {}

    for jf in sorted(images_dir.glob(f"{prefix}_*.json")):
        with open(jf) as f:
            ann_data = json.load(f)
        img_meta = ann_data["image"]
        images_meta.append(img_meta)
        annotations[img_meta["filename"]] = {"boxes": ann_data["boxes"]}

    output_data = {"images": images_meta, "annotations": annotations}
    with open(images_json_path, "w") as f:
        json.dump(output_data, f, indent=2)

    n_total_boxes = sum(len(a["boxes"]) for a in annotations.values())
    print(f"Done — {generated}/{n} images in {out_dir} ({total_failures} failures)")
    print(f"Annotations: {n_total_boxes} boxes across {len(annotations)} images")
    print(f"Review them with: moku annotate serve --data-dir {out_dir}")
