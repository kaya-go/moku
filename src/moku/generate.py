"""AI image generation for goban training data."""

from __future__ import annotations

import io
import random
import time
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

# ---------------------------------------------------------------------------
# Prompt building blocks
# ---------------------------------------------------------------------------

BOARD_SIZES = [
    ("19×19", 0.7),
    ("13×13", 0.15),
    ("9×9", 0.15),
]

ANGLES = [
    "a straight top-down bird's-eye view, looking directly down at the board",
    "a nearly overhead view, tilted about 5 degrees from vertical",
    "a very slight angle, about 10 degrees from vertical",
    "a gentle overhead angle, about 15 degrees from vertical",
    "a mild top-down angle, about 20 degrees from vertical",
]

LIGHTINGS = [
    "Warm natural window light from the side",
    "Soft diffused overhead light",
    "Bright even daylight",
    "Evening warm lamp light casting gentle shadows",
    "Cool neutral overhead light with no harsh shadows",
]

SURFACES = [
    "a wooden table",
    "a tatami mat",
    "a dark tablecloth",
    "a clean desk",
    "a stone countertop",
    "a bamboo mat",
    "a green felt surface",
]

GAME_STATES = {
    "19×19": [
        "in the early opening with about 15-25 stones placed",
        "in mid-game with about 60-100 stones on the board",
        "in a complex mid-game with around 100-150 stones",
        "near the end with about 180-250 stones covering most intersections",
        "with a few scattered groups of about 30-50 stones",
    ],
    "13×13": [
        "in the early opening with about 10-15 stones placed",
        "in mid-game with about 30-50 stones on the board",
        "near the end with about 60-100 stones on the board",
        "with a few scattered groups of about 20-30 stones",
    ],
    "9×9": [
        "in the early opening with about 5-10 stones placed",
        "in mid-game with about 15-30 stones on the board",
        "near the end with about 40-60 stones on the board",
        "with a few scattered groups of about 10-20 stones",
    ],
}


def _pick_board_size() -> str:
    """Weighted random choice of board size."""
    sizes, weights = zip(*BOARD_SIZES)
    return random.choices(sizes, weights=weights, k=1)[0]


def make_prompt(board_size: str | None = None) -> tuple[str, str]:
    """Build a randomized prompt for a goban photo.

    Returns (prompt, board_size) tuple.
    """
    if board_size is None:
        board_size = _pick_board_size()

    angle = random.choice(ANGLES)
    lighting = random.choice(LIGHTINGS)
    surface = random.choice(SURFACES)
    game_state = random.choice(GAME_STATES[board_size])

    prompt = (
        f"A photorealistic photograph of a complete {board_size} Go board (goban) "
        f"seen from above. The board is centered in the frame with visible margin "
        f"on all sides — all four corners of the board are fully visible and not "
        f"cropped. The entire board edge is within the photo boundaries. "
        f"The game is {game_state}. "
        f"Every single stone is placed exactly on a grid intersection, perfectly "
        f"centered on the crossing point of two lines — no stone is ever placed "
        f"between lines or off-grid. "
        f"The camera is positioned at {angle}. "
        f"{lighting}. The board sits on {surface}. "
        f"The wooden board has natural grain texture with clearly engraved grid "
        f"lines. The stones are perfectly round, smooth, and slightly convex. "
        f"Sharp focus, high resolution photo. No text, watermarks, or overlays."
    )
    return prompt, board_size


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------


def generate_image(
    client: genai.Client,
    prompt: str,
    *,
    model: str = "gemini-3.1-flash-image-preview",
) -> Image.Image | None:
    """Generate a single goban image.

    Supports both Imagen (``imagen-*``) and Gemini (``gemini-*``) models.
    Returns a PIL Image or None on failure.
    """
    if model.startswith("imagen"):
        return _generate_imagen(client, prompt, model)
    return _generate_gemini(client, prompt, model)


def _generate_imagen(
    client: genai.Client,
    prompt: str,
    model: str,
) -> Image.Image | None:
    response = client.models.generate_images(
        model=model,
        prompt=prompt,
        config=types.GenerateImagesConfig(
            number_of_images=1,
            aspect_ratio="1:1",
        ),
    )
    if response.generated_images:
        data = response.generated_images[0].image.image_bytes
        return Image.open(io.BytesIO(data))
    return None


def _generate_gemini(
    client: genai.Client,
    prompt: str,
    model: str,
) -> Image.Image | None:
    response = client.models.generate_content(
        model=model,
        contents=[prompt],
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


def generate_batch(
    client: genai.Client,
    n: int,
    out_dir: Path,
    *,
    model: str = "gemini-3.1-flash-image-preview",
    prefix: str = "gen",
    delay: float = 2.0,
    start_index: int = 0,
) -> list[Path]:
    """Generate *n* goban images and save them to *out_dir*.

    Returns the list of saved file paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for i in range(n):
        idx = start_index + i
        prompt, board_size = make_prompt()
        print(f"[{model}] {i + 1}/{n} ({board_size}) ...")
        img = generate_image(client, prompt, model=model)
        if img is not None:
            path = out_dir / f"{prefix}_{idx:04d}.png"
            img.save(path)
            print(f"  Saved {path.name} ({img.size})")
            saved.append(path)
        else:
            print("  No image returned, skipping")
        if i < n - 1:
            time.sleep(delay)

    print(f"Done — {len(saved)}/{n} images saved")
    return saved
