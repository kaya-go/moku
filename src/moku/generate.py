"""AI image generation for goban training data.

Provides synthetic → photorealistic style transfer using Gemini.
"""

from __future__ import annotations

import asyncio
import io
import random

from google import genai
from google.genai import types
from PIL import Image

# ---------------------------------------------------------------------------
# Style-transfer prompt building blocks for synthetic → photorealistic
# ---------------------------------------------------------------------------

BOARD_MATERIALS = [
    "natural wood with visible grain texture",
    "light bamboo with fine grain",
    "dark walnut wood with rich grain",
    "maple wood with subtle grain",
    "plastic board with a matte finish",
    "thick glass board with a smooth surface",
    "aluminum board with a brushed metal finish",
    "lacquered wood with a glossy surface",
    "aged kaya wood with deep golden tones",
]

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
        board_material=random.choice(BOARD_MATERIALS),
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
