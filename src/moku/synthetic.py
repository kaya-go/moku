"""Synthetic goban image generator.

Generates realistic Go board images with perfect COCO annotations for training.
Each generated sample includes board corners, black stones, and white stones
placed on valid grid intersections, with optional perspective distortion,
lighting effects, and partial board crops.
"""

from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from moku.dataset import CATEGORIES


def _wood_texture(width: int, height: int, base_color: tuple[int, ...] = (210, 170, 110)) -> np.ndarray:
    """Generate a simple procedural wood-grain texture.

    Uses layered horizontal bands with noise to simulate wood grain.
    Returns an HWC uint8 RGB array.
    """
    img = np.zeros((height, width, 3), dtype=np.float64)

    # Base color
    for c in range(3):
        img[:, :, c] = base_color[c]

    # Horizontal grain bands
    y_coords = np.arange(height).reshape(-1, 1)
    # Multiple frequency bands for grain
    for freq, amp in [(0.03, 15), (0.07, 8), (0.15, 5)]:
        phase = random.uniform(0, 2 * math.pi)
        wave = amp * np.sin(freq * y_coords + phase)
        # Add slight horizontal variation
        x_coords = np.arange(width).reshape(1, -1)
        x_var = 3 * np.sin(0.01 * x_coords + random.uniform(0, math.pi))
        for c in range(3):
            img[:, :, c] += wave + x_var

    # Random noise for texture
    noise = np.random.normal(0, 4, (height, width, 3))
    img += noise

    return np.clip(img, 0, 255).astype(np.uint8)


def _draw_grid(
    draw: ImageDraw.Draw,
    top_left: tuple[float, float],
    cell_size: float,
    board_size: int,
    line_width: int = 1,
    line_color: tuple[int, ...] = (40, 30, 20),
) -> None:
    """Draw Go board grid lines."""
    x0, y0 = top_left
    grid_span = cell_size * (board_size - 1)

    for i in range(board_size):
        offset = i * cell_size
        # Horizontal
        draw.line(
            [(x0, y0 + offset), (x0 + grid_span, y0 + offset)],
            fill=line_color,
            width=line_width,
        )
        # Vertical
        draw.line(
            [(x0 + offset, y0), (x0 + offset, y0 + grid_span)],
            fill=line_color,
            width=line_width,
        )

    # Star points (hoshi)
    hoshi_positions = {
        9: [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)],
        13: [(3, 3), (3, 9), (6, 6), (9, 3), (9, 9)],
        19: [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)],
    }
    dot_r = max(2, int(cell_size * 0.08))
    for row, col in hoshi_positions.get(board_size, []):
        cx = x0 + col * cell_size
        cy = y0 + row * cell_size
        draw.ellipse(
            [(cx - dot_r, cy - dot_r), (cx + dot_r, cy + dot_r)],
            fill=line_color,
        )


def _draw_stone(
    draw: ImageDraw.Draw,
    cx: float,
    cy: float,
    radius: float,
    color: str,
) -> None:
    """Draw a single Go stone with specular highlight and subtle shadow."""
    # Drop shadow (slightly offset, blurred via larger ellipse)
    shadow_offset = radius * 0.08
    shadow_r = radius * 1.02
    draw.ellipse(
        [
            (cx - shadow_r + shadow_offset, cy - shadow_r + shadow_offset),
            (cx + shadow_r + shadow_offset, cy + shadow_r + shadow_offset),
        ],
        fill=(30, 30, 30, 80),
    )

    # Main stone body
    if color == "black":
        fill = (20, 20, 20)
        edge = (10, 10, 10)
    else:
        fill = (240, 240, 240)
        edge = (180, 180, 180)

    draw.ellipse(
        [(cx - radius, cy - radius), (cx + radius, cy + radius)],
        fill=fill,
        outline=edge,
        width=1,
    )

    # Specular highlight (small off-center ellipse)
    hl_r = radius * 0.3
    hl_cx = cx - radius * 0.25
    hl_cy = cy - radius * 0.25
    if color == "black":
        hl_fill = (80, 80, 80)
    else:
        hl_fill = (255, 255, 255)
    draw.ellipse(
        [(hl_cx - hl_r, hl_cy - hl_r), (hl_cx + hl_r, hl_cy + hl_r)],
        fill=hl_fill,
    )


def _perspective_transform(
    image: Image.Image,
    corners: list[tuple[float, float]],
    stone_positions: list[tuple[float, float, str]],
    strength: float = 0.1,
) -> tuple[Image.Image, list[tuple[float, float]], list[tuple[float, float, str]]]:
    """Apply a random perspective distortion to image, corners, and stone positions.

    Uses PIL's ``transform(PERSPECTIVE)`` with random perturbation of the 4
    image corners. Returns the transformed image plus updated corner and stone
    coordinates.
    """
    w, h = image.size

    # Random perturbation for each image corner
    dx = strength * w
    dy = strength * h

    # Source corners (image boundaries)
    src = np.array(
        [[0, 0], [w, 0], [w, h], [0, h]],
        dtype=np.float64,
    )
    # Destination corners (perturbed)
    dst = np.array(
        [
            [random.uniform(0, dx), random.uniform(0, dy)],
            [w - random.uniform(0, dx), random.uniform(0, dy)],
            [w - random.uniform(0, dx), h - random.uniform(0, dy)],
            [random.uniform(0, dx), h - random.uniform(0, dy)],
        ],
        dtype=np.float64,
    )

    # Compute perspective coefficients: dst -> src (to fill output)
    coeffs = _find_perspective_coeffs(dst, src)
    transformed = image.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)

    # Forward mapping: src -> dst to update annotation coordinates
    fwd_coeffs = _find_perspective_coeffs(src, dst)

    def _map_point(px: float, py: float) -> tuple[float, float]:
        a, b, c, d, e, f, g, hh = fwd_coeffs
        denom = g * px + hh * py + 1.0
        nx = (a * px + b * py + c) / denom
        ny = (d * px + e * py + f) / denom
        return nx, ny

    new_corners = [_map_point(cx, cy) for cx, cy in corners]
    new_stones = [(*_map_point(sx, sy), clr) for sx, sy, clr in stone_positions]

    return transformed, new_corners, new_stones


def _find_perspective_coeffs(
    src: np.ndarray,
    dst: np.ndarray,
) -> tuple[float, ...]:
    """Compute the 8 perspective transform coefficients for PIL."""
    matrix = []
    for (x, y), (X, Y) in zip(dst, src):
        matrix.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        matrix.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
    A = np.array(matrix, dtype=np.float64)
    B = np.array([c for pt in src for c in pt], dtype=np.float64)
    res = np.linalg.lstsq(A, B, rcond=None)[0]
    return tuple(res.tolist())


def _add_vignette(image: Image.Image, strength: float = 0.3) -> Image.Image:
    """Apply a subtle vignette (darkened edges) to the image."""
    w, h = image.size
    img_arr = np.array(image, dtype=np.float64)

    # Create radial gradient
    Y, X = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    max_dist = math.sqrt(cx**2 + cy**2)
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) / max_dist
    vignette = 1.0 - strength * dist**2

    for c in range(3):
        img_arr[:, :, c] *= vignette

    return Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))


def generate_synthetic_sample(
    board_size: int = 19,
    image_size: int = 640,
    n_stones: int | None = None,
    perspective_strength: float = 0.08,
    apply_blur: bool = True,
    apply_vignette: bool = True,
) -> tuple[Image.Image, dict]:
    """Generate a single synthetic goban image with COCO annotations.

    Args:
        board_size: Number of lines (9, 13, or 19).
        image_size: Output image size (square).
        n_stones: Number of stones to place. Random if None (5..board_size^2*0.4).
        perspective_strength: Strength of perspective distortion (0 = none).
        apply_blur: Whether to apply slight Gaussian blur for realism.
        apply_vignette: Whether to apply vignette lighting effect.

    Returns:
        (PIL Image, COCO-style annotation dict) where annotation dict has keys:
        ``image_id``, ``width``, ``height``, ``objects`` with
        ``bbox`` (COCO [x,y,w,h]), ``category``, ``area``, ``id``.
    """
    margin_frac = random.uniform(0.08, 0.15)
    margin = int(image_size * margin_frac)
    grid_span = image_size - 2 * margin
    cell_size = grid_span / (board_size - 1)
    stone_radius = cell_size * 0.44

    # 1. Wood texture background
    bg = _wood_texture(image_size, image_size)
    image = Image.fromarray(bg).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # 2. Grid lines
    _draw_grid(
        ImageDraw.Draw(image),
        (margin, margin),
        cell_size,
        board_size,
        line_width=max(1, int(cell_size * 0.04)),
    )

    # 3. Place stones on random intersections
    if n_stones is None:
        max_stones = int(board_size * board_size * 0.4)
        n_stones = random.randint(5, max(6, max_stones))

    all_positions = [(r, c) for r in range(board_size) for c in range(board_size)]
    random.shuffle(all_positions)
    selected = all_positions[:n_stones]

    stone_positions: list[tuple[float, float, str]] = []
    for row, col in selected:
        cx = margin + col * cell_size
        cy = margin + row * cell_size
        color = random.choice(["black", "white"])
        _draw_stone(draw, cx, cy, stone_radius, color)
        stone_positions.append((cx, cy, color))

    # Composite stones onto board
    image = Image.alpha_composite(image, overlay).convert("RGB")

    # 4. Corner positions (exact grid corners)
    corners = [
        (margin, margin),  # TL
        (margin + (board_size - 1) * cell_size, margin),  # TR
        (margin + (board_size - 1) * cell_size, margin + (board_size - 1) * cell_size),  # BR
        (margin, margin + (board_size - 1) * cell_size),  # BL
    ]

    # 5. Perspective distortion
    if perspective_strength > 0:
        image, corners, stone_positions = _perspective_transform(
            image, corners, stone_positions, strength=perspective_strength
        )

    # 6. Post-processing
    if apply_blur:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
    if apply_vignette:
        image = _add_vignette(image, strength=random.uniform(0.1, 0.3))

    # 7. Build COCO annotations
    corner_bbox_size = max(10, cell_size * 0.4)
    ann_id = 0
    objects: dict[str, list] = {"id": [], "bbox": [], "category": [], "area": [], "iscrowd": []}

    # Corners
    for cx, cy in corners:
        half = corner_bbox_size / 2
        x = max(0, cx - half)
        y = max(0, cy - half)
        w = min(corner_bbox_size, image_size - x)
        h = min(corner_bbox_size, image_size - y)
        objects["id"].append(ann_id)
        objects["bbox"].append([float(x), float(y), float(w), float(h)])
        objects["category"].append(CATEGORIES["board_corner"])
        objects["area"].append(float(w * h))
        objects["iscrowd"].append(0)
        ann_id += 1

    # Stones
    for sx, sy, color in stone_positions:
        cat_name = "black_stone" if color == "black" else "white_stone"
        half = stone_radius
        x = max(0, sx - half)
        y = max(0, sy - half)
        w = min(2 * half, image_size - x)
        h = min(2 * half, image_size - y)
        objects["id"].append(ann_id)
        objects["bbox"].append([float(x), float(y), float(w), float(h)])
        objects["category"].append(CATEGORIES[cat_name])
        objects["area"].append(float(w * h))
        objects["iscrowd"].append(0)
        ann_id += 1

    annotation = {
        "image_id": 0,
        "width": image_size,
        "height": image_size,
        "objects": objects,
    }

    return image, annotation
