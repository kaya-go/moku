"""Synthetic goban image generator.

Generates realistic Go board images with perfect COCO annotations for training.
Each generated sample includes board corners, black stones, and white stones
placed on valid grid intersections, with 3D perspective distortion,
diverse backgrounds, and lighting effects.
"""

from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from moku.dataset import CATEGORIES

# ---------------------------------------------------------------------------
# Background palettes
# ---------------------------------------------------------------------------

# (base_color, grain_scale, grid_line_color)
_WOOD_PALETTES = [
    ((210, 170, 110), 1.0, (40, 30, 20)),  # Classic kaya
    ((220, 200, 140), 0.8, (60, 50, 30)),  # Bamboo
    ((230, 190, 130), 0.9, (50, 35, 25)),  # Maple
    ((130, 90, 55), 1.2, (30, 20, 10)),  # Walnut
    ((180, 110, 70), 1.1, (40, 25, 15)),  # Cherry
    ((235, 215, 170), 0.7, (50, 40, 30)),  # Birch
    ((110, 55, 35), 1.3, (25, 15, 10)),  # Rosewood
]

# (base_color, grid_line_color)
_SOLID_PALETTES = [
    ((50, 120, 60), (20, 50, 20)),  # Green felt
    ((30, 80, 40), (15, 40, 15)),  # Dark green
    ((140, 140, 145), (50, 50, 55)),  # Gray stone
    ((200, 185, 155), (70, 60, 45)),  # Beige cloth
    ((180, 140, 95), (60, 45, 30)),  # Tan leather
]


def _wood_texture(
    width: int,
    height: int,
    base_color: tuple[int, ...] = (210, 170, 110),
    grain_scale: float = 1.0,
) -> np.ndarray:
    """Generate a procedural wood-grain texture. Returns HWC uint8 RGB."""
    img = np.zeros((height, width, 3), dtype=np.float64)
    for c in range(3):
        img[:, :, c] = base_color[c]

    y_coords = np.arange(height).reshape(-1, 1)
    for freq, amp in [(0.03, 15), (0.07, 8), (0.15, 5)]:
        phase = random.uniform(0, 2 * math.pi)
        wave = amp * grain_scale * np.sin(freq * y_coords + phase)
        x_coords = np.arange(width).reshape(1, -1)
        x_var = 3 * np.sin(0.01 * x_coords + random.uniform(0, math.pi))
        for c in range(3):
            img[:, :, c] += wave + x_var

    noise = np.random.normal(0, 4, (height, width, 3))
    img += noise
    return np.clip(img, 0, 255).astype(np.uint8)


def _solid_texture(width: int, height: int, base_color: tuple[int, ...]) -> np.ndarray:
    """Generate a solid-color surface with subtle noise. Returns HWC uint8 RGB."""
    img = np.zeros((height, width, 3), dtype=np.float64)
    for c in range(3):
        img[:, :, c] = base_color[c]
    noise = np.random.normal(0, 6, (height, width, 3))
    img += noise
    y_coords = np.arange(height).reshape(-1, 1)
    x_coords = np.arange(width).reshape(1, -1)
    for c in range(3):
        img[:, :, c] += 5 * np.sin(0.005 * y_coords + random.uniform(0, 2 * math.pi))
        img[:, :, c] += 3 * np.sin(0.008 * x_coords + random.uniform(0, 2 * math.pi))
    return np.clip(img, 0, 255).astype(np.uint8)


def _generate_background(width: int, height: int) -> tuple[np.ndarray, tuple[int, ...]]:
    """Generate a random background texture. Returns (HWC uint8 RGB, grid line color)."""
    if random.random() < 0.75:
        base_color, grain_scale, line_color = random.choice(_WOOD_PALETTES)
        shift = tuple(random.randint(-15, 15) for _ in range(3))
        base_color = tuple(max(0, min(255, b + s)) for b, s in zip(base_color, shift))
        bg = _wood_texture(width, height, base_color, grain_scale)
    else:
        base_color, line_color = random.choice(_SOLID_PALETTES)
        shift = tuple(random.randint(-10, 10) for _ in range(3))
        base_color = tuple(max(0, min(255, b + s)) for b, s in zip(base_color, shift))
        bg = _solid_texture(width, height, base_color)
    return bg, line_color


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
    light_angle: float = -0.7,
) -> None:
    """Draw a single Go stone with specular highlight and subtle shadow."""
    shadow_offset = radius * 0.08
    shadow_r = radius * 1.02
    draw.ellipse(
        [
            (cx - shadow_r + shadow_offset, cy - shadow_r + shadow_offset),
            (cx + shadow_r + shadow_offset, cy + shadow_r + shadow_offset),
        ],
        fill=(30, 30, 30, 80),
    )

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

    # Specular highlight positioned by light direction
    hl_r = radius * 0.3
    hl_cx = cx + radius * 0.25 * math.cos(light_angle)
    hl_cy = cy + radius * 0.25 * math.sin(light_angle)
    hl_fill = (80, 80, 80) if color == "black" else (255, 255, 255)
    draw.ellipse(
        [(hl_cx - hl_r, hl_cy - hl_r), (hl_cx + hl_r, hl_cy + hl_r)],
        fill=hl_fill,
    )


def _perspective_transform_3d(
    image: Image.Image,
    corners: list[tuple[float, float]],
    stone_positions: list[tuple[float, float, str]],
    pitch_deg: float = 0,
    yaw_deg: float = 0,
    roll_deg: float = 0,
    focal_ratio: float = 1.5,
) -> tuple[Image.Image, list[tuple[float, float]], list[tuple[float, float, str]]]:
    """Apply 3D rotation-based perspective distortion.

    Simulates a camera viewing the board from an angle defined by pitch
    (tilt forward/back), yaw (left/right), and roll (in-plane rotation).
    Uses a pinhole camera model for realistic projective distortion.
    """
    w, h = image.size
    cx, cy = w / 2.0, h / 2.0
    f = focal_ratio * max(w, h)

    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    roll = math.radians(roll_deg)

    cp, sp = math.cos(pitch), math.sin(pitch)
    cyaw, syaw = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)

    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Ry = np.array([[cyaw, 0, syaw], [0, 1, 0], [-syaw, 0, cyaw]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx

    # Homography induced by rotating the image plane: H = K @ R @ K^-1
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    K_inv = np.array([[1 / f, 0, -cx / f], [0, 1 / f, -cy / f], [0, 0, 1]])
    H = K @ R @ K_inv

    # PIL perspective coefficients (inverse mapping: output -> input)
    H_inv = np.linalg.inv(H)
    H_inv /= H_inv[2, 2]
    coeffs = (
        H_inv[0, 0],
        H_inv[0, 1],
        H_inv[0, 2],
        H_inv[1, 0],
        H_inv[1, 1],
        H_inv[1, 2],
        H_inv[2, 0],
        H_inv[2, 1],
    )
    transformed = image.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)

    # Forward mapping for annotations
    def _map_point(px: float, py: float) -> tuple[float, float]:
        pt = H @ np.array([px, py, 1.0])
        return float(pt[0] / pt[2]), float(pt[1] / pt[2])

    new_corners = [_map_point(x, y) for x, y in corners]
    new_stones = [(*_map_point(x, y), clr) for x, y, clr in stone_positions]

    return transformed, new_corners, new_stones


# ---------------------------------------------------------------------------
# Lighting effects
# ---------------------------------------------------------------------------


def _add_vignette(image: Image.Image, strength: float = 0.3) -> Image.Image:
    """Apply a subtle vignette (darkened edges)."""
    w, h = image.size
    img_arr = np.array(image, dtype=np.float64)
    Y, X = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    max_dist = math.sqrt(cx**2 + cy**2)
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) / max_dist
    vignette = 1.0 - strength * dist**2
    for c in range(3):
        img_arr[:, :, c] *= vignette
    return Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))


def _adjust_brightness_contrast(
    image: Image.Image,
    brightness: float = 0.0,
    contrast: float = 1.0,
) -> Image.Image:
    """Adjust brightness (additive) and contrast (multiplicative around mean)."""
    img_arr = np.array(image, dtype=np.float64)
    mean = img_arr.mean()
    img_arr = (img_arr - mean) * contrast + mean + brightness
    return Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))


def _color_temperature(image: Image.Image, temp: float = 0.0) -> Image.Image:
    """Shift color temperature. Positive = warm (more red), negative = cool (more blue)."""
    img_arr = np.array(image, dtype=np.float64)
    img_arr[:, :, 0] += temp * 15  # Red channel
    img_arr[:, :, 2] -= temp * 15  # Blue channel
    return Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))


def _directional_light(
    image: Image.Image,
    angle_deg: float = 0.0,
    strength: float = 0.15,
) -> Image.Image:
    """Apply a directional lighting gradient across the image."""
    w, h = image.size
    img_arr = np.array(image, dtype=np.float64)
    angle = math.radians(angle_deg)
    Y, X = np.mgrid[0:h, 0:w]
    Xn = (X - w / 2) / (w / 2)
    Yn = (Y - h / 2) / (h / 2)
    gradient = math.cos(angle) * Xn + math.sin(angle) * Yn
    multiplier = 1.0 + strength * gradient
    for c in range(3):
        img_arr[:, :, c] *= multiplier
    return Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))


def _apply_lighting(image: Image.Image) -> Image.Image:
    """Apply random lighting effects to simulate diverse real-world conditions."""
    # Brightness / contrast jitter
    image = _adjust_brightness_contrast(
        image,
        brightness=random.uniform(-25, 25),
        contrast=random.uniform(0.85, 1.15),
    )
    # Color temperature shift (70% chance)
    if random.random() < 0.7:
        image = _color_temperature(image, temp=random.uniform(-1.0, 1.0))
    # Directional lighting (50% chance)
    if random.random() < 0.5:
        image = _directional_light(
            image,
            angle_deg=random.uniform(0, 360),
            strength=random.uniform(0.05, 0.2),
        )
    # Vignette (60% chance)
    if random.random() < 0.6:
        image = _add_vignette(image, strength=random.uniform(0.1, 0.35))
    return image


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------


def generate_synthetic_sample(
    board_size: int = 19,
    image_size: int = 640,
    n_stones: int | None = None,
    perspective_strength: float = 0.08,
    apply_blur: bool = True,
    apply_vignette: bool = True,
) -> tuple[Image.Image, dict]:
    """Generate a single synthetic goban image with COCO annotations.

    Uses diverse backgrounds, 3D perspective transforms, and lighting effects.

    Args:
        board_size: Number of lines (9, 13, or 19).
        image_size: Output image size (square).
        n_stones: Number of stones to place. Random if None.
        perspective_strength: Controls 3D rotation angle range
            (0 = none, 0.08 = subtle, 0.15 = moderate).
        apply_blur: Whether to apply slight Gaussian blur for realism.
        apply_vignette: Ignored (lighting is applied automatically).

    Returns:
        (PIL Image, COCO-style annotation dict)
    """
    margin_frac = random.uniform(0.08, 0.15)
    margin = int(image_size * margin_frac)
    grid_span = image_size - 2 * margin
    cell_size = grid_span / (board_size - 1)
    stone_radius = cell_size * 0.44

    # 1. Diverse background
    bg, line_color = _generate_background(image_size, image_size)
    image = Image.fromarray(bg).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Random light direction for stone highlights
    light_angle = random.uniform(-math.pi, math.pi)

    # 2. Grid lines (using background-matched line color)
    _draw_grid(
        ImageDraw.Draw(image),
        (margin, margin),
        cell_size,
        board_size,
        line_width=max(1, int(cell_size * 0.04)),
        line_color=line_color,
    )

    # 3. Place stones on random intersections
    if n_stones is None:
        max_stones = int(board_size * board_size * 0.4)
        n_stones = random.randint(5, max(6, max_stones))

    all_positions = [(r, c) for r in range(board_size) for c in range(board_size)]
    random.shuffle(all_positions)
    selected = all_positions[:n_stones]

    # Max jitter: up to 15% of cell_size, but never enough for overlap
    max_jitter = cell_size * 0.15
    stone_positions: list[tuple[float, float, str]] = []
    occupied: set[tuple[int, int]] = set()
    for row, col in selected:
        cx = margin + col * cell_size + random.gauss(0, max_jitter * 0.5)
        cy = margin + row * cell_size + random.gauss(0, max_jitter * 0.5)
        # Clamp so stone stays within image
        cx = max(stone_radius, min(image_size - stone_radius, cx))
        cy = max(stone_radius, min(image_size - stone_radius, cy))
        color = random.choice(["black", "white"])
        _draw_stone(draw, cx, cy, stone_radius, color, light_angle=light_angle)
        stone_positions.append((cx, cy, color))
        occupied.add((row, col))

    # Composite stones onto board
    image = Image.alpha_composite(image, overlay).convert("RGB")

    # 4. Corner positions (exact grid corners)
    corners = [
        (margin, margin),
        (margin + (board_size - 1) * cell_size, margin),
        (margin + (board_size - 1) * cell_size, margin + (board_size - 1) * cell_size),
        (margin, margin + (board_size - 1) * cell_size),
    ]

    # 5. 3D perspective distortion (retry with reduced angles if corners leave bounds)
    if perspective_strength > 0:
        angle_range = perspective_strength * 70
        corner_bbox_half = max(10, cell_size * 0.4) / 2
        for attempt in range(10):
            scale = 1.0 / (1.0 + 0.3 * attempt)  # progressively reduce
            pitch = random.uniform(-angle_range * scale, angle_range * scale)
            yaw = random.uniform(-angle_range * scale, angle_range * scale)
            roll = random.uniform(-angle_range * 0.3 * scale, angle_range * 0.3 * scale)
            new_img, new_corners, new_stones = _perspective_transform_3d(
                image,
                corners,
                stone_positions,
                pitch_deg=pitch,
                yaw_deg=yaw,
                roll_deg=roll,
            )
            # Ensure all 4 corners (with their bbox) stay inside the image
            if all(
                corner_bbox_half <= cx <= image_size - corner_bbox_half
                and corner_bbox_half <= cy <= image_size - corner_bbox_half
                for cx, cy in new_corners
            ):
                image, corners, stone_positions = new_img, new_corners, new_stones
                break

    # 6. Post-processing
    if apply_blur:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
    image = _apply_lighting(image)

    # 7. Build COCO annotations
    corner_bbox_size = max(10, cell_size * 0.4)
    ann_id = 0
    objects: dict[str, list] = {"id": [], "bbox": [], "category": [], "area": [], "iscrowd": []}

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
