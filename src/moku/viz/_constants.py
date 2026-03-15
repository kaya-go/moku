"""Shared constants for visualization."""

# High-contrast colors for bounding boxes
CATEGORY_COLORS = {
    0: "#e7298a",  # black_stone — magenta/pink
    1: "#1b9e77",  # white_stone — teal/green
    2: "#e6ab02",  # board_corner — yellow/gold
}

CATEGORY_LINEWIDTHS = {
    0: 2,
    1: 2,
    2: 2,
}

# Standard star point (hoshi) positions, 0-indexed
HOSHI_POINTS: dict[int, list[tuple[int, int]]] = {
    9: [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)],
    13: [(3, 3), (3, 6), (3, 9), (6, 3), (6, 6), (6, 9), (9, 3), (9, 6), (9, 9)],
    19: [
        (3, 3),
        (3, 9),
        (3, 15),
        (9, 3),
        (9, 9),
        (9, 15),
        (15, 3),
        (15, 9),
        (15, 15),
    ],
}
