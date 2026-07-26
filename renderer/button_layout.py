"""Shared UI button layout calculations for the bottom control panel.

Both the renderer (scene.py) and the input handler (main.py) import this
module so that button positions are computed in exactly one place — no drift.
"""

from typing import List, Tuple

# Button dimensions and layout constants
BTN_W = 130.0
BTN_H = 38.0
BTN_GAP = 10.0      # horizontal gap between buttons
BTN_MARGIN = 15.0    # margin from screen edges
NUM_BUTTONS = 5      # Style, X-Rot, Y-Rot, Z-Rot, Reset


def get_button_rects(
    screen_w: float, screen_h: float
) -> List[Tuple[float, float, float, float]]:
    """Return a list of (x, y, w, h) tuples for each bottom-panel button.

    Buttons are laid out right-to-left from the screen edge:
        index 0 = Style toggle  (leftmost)
        index 1 = X-Rot 90
        index 2 = Y-Rot 90
        index 3 = Z-Rot 90
        index 4 = Reset         (rightmost)

    Args:
        screen_w: Framebuffer width in pixels.
        screen_h: Framebuffer height in pixels.

    Returns:
        List of 5 tuples, each (x, y, width, height).
    """
    btn_y = screen_h - BTN_H - BTN_MARGIN
    rects: List[Tuple[float, float, float, float]] = []
    for i in range(NUM_BUTTONS):
        # rightmost button (index 4) sits at the right edge;
        # each preceding button is offset further left
        idx_from_right = (NUM_BUTTONS - 1) - i
        bx = screen_w - BTN_MARGIN - (idx_from_right + 1) * BTN_W - idx_from_right * BTN_GAP
        rects.append((bx, btn_y, BTN_W, BTN_H))
    return rects


def hit_test(
    mx: float, my: float, rect: Tuple[float, float, float, float]
) -> bool:
    """Return True if (mx, my) is inside the given (x, y, w, h) rectangle."""
    x, y, w, h = rect
    return x <= mx <= x + w and y <= my <= y + h
