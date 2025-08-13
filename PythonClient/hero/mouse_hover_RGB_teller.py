#!/usr/bin/env python3
"""
Hover to read RGB values (image + separate info panel).
- Image is shown 1:1 (no scaling).
- Panel never overlaps image.
- Panel has a minimum height so elements fit even for small images.
"""

import cv2
import numpy as np

# ---- Set your image path here ----
IMAGE_PATH = "/home/sgarimella34/Pictures/Screenshots/segroi.png"
# ----------------------------------

# Load image (OpenCV loads as BGR)
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError(f"Could not read image at: {IMAGE_PATH}")

img_h, img_w = img.shape[:2]

# UI config
PANEL_W = 300
PANEL_MIN_H = 360  # ensure enough vertical space for text + swatches + bars
PANEL_BG = (24, 24, 24)
TEXT_COLOR = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
THICK = 2
LINE = cv2.LINE_AA

# Canvas size: make sure total height can fit the panel
CANVAS_H = max(img_h, PANEL_MIN_H)
CANVAS_W = img_w + PANEL_W

win_name = "Hover for RGB (press 'q' to quit)"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, min(CANVAS_W + 100, 1800), min(CANVAS_H + 100, 1200))

state = {"x": None, "y": None, "rgb": None}

def on_mouse(event, x, y, flags, param):
    # Only record when cursor is over the image area (exclude panel/padding)
    if event == cv2.EVENT_MOUSEMOVE:
        if 0 <= x < img_w and 0 <= y < img_h:
            b, g, r = img[y, x]
            state["x"], state["y"] = x, y
            state["rgb"] = (int(r), int(g), int(b))
        else:
            state["x"], state["y"], state["rgb"] = None, None, None

cv2.setMouseCallback(win_name, on_mouse)

def clamp(val, lo, hi):
    return max(lo, min(val, hi))

def draw_panel(panel_img, xy, rgb):
    """Draws the sidebar. Never writes outside panel bounds."""
    H, W = panel_img.shape[:2]
    panel_img[:] = PANEL_BG

    # Layout constants
    left = 20
    y_cursor = 40
    gap = 30

    def put(text, y, scale=FONT_SCALE, thick=THICK):
        y = clamp(y, 0, H - 1)
        cv2.putText(panel_img, text, (left, y), FONT, scale, TEXT_COLOR, thick, LINE)
        return y

    # Title
    y_cursor = put("Pixel Inspector", 30, scale=0.8)

    # Space after title
    y_cursor += 20

    if xy is None or rgb is None:
        y_cursor = put("Position:", y_cursor)
        y_cursor += gap
        y_cursor = put("(move over image)", y_cursor)
        y_cursor += gap
        y_cursor = put("RGB:", y_cursor)
        y_cursor += gap
        y_cursor = put("(move over image)", y_cursor)
        y_cursor += gap

        # Empty swatch box (only if space permits)
        sw_h, sw_w = 60, 120
        y_top = clamp(y_cursor, 0, max(0, H - sw_h - 1))
        cv2.rectangle(panel_img, (left, y_top), (left + sw_w, y_top + sw_h), (64, 64, 64), 2)
        return

    x, y = xy
    r, g, b = rgb

    y_cursor = put("Position:", y_cursor)
    y_cursor += gap
    y_cursor = put(f"(x={x}, y={y})", y_cursor)
    y_cursor += gap

    y_cursor = put("RGB:", y_cursor)
    y_cursor += gap
    y_cursor = put(f"({r}, {g}, {b})", y_cursor)
    y_cursor += gap

    # Color swatch of the current pixel (clamped to fit)
    sw_h, sw_w = 60, 120
    y_top = clamp(y_cursor, 0, max(0, H - sw_h))
    panel_img[y_top:y_top + sw_h, left:left + sw_w] = (b, g, r)
    y_cursor = y_top + sw_h + 20

    # Channel bars (draw only if they fit)
    bar_w, bar_h = 200, 14
    def bar(ypos, value, label):
        if ypos + bar_h + 2 > H:
            return ypos  # not enough room; skip
        cv2.putText(panel_img, f"{label}: {value:3d}", (left, ypos - 6), FONT, 0.5, TEXT_COLOR, 1, LINE)
        cv2.rectangle(panel_img, (left, ypos), (left + bar_w, ypos + bar_h), (80, 80, 80), 1)
        filled = int((value / 255.0) * bar_w)
        cv2.rectangle(panel_img, (left, ypos), (left + filled, ypos + bar_h), (200, 200, 200), -1)
        return ypos + bar_h + 16

    y_cursor = bar(y_cursor, r, "R") or y_cursor
    y_cursor = bar(y_cursor, g, "G") or y_cursor
    y_cursor = bar(y_cursor, b, "B") or y_cursor

while True:
    # Frame area: place the image at top-left, pad below if canvas taller
    frame = np.zeros((CANVAS_H, img_w, 3), dtype=np.uint8)
    frame[:img_h, :, :] = img

    # Panel area has fixed width and CANVAS_H height
    panel = np.zeros((CANVAS_H, PANEL_W, 3), dtype=np.uint8)

    if state["rgb"] is not None:
        xy = (state["x"], state["y"])
        rgb = state["rgb"]
    else:
        xy = None
        rgb = None

    draw_panel(panel, xy, rgb)

    # Crosshair on the image region only
    if xy is not None:
        x, y = xy
        cv2.drawMarker(frame, (x, y), color=(0, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
        cv2.drawMarker(frame, (x, y), color=(255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)
        r, g, b = rgb
        cv2.setWindowTitle(win_name, f"{win_name}  |  (x={x}, y={y})  RGB=({r}, {g}, {b})")
    else:
        cv2.setWindowTitle(win_name, win_name)

    canvas = np.hstack([frame, panel])
    cv2.imshow(win_name, canvas)

    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q')):
        break

cv2.destroyAllWindows()
