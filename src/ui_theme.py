from __future__ import annotations

import cv2
import numpy as np


CANVAS = (16, 16, 16)
CANVAS_SOFT = (26, 26, 26)
CANVAS_RAISED = (22, 22, 22)
HAIRLINE = (57, 58, 61)
HAIRLINE_SOFT = (82, 84, 88)
PRIMARY = (146, 217, 0)
PRIMARY_SOFT = (161, 214, 47)
PRIMARY_DEEP = (129, 185, 16)
INK = (242, 242, 242)
INK_STRONG = (255, 255, 255)
BODY = (189, 189, 189)
MUTE = (158, 148, 139)
WARNING = (82, 190, 255)
RECORD = (70, 92, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def fill_canvas(frame) -> None:
    frame[:, :] = CANVAS
    height, width = frame.shape[:2]

    for x in range(0, width, 48):
        cv2.line(frame, (x, 0), (x, height), (20, 20, 20), 1, cv2.LINE_AA)

    for y in range(0, height, 48):
        cv2.line(frame, (0, y), (width, y), (20, 20, 20), 1, cv2.LINE_AA)


def blend_rect(
    frame,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def draw_glow(
    frame,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    strength: float = 0.6,
) -> None:
    """Draw a soft radial glow that remains visible on light backgrounds."""
    blur = radius * 2 + 1

    if blur % 2 == 0:
        blur += 1

    height, width = frame.shape[:2]
    pad = radius + blur
    x0 = max(center[0] - pad, 0)
    y0 = max(center[1] - pad, 0)
    x1 = min(center[0] + pad, width)
    y1 = min(center[1] + pad, height)

    if x1 <= x0 or y1 <= y0:
        return

    roi = frame[y0:y1, x0:x1]
    mask = np.zeros(roi.shape[:2], np.float32)
    cv2.circle(
        mask,
        (center[0] - x0, center[1] - y0),
        radius,
        strength,
        -1,
        cv2.LINE_AA,
    )
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)[..., None]
    color_arr = np.array(color, np.float32)
    roi[:] = (
        roi.astype(np.float32) * (1.0 - mask) + color_arr * mask
    ).astype(np.uint8)


def draw_hairline_panel(
    frame,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    fill: tuple[int, int, int] = CANVAS_RAISED,
    border: tuple[int, int, int] = HAIRLINE,
    alpha: float = 0.94,
) -> None:
    blend_rect(frame, top_left, bottom_right, fill, alpha)
    cv2.rectangle(frame, top_left, bottom_right, border, 1, cv2.LINE_AA)


def draw_text(
    frame,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = INK,
    thickness: int = 1,
) -> None:
    cv2.putText(frame, text, origin, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_label(
    frame,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = MUTE,
) -> None:
    draw_text(frame, text.upper(), origin, 0.44, color, 1)


def draw_chip(
    frame,
    text: str,
    top_left: tuple[int, int],
    accent: tuple[int, int, int] = PRIMARY,
    fill: tuple[int, int, int] = CANVAS_SOFT,
) -> tuple[int, int, int, int]:
    text_size, _ = cv2.getTextSize(text, FONT, 0.5, 1)
    x, y = top_left
    width = text_size[0] + 24
    height = 28
    right = x + width
    bottom = y + height

    cv2.rectangle(frame, (x, y), (right, bottom), fill, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y), (right, bottom), HAIRLINE, 1, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y), (x + 4, bottom), accent, -1, cv2.LINE_AA)
    draw_text(frame, text, (x + 12, y + 19), 0.5, INK, 1)
    return x, y, right, bottom


def draw_progress_bar(
    frame,
    top_left: tuple[int, int],
    width: int,
    progress: float,
    height: int = 10,
    accent: tuple[int, int, int] = PRIMARY,
) -> None:
    x, y = top_left
    progress = max(0.0, min(1.0, progress))
    cv2.rectangle(frame, (x, y), (x + width, y + height), CANVAS_SOFT, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y), (x + width, y + height), HAIRLINE, 1, cv2.LINE_AA)
    filled_width = int(width * progress)

    if filled_width > 0:
        cv2.rectangle(
            frame,
            (x, y),
            (x + filled_width, y + height),
            accent,
            -1,
            cv2.LINE_AA,
        )


def draw_dashed_line(
    frame,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int] = HAIRLINE_SOFT,
    dash_length: int = 12,
    gap_length: int = 8,
) -> None:
    x1, y1 = start
    x2, y2 = end
    length = int(np.hypot(x2 - x1, y2 - y1))

    if length <= 0:
        return

    for offset in range(0, length, dash_length + gap_length):
        dash_start = offset / length
        dash_end = min(length, offset + dash_length) / length
        point_a = (
            int(x1 + (x2 - x1) * dash_start),
            int(y1 + (y2 - y1) * dash_start),
        )
        point_b = (
            int(x1 + (x2 - x1) * dash_end),
            int(y1 + (y2 - y1) * dash_end),
        )
        cv2.line(frame, point_a, point_b, color, 1, cv2.LINE_AA)


def fit_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text

    if max_chars <= 3:
        return text[:max_chars]

    return "..." + text[-(max_chars - 3) :]
