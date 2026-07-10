from __future__ import annotations

import cv2
import numpy as np

from fretboard_position_mapper import FretboardMappingConfig
from ui_theme import (
    BODY,
    CANVAS,
    CANVAS_RAISED,
    CANVAS_SOFT,
    HAIRLINE,
    HAIRLINE_SOFT,
    INK,
    INK_STRONG,
    MUTE,
    PRIMARY,
    draw_chip,
    draw_dashed_line,
    draw_glow,
    fill_canvas,
)


DEFAULT_DISPLAY_WIDTH = 1280
DEFAULT_DISPLAY_HEIGHT = 720
DEFAULT_TOTAL_FRETS = 12
DEFAULT_BLOCK_SIZE = 4
OPEN_STRING_LABELS = {
    6: "6 E",
    5: "5 A",
    4: "4 D",
    3: "3 G",
    2: "2 B",
    1: "1 E",
}


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


def draw_soft_circle(
    frame,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    overlay = frame.copy()
    cv2.circle(overlay, center, radius, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def draw_background(frame) -> None:
    height, width = frame.shape[:2]
    fill_canvas(frame)
    draw_dashed_line(frame, (48, 128), (width - 48, 128), HAIRLINE_SOFT)
    draw_dashed_line(frame, (48, height - 120), (width - 48, height - 120), HAIRLINE_SOFT)


def draw_panel_shadow(
    frame,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
) -> None:
    left, top = top_left
    right, bottom = bottom_right
    blend_rect(frame, (left + 6, top + 8), (right + 6, bottom + 8), (0, 0, 0), 0.24)


def get_fretboard_display_rect(frame) -> tuple[int, int, int, int]:
    height, width = frame.shape[:2]
    left = max(96, int(width * 0.09))
    right = min(width - 64, int(width * 0.95))
    top = max(172, int(height * 0.29))
    bottom = min(height - 168, int(height * 0.72))
    return left, top, right, bottom


def get_virtual_fret_boundaries(
    config: FretboardMappingConfig,
    fret_boundaries,
) -> tuple[float, ...]:
    if fret_boundaries is not None and len(fret_boundaries) == config.visible_frets + 1:
        return tuple(float(boundary) for boundary in fret_boundaries)

    fret_width = config.virtual_width / config.visible_frets
    return tuple(fret_width * index for index in range(config.visible_frets + 1))


def get_display_fret_numbers(
    total_frets: int = DEFAULT_TOTAL_FRETS,
    start_fret: int = 1,
) -> list[int]:
    return list(range(start_fret, start_fret + total_frets))


def get_current_block_start(
    current_fret: int | None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    display_start_fret: int = 1,
) -> int:
    if current_fret is None:
        current_fret = display_start_fret

    offset = max(0, current_fret - display_start_fret)
    return display_start_fret + (offset // block_size) * block_size


def map_display_point_to_virtual_fretboard(
    x: int,
    y: int,
    frame,
    config: FretboardMappingConfig,
) -> tuple[float, float] | None:
    left, top, right, bottom = get_fretboard_display_rect(frame)

    if x < left or x > right or y < top or y > bottom:
        return None

    virtual_x = (x - left) / (right - left) * config.virtual_width
    virtual_y = (y - top) / (bottom - top) * config.virtual_height
    return virtual_x, virtual_y


def map_display_point_to_fret_position(
    x: int,
    y: int,
    frame,
    config: FretboardMappingConfig,
    total_frets: int = DEFAULT_TOTAL_FRETS,
    display_start_fret: int = 1,
) -> dict | None:
    left, top, right, bottom = get_fretboard_display_rect(frame)

    if x < left or x > right or y < top or y > bottom:
        return None

    fretboard_width = right - left
    fretboard_height = bottom - top
    fret_width = fretboard_width / total_frets
    string_height = fretboard_height / 6
    fret_index = min(max(int((x - left) // fret_width), 0), total_frets - 1)
    string_index = min(max(int((y - top) // string_height), 0), 5)

    return {
        "inside_fretboard": True,
        "string_number": config.string_order_top_to_bottom[string_index],
        "fret_number": display_start_fret + fret_index,
        "display_fret_index": fret_index,
    }


def draw_text(
    frame,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def get_highlight_rect(
    frame,
    config: FretboardMappingConfig,
    fret_boundaries,
    highlight_position: dict,
    total_frets: int = DEFAULT_TOTAL_FRETS,
    display_start_fret: int = 1,
) -> tuple[int, int, int, int] | None:
    string_number = highlight_position.get("string_number")
    fret_number = highlight_position.get("fret_number")

    if string_number not in config.string_order_top_to_bottom or fret_number is None:
        return None

    fret_index = fret_number - display_start_fret

    if fret_index < 0 or fret_index >= total_frets:
        return None

    left, top, right, bottom = get_fretboard_display_rect(frame)
    fretboard_width = right - left
    fretboard_height = bottom - top
    string_index = config.string_order_top_to_bottom.index(string_number)
    fret_width = fretboard_width / total_frets

    cell_left = left + int(fret_index * fret_width)
    cell_right = left + int((fret_index + 1) * fret_width)
    cell_top = top + int(string_index / 6 * fretboard_height)
    cell_bottom = top + int((string_index + 1) / 6 * fretboard_height)
    return cell_left, cell_top, cell_right, cell_bottom


def draw_position_markers(
    frame,
    left: int,
    top: int,
    right: int,
    bottom: int,
    total_frets: int = DEFAULT_TOTAL_FRETS,
    display_start_fret: int = 1,
) -> None:
    fretboard_width = right - left
    marker_frets = {3, 5, 7, 9, 15, 17, 19, 21}
    double_marker_frets = {12, 24}

    for index in range(total_frets):
        fret_number = display_start_fret + index

        if fret_number not in marker_frets and fret_number not in double_marker_frets:
            continue

        cell_left = left + int(index / total_frets * fretboard_width)
        cell_right = left + int((index + 1) / total_frets * fretboard_width)
        center_x = (cell_left + cell_right) // 2

        if fret_number in double_marker_frets:
            marker_ys = (int(top + (bottom - top) * 0.36), int(top + (bottom - top) * 0.64))
        else:
            marker_ys = (int((top + bottom) / 2),)

        for center_y in marker_ys:
            draw_soft_circle(frame, (center_x, center_y), 20, (180, 180, 172), 0.24)
            cv2.circle(frame, (center_x, center_y), 12, (28, 28, 26), -1, cv2.LINE_AA)
            cv2.circle(frame, (center_x - 4, center_y - 4), 4, (100, 100, 94), -1, cv2.LINE_AA)
            cv2.circle(frame, (center_x, center_y), 12, (8, 8, 8), 1, cv2.LINE_AA)


def draw_string(
    frame,
    start: tuple[int, int],
    end: tuple[int, int],
    thickness: int,
    color: tuple[int, int, int],
) -> None:
    shadow_start = (start[0], start[1] + 3)
    shadow_end = (end[0], end[1] + 3)
    cv2.line(frame, shadow_start, shadow_end, (190, 190, 184), thickness + 2, cv2.LINE_AA)
    cv2.line(frame, start, end, color, thickness, cv2.LINE_AA)
    cv2.line(frame, start, end, (84, 84, 80), 1, cv2.LINE_AA)


def draw_fretboard_diagram(
    frame,
    config: FretboardMappingConfig,
    fret_boundaries=None,
    highlight_position: dict | None = None,
    total_frets: int = DEFAULT_TOTAL_FRETS,
    block_size: int = DEFAULT_BLOCK_SIZE,
    current_block_start: int | None = None,
    display_start_fret: int = 1,
    title: str = "FretTone",
    subtitle: str = "virtual fretboard",
) -> None:
    draw_background(frame)
    left, top, right, bottom = get_fretboard_display_rect(frame)
    fretboard_width = right - left
    fretboard_height = bottom - top
    fret_width = fretboard_width / total_frets

    if current_block_start is None:
        current_block_start = get_current_block_start(
            highlight_position.get("fret_number") if highlight_position else config.start_fret,
            block_size=block_size,
            display_start_fret=display_start_fret,
        )

    panel_left = left - 92
    panel_top = top - 52
    panel_right = right + 32
    panel_bottom = bottom + 92
    draw_panel_shadow(frame, (panel_left, panel_top), (panel_right, panel_bottom))
    blend_rect(frame, (panel_left, panel_top), (panel_right, panel_bottom), (240, 241, 236), 0.98)
    cv2.rectangle(
        frame,
        (panel_left, panel_top),
        (panel_right, panel_bottom),
        (112, 112, 106),
        1,
        cv2.LINE_AA,
    )

    cv2.rectangle(frame, (left, top), (right, bottom), (248, 249, 245), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (left, top), (right, top + 34), (237, 241, 232), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (left, bottom - 34), (right, bottom), (232, 235, 228), -1, cv2.LINE_AA)

    wood_overlay = frame.copy()
    for offset in range(-fretboard_height, fretboard_height * 2, 18):
        color = (220, 224 + (offset % 8), 214 + (offset % 12))
        cv2.line(
            wood_overlay,
            (left, top + offset),
            (right, top + offset + 42),
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.addWeighted(wood_overlay, 0.18, frame, 0.82, 0, frame)

    for block_start in range(display_start_fret, display_start_fret + total_frets, block_size):
        block_index = (block_start - display_start_fret) // block_size
        block_left = left + int((block_start - display_start_fret) * fret_width)
        block_right = left + int(
            min(total_frets, block_start - display_start_fret + block_size)
            * fret_width
        )
        block_color = (232, 237, 229) if block_index % 2 == 0 else (246, 246, 241)
        blend_rect(frame, (block_left, top), (block_right, bottom), block_color, 0.35)

        label = f"{block_start}-{block_start + block_size - 1}"
        draw_text(
            frame,
            label,
            (block_left + 10, top - 22),
            0.58,
            (82, 82, 78),
            1,
        )

    current_block_left = left + int((current_block_start - display_start_fret) * fret_width)
    current_block_right = left + int(
        min(total_frets, current_block_start - display_start_fret + block_size)
        * fret_width
    )
    blend_rect(
        frame,
        (current_block_left, top),
        (current_block_right, bottom),
        PRIMARY,
        0.1,
    )
    cv2.rectangle(
        frame,
        (current_block_left, top - 18),
        (current_block_right, bottom + 18),
        PRIMARY,
        3,
        cv2.LINE_AA,
    )
    draw_text(
        frame,
        "CURRENT BLOCK",
        (current_block_left + 12, bottom + 74),
        0.54,
        PRIMARY,
        2,
    )

    if highlight_position is not None:
        highlight_rect = get_highlight_rect(
            frame,
            config,
            fret_boundaries,
            highlight_position,
            total_frets=total_frets,
            display_start_fret=display_start_fret,
        )

        if highlight_rect is not None:
            cell_left, cell_top, cell_right, cell_bottom = highlight_rect
            center = ((cell_left + cell_right) // 2, (cell_top + cell_bottom) // 2)
            blend_rect(frame, (cell_left, cell_top), (cell_right, cell_bottom), PRIMARY, 0.28)
            draw_glow(frame, center, 52, PRIMARY, 0.72)
            cv2.circle(
                frame,
                center,
                24,
                PRIMARY,
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                frame,
                (center[0] - 7, center[1] - 7),
                7,
                (210, 255, 238),
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                frame,
                center,
                34,
                PRIMARY,
                3,
                cv2.LINE_AA,
            )

    draw_position_markers(
        frame,
        left,
        top,
        right,
        bottom,
        total_frets=total_frets,
        display_start_fret=display_start_fret,
    )

    for boundary_index in range(total_frets + 1):
        x = left + int(boundary_index * fret_width)
        is_block_boundary = boundary_index % block_size == 0
        color = (48, 48, 46) if is_block_boundary else (88, 88, 84)
        thickness = 4 if is_block_boundary else 2

        if boundary_index == 0:
            thickness = 6
            color = (24, 24, 23)

        cv2.line(frame, (x + 3, top - 8), (x + 3, bottom + 10), (192, 192, 184), thickness, cv2.LINE_AA)
        cv2.line(frame, (x, top - 10), (x, bottom + 10), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x - 1, top - 8), (x - 1, bottom + 8), (246, 246, 242), 1, cv2.LINE_AA)

        if boundary_index < total_frets:
            next_x = left + int((boundary_index + 1) * fret_width)
            fret_number = display_start_fret + boundary_index
            text_scale = 0.5 if total_frets > 12 else 0.72
            draw_text(
                frame,
                str(fret_number),
                ((x + next_x) // 2 - 8, bottom + 48),
                text_scale,
                (44, 44, 42),
                1 if total_frets > 12 else 2,
            )

    for index, string_number in enumerate(config.string_order_top_to_bottom):
        y = top + int((index + 0.5) / 6 * fretboard_height)
        thickness = 5 if string_number >= 5 else 4 if string_number >= 3 else 3
        color = (12, 12, 12) if string_number >= 4 else (28, 28, 26)
        draw_string(frame, (left, y), (right, y), thickness, color)
        draw_text(
            frame,
            OPEN_STRING_LABELS.get(string_number, str(string_number)),
            (left - 74, y + 9),
            0.72,
            (18, 18, 18),
        )

    cv2.rectangle(frame, (left, top), (right, bottom), (82, 82, 78), 2, cv2.LINE_AA)
    cv2.line(frame, (left, top), (right, top), PRIMARY, 2, cv2.LINE_AA)
    draw_chip(frame, "FRETLOG", (left, 48), accent=PRIMARY)
    draw_text(frame, title, (left + 128, 72), 1.02, INK_STRONG, 2)
    draw_text(frame, subtitle, (left + 128, 108), 0.62, BODY, 1)
