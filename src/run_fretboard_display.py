import argparse

import cv2
import numpy as np

from fretboard_display import (
    DEFAULT_DISPLAY_HEIGHT,
    DEFAULT_DISPLAY_WIDTH,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_TOTAL_FRETS,
    draw_fretboard_diagram,
    get_current_block_start,
    map_display_point_to_fret_position,
)
from fretboard_position_mapper import FretboardMappingConfig


WINDOW_NAME = "FretTone - Fretboard Display"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Display the virtual guitar fretboard without opening the camera.",
    )
    parser.add_argument("--start-fret", type=int, default=5)
    parser.add_argument("--visible-frets", type=int, default=4)
    parser.add_argument("--total-frets", type=int, default=DEFAULT_TOTAL_FRETS)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--width", type=int, default=DEFAULT_DISPLAY_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_DISPLAY_HEIGHT)
    return parser.parse_args()


def run_fretboard_display() -> None:
    args = parse_args()
    config = FretboardMappingConfig(
        virtual_width=800,
        virtual_height=240,
        start_fret=args.start_fret,
        visible_frets=args.visible_frets,
        string_order_top_to_bottom=(6, 5, 4, 3, 2, 1),
    )
    highlight_position = None
    frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)

    def redraw() -> None:
        draw_fretboard_diagram(
            frame,
            config,
            highlight_position=highlight_position,
            total_frets=args.total_frets,
            block_size=args.block_size,
            current_block_start=get_current_block_start(
                highlight_position["fret_number"] if highlight_position else config.start_fret,
                block_size=args.block_size,
            ),
            title="FretTone",
            subtitle="12 frets / 3 color blocks",
        )
        cv2.putText(
            frame,
            "click: light fret  a/d: current block  q: quit",
            (24, args.height - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(WINDOW_NAME, frame)

    def handle_mouse_click(event, x, y, flags, param):
        nonlocal highlight_position

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        highlight_position = map_display_point_to_fret_position(
            x,
            y,
            frame,
            config,
            total_frets=args.total_frets,
        )

        if highlight_position is not None and highlight_position["inside_fretboard"]:
            config.start_fret = get_current_block_start(
                highlight_position["fret_number"],
                block_size=args.block_size,
            )
            print(
                f'{highlight_position["string_number"]} string '
                f'{highlight_position["fret_number"]} fret'
            )

        redraw()

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, handle_mouse_click)
    redraw()
    print("fretboard display started")
    print("click a fret to light it up")

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord("a"):
            config.start_fret = max(1, config.start_fret - args.block_size)
            highlight_position = None
            redraw()
        elif key == ord("d"):
            config.start_fret = min(
                args.total_frets - args.block_size + 1,
                config.start_fret + args.block_size,
            )
            highlight_position = None
            redraw()
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_fretboard_display()
