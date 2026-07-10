import argparse
from pathlib import Path

import cv2
import numpy as np

from fretboard_display import (
    DEFAULT_DISPLAY_HEIGHT,
    DEFAULT_DISPLAY_WIDTH,
    draw_fretboard_diagram,
)
from fretboard_position_mapper import FretboardMappingConfig
from run_fretlog_recorder import draw_overlay as draw_recorder_overlay
from run_sync_playback import draw_playback_overlay
from ui_theme import CANVAS, HAIRLINE, PRIMARY, fill_canvas


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "ui_previews"


class FakeRecorder:
    def __init__(self, enabled: bool, session_dir: Path | None, elapsed_seconds: float) -> None:
        self.enabled = enabled
        self.session_dir = session_dir
        self._elapsed_seconds = elapsed_seconds

    def elapsed_seconds(self) -> float:
        return self._elapsed_seconds


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render static UI preview PNGs without camera or microphone.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=DEFAULT_DISPLAY_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_DISPLAY_HEIGHT)
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Only write PNG files. Do not open the preview window.",
    )
    parser.add_argument(
        "--contact-sheet-window",
        action="store_true",
        help="Open the scaled contact sheet instead of full-size previews.",
    )
    return parser.parse_args()


def make_config() -> FretboardMappingConfig:
    return FretboardMappingConfig(
        virtual_width=800,
        virtual_height=240,
        start_fret=5,
        visible_frets=4,
        string_order_top_to_bottom=(6, 5, 4, 3, 2, 1),
    )


def render_fretboard(width: int, height: int):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    draw_fretboard_diagram(
        frame,
        make_config(),
        highlight_position={
            "inside_fretboard": True,
            "string_number": 3,
            "fret_number": 7,
            "note_name": "D4",
            "midi_pitch": 62,
        },
        current_block_start=5,
        title="FretTone",
        subtitle="12 frets / offline position view",
    )
    return frame


def render_playback(width: int, height: int):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    highlight = {
        "inside_fretboard": True,
        "string_number": 2,
        "fret_number": 5,
        "note_name": "E4",
        "midi_pitch": 64,
        "confidence": 0.86,
        "current_block_start": 5,
    }
    draw_fretboard_diagram(
        frame,
        make_config(),
        highlight_position=highlight,
        current_block_start=5,
        title="FretTone Playback",
        subtitle="recorded sync session",
    )
    draw_playback_overlay(
        frame,
        session_dir=PROJECT_ROOT / "data" / "sync_sessions" / "sync_preview",
        elapsed_seconds=7.35,
        duration_seconds=18.0,
        highlight_position=highlight,
        upcoming_notes=[
            {
                "t_onset": 8.1,
                "string": 3,
                "fret": 7,
                "note_name": "D4",
            },
            {
                "t_onset": 9.4,
                "string": 2,
                "fret": 8,
                "note_name": "G4",
            },
        ],
        playing=True,
    )
    return frame


def render_recorder(width: int, height: int):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    fill_canvas(frame)

    for x in range(0, width, 80):
        cv2.line(frame, (x, 0), (x + height // 3, height), HAIRLINE, 1, cv2.LINE_AA)

    cv2.rectangle(frame, (80, 160), (width - 80, height - 80), (26, 30, 34), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (80, 160), (width - 80, height - 80), HAIRLINE, 1, cv2.LINE_AA)
    cv2.line(frame, (80, 160), (width - 80, height - 80), PRIMARY, 2, cv2.LINE_AA)
    cv2.line(frame, (width - 80, 160), (80, height - 80), PRIMARY, 2, cv2.LINE_AA)

    recorder = FakeRecorder(
        enabled=True,
        session_dir=PROJECT_ROOT / "data" / "sync_sessions" / "sync_preview",
        elapsed_seconds=12.4,
    )
    draw_recorder_overlay(frame, recorder)
    return frame


def write_preview(output_dir: Path, filename: str, frame) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    cv2.imwrite(str(output_path), frame)
    return output_path


def add_preview_label(frame, label: str):
    labeled = frame.copy()
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 48), (15, 15, 15), -1, cv2.LINE_AA)
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 48), HAIRLINE, 1, cv2.LINE_AA)
    cv2.putText(
        labeled,
        label,
        (18, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )
    return labeled


def make_contact_sheet(previews: dict[str, np.ndarray], preview_width: int = 560):
    labeled_frames = []

    for label, frame in previews.items():
        scale = preview_width / frame.shape[1]
        preview_height = int(frame.shape[0] * scale)
        resized = cv2.resize(frame, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
        labeled_frames.append(add_preview_label(resized, label))

    gap = 18
    sheet_width = preview_width
    sheet_height = sum(frame.shape[0] for frame in labeled_frames) + gap * (len(labeled_frames) - 1)
    sheet = np.full((sheet_height, sheet_width, 3), CANVAS, dtype=np.uint8)
    y = 0

    for frame in labeled_frames:
        sheet[y : y + frame.shape[0], 0:sheet_width] = frame
        y += frame.shape[0] + gap

    return sheet


def show_contact_sheet(contact_sheet) -> None:
    window_name = "FretLog UI Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, contact_sheet)
    print("UI preview window opened. Press any key in the window to close.")
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


def draw_window_label(frame, label: str, index: int, count: int):
    labeled = frame.copy()
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 52), (15, 15, 15), -1, cv2.LINE_AA)
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 52), HAIRLINE, 1, cv2.LINE_AA)
    cv2.putText(
        labeled,
        f"{label}  {index + 1}/{count}",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        labeled,
        "n/d: next  p/a: prev  q/esc: close",
        (labeled.shape[1] - 420, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (180, 180, 176),
        1,
        cv2.LINE_AA,
    )
    return labeled


def show_full_size_previews(previews: dict[str, np.ndarray]) -> None:
    window_name = "FretLog UI Preview"
    items = list(previews.items())
    current_index = 0

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print("UI preview window opened.")
    print("n/d: next, p/a: previous, q/esc: close")

    while True:
        label, frame = items[current_index]
        cv2.imshow(
            window_name,
            draw_window_label(frame, label, current_index, len(items)),
        )
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):
            break
        if key in (ord("n"), ord("d"), ord(" ")):
            current_index = (current_index + 1) % len(items)
        elif key in (ord("p"), ord("a")):
            current_index = (current_index - 1) % len(items)

    cv2.destroyWindow(window_name)


def make_ui_previews() -> None:
    args = parse_args()
    previews = {
        "Fretboard": render_fretboard(args.width, args.height),
        "Playback": render_playback(args.width, args.height),
        "Recorder": render_recorder(args.width, args.height),
    }
    filenames = {
        "Fretboard": "fretboard_ui.png",
        "Playback": "playback_ui.png",
        "Recorder": "recorder_ui.png",
    }

    for label, frame in previews.items():
        filename = filenames[label]
        output_path = write_preview(args.output_dir, filename, frame)
        print(output_path)

    contact_sheet = make_contact_sheet(previews)
    contact_sheet_path = write_preview(args.output_dir, "ui_contact_sheet.png", contact_sheet)
    print(contact_sheet_path)

    if not args.no_window:
        if args.contact_sheet_window:
            show_contact_sheet(contact_sheet)
        else:
            show_full_size_previews(previews)


if __name__ == "__main__":
    make_ui_previews()
