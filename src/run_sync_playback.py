import argparse
import json
import time
import wave
from pathlib import Path

import cv2
import numpy as np

from fretboard_display import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_DISPLAY_HEIGHT,
    DEFAULT_DISPLAY_WIDTH,
    DEFAULT_TOTAL_FRETS,
    draw_fretboard_diagram,
    get_current_block_start,
)
from fretboard_position_mapper import FretboardMappingConfig
from ui_theme import (
    BODY,
    CANVAS_RAISED,
    HAIRLINE,
    INK,
    MUTE,
    PRIMARY,
    WARNING,
    draw_chip,
    draw_hairline_panel,
    draw_label,
    draw_progress_bar,
    draw_text,
    fit_text,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYNC_OUTPUT_DIR = PROJECT_ROOT / "data" / "sync_sessions"
WINDOW_NAME = "FretTone - Sync Playback"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay a saved sync session with audio and fretboard highlights.",
    )
    parser.add_argument(
        "session_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Path to data/sync_sessions/sync_YYYYMMDD_HHMMSS.",
    )
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--sync-root", type=Path, default=DEFAULT_SYNC_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=DEFAULT_DISPLAY_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_DISPLAY_HEIGHT)
    parser.add_argument("--total-frets", type=int, default=DEFAULT_TOTAL_FRETS)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--no-audio", action="store_true")
    return parser.parse_args()


def find_latest_session(sync_root: Path) -> Path:
    candidates = sorted(
        [path for path in sync_root.glob("sync_*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(f"No sync sessions found in {sync_root}")

    return candidates[0]


def load_events(session_dir: Path) -> list[dict]:
    events_path = session_dir / "events.json"

    if not events_path.exists():
        return []

    with events_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_notes(session_dir: Path) -> list[dict]:
    notes_path = session_dir / "notes.json"

    if not notes_path.exists():
        return []

    with notes_path.open("r", encoding="utf-8") as file:
        raw_notes = json.load(file)

    return raw_notes.get("notes", [])


def load_audio(audio_path: Path):
    if not audio_path.exists():
        return None

    with wave.open(str(audio_path), "rb") as file:
        sample_rate = file.getframerate()
        channels = file.getnchannels()
        sample_width = file.getsampwidth()
        frame_count = file.getnframes()
        raw_audio = file.readframes(frame_count)

    if sample_width != 2:
        raise ValueError("Only 16-bit PCM audio.wav is supported")

    audio = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0

    if channels > 1:
        audio = audio.reshape(-1, channels)

    duration = frame_count / sample_rate if sample_rate > 0 else 0.0
    return {
        "audio": audio,
        "sample_rate": sample_rate,
        "duration": duration,
    }


def start_audio_playback(audio_data, no_audio: bool, offset_seconds: float = 0.0) -> bool:
    if no_audio or audio_data is None:
        return False

    try:
        import sounddevice as sd

        sample_offset = int(offset_seconds * audio_data["sample_rate"])
        sd.play(audio_data["audio"][sample_offset:], audio_data["sample_rate"])
        return True
    except Exception as error:
        print(f"audio playback disabled: {error}")
        return False


def stop_audio_playback() -> None:
    try:
        import sounddevice as sd

        sd.stop()
    except Exception:
        pass


def get_highlight_at_time(events: list[dict], elapsed_seconds: float):
    highlight = None

    for event in events:
        if event.get("time_seconds", 0.0) > elapsed_seconds:
            break

        if event.get("event_type") == "fretboard_highlight":
            highlight = {
                "inside_fretboard": True,
                "string_number": event.get("string_number"),
                "fret_number": event.get("fret_number"),
                "midi_pitch": event.get("midi_pitch"),
                "note_name": event.get("note_name"),
                "pitch_source": event.get("pitch_source"),
                "match_status": event.get("match_status"),
                "current_block_start": event.get("current_block_start"),
            }

    return highlight


def get_note_highlight_at_time(notes: list[dict], elapsed_seconds: float):
    for note in notes:
        if note.get("t_onset", 0.0) <= elapsed_seconds <= note.get("t_offset", 0.0):
            string_number = note.get("string")
            fret_number = note.get("fret")

            if string_number is None or fret_number is None:
                return None

            return {
                "inside_fretboard": True,
                "string_number": string_number,
                "fret_number": fret_number,
                "midi_pitch": note.get("midi_pitch"),
                "note_name": note.get("note_name"),
                "pitch_source": "offline",
                "match_status": note.get("status"),
                "current_block_start": get_current_block_start(
                    fret_number,
                    block_size=DEFAULT_BLOCK_SIZE,
                ),
                "confidence": note.get("confidence"),
            }

    return None


def get_upcoming_notes(notes: list[dict], elapsed_seconds: float, limit: int = 2) -> list[dict]:
    upcoming = []

    for note in notes:
        if note.get("t_onset", 0.0) <= elapsed_seconds:
            continue

        if note.get("string") is None or note.get("fret") is None:
            continue

        upcoming.append(note)

        if len(upcoming) >= limit:
            break

    return upcoming


def format_note_preview(note: dict, elapsed_seconds: float) -> str:
    wait_seconds = max(0.0, float(note.get("t_onset", 0.0)) - elapsed_seconds)
    note_name = note.get("note_name") or "-"
    return (
        f'{note.get("string")}s/{note.get("fret")}F '
        f"{note_name}  in {wait_seconds:.1f}s"
    )


def get_session_duration(events: list[dict], audio_data) -> float:
    event_duration = max((event.get("time_seconds", 0.0) for event in events), default=0.0)

    if audio_data is None:
        return event_duration

    return max(event_duration, audio_data["duration"])


def get_notes_duration(notes: list[dict]) -> float:
    return max((note.get("t_offset", 0.0) for note in notes), default=0.0)


def draw_playback_overlay(
    frame,
    session_dir: Path,
    elapsed_seconds: float,
    duration_seconds: float,
    highlight_position: dict | None,
    upcoming_notes: list[dict],
    playing: bool,
) -> None:
    height, width = frame.shape[:2]
    progress = 0.0 if duration_seconds <= 0 else min(1.0, elapsed_seconds / duration_seconds)
    panel_left = 72
    panel_right = width - 72
    panel_top = height - 112
    panel_bottom = height - 28
    bar_left = panel_left + 166
    bar_width = max(160, panel_right - bar_left - 24)
    bar_y = panel_top + 37
    status = "PLAY" if playing else "PAUSE"

    draw_hairline_panel(
        frame,
        (panel_left, panel_top),
        (panel_right, panel_bottom),
        fill=CANVAS_RAISED,
        border=HAIRLINE,
    )
    draw_label(frame, "SESSION PLAYBACK", (panel_left + 18, panel_top + 24), MUTE)
    draw_chip(frame, status, (panel_left + 18, panel_top + 39), accent=PRIMARY if playing else WARNING)
    draw_text(
        frame,
        f"{elapsed_seconds:05.2f}s / {duration_seconds:05.2f}s",
        (bar_left, panel_top + 24),
        0.58,
        INK,
        2,
    )
    draw_progress_bar(frame, (bar_left, bar_y), bar_width, progress, height=10)
    draw_text(
        frame,
        "space: play/pause  r: restart  q: quit",
        (bar_left, panel_top + 68),
        0.48,
        BODY,
        1,
    )
    draw_text(
        frame,
        fit_text(str(session_dir), 92),
        (bar_left + 360, panel_top + 68),
        0.42,
        MUTE,
        1,
    )

    if highlight_position is None and not upcoming_notes:
        return

    note_panel_right = width - 72
    note_panel_left = max(72, note_panel_right - 610)
    draw_hairline_panel(
        frame,
        (note_panel_left, 48),
        (note_panel_right, 158),
        fill=CANVAS_RAISED,
    )

    if highlight_position is not None:
        note_name = highlight_position.get("note_name") or "-"
        midi_pitch = highlight_position.get("midi_pitch")
        confidence = highlight_position.get("confidence")
        pitch_text = f'{highlight_position["string_number"]} string / {highlight_position["fret_number"]} fret'

        if midi_pitch is not None:
            pitch_text += f"  {note_name} MIDI {midi_pitch}"

        if confidence is not None:
            pitch_text += f"  conf {confidence:.2f}"

        draw_label(frame, "CURRENT NOTE", (note_panel_left + 20, 74), MUTE)
        draw_text(
            frame,
            pitch_text,
            (note_panel_left + 20, 100),
            0.6,
            PRIMARY,
            2,
        )
    else:
        draw_label(frame, "CURRENT NOTE", (note_panel_left + 20, 74), MUTE)
        draw_text(frame, "-", (note_panel_left + 20, 100), 0.6, BODY, 2)

    if upcoming_notes:
        draw_label(frame, "NEXT", (note_panel_left + 20, 126), MUTE)

    for index, note in enumerate(upcoming_notes[:2]):
        label = "NEXT" if index == 0 else "NEXT+1"
        y = 148 if index == 0 else 148

        if index == 0:
            x = note_panel_left + 96
        else:
            x = note_panel_left + 342

        draw_text(
            frame,
            f"{label} {format_note_preview(note, elapsed_seconds)}",
            (x, y),
            0.42,
            BODY,
            1,
        )


def run_sync_playback() -> None:
    args = parse_args()
    session_dir = args.session_dir

    if args.latest or session_dir is None:
        session_dir = find_latest_session(args.sync_root)

    events = load_events(session_dir)
    notes = load_notes(session_dir)
    audio_data = load_audio(session_dir / "audio.wav")
    duration_seconds = max(get_session_duration(events, audio_data), get_notes_duration(notes))
    config = FretboardMappingConfig(
        virtual_width=800,
        virtual_height=240,
        start_fret=1,
        visible_frets=args.block_size,
        string_order_top_to_bottom=(6, 5, 4, 3, 2, 1),
    )
    frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    playing = True
    playback_started_at = time.monotonic()
    paused_at_seconds = 0.0
    audio_started = start_audio_playback(audio_data, args.no_audio)

    cv2.namedWindow(WINDOW_NAME)
    print(f"playing sync session: {session_dir}")

    try:
        while True:
            elapsed_seconds = (
                time.monotonic() - playback_started_at
                if playing
                else paused_at_seconds
            )
            elapsed_seconds = min(elapsed_seconds, duration_seconds)
            highlight_position = (
                get_note_highlight_at_time(notes, elapsed_seconds)
                or get_highlight_at_time(events, elapsed_seconds)
            )
            upcoming_notes = get_upcoming_notes(notes, elapsed_seconds, limit=2)

            if highlight_position is not None:
                config.start_fret = get_current_block_start(
                    highlight_position.get("fret_number"),
                    block_size=args.block_size,
                )

            current_block_start = (
                highlight_position.get("current_block_start")
                if highlight_position is not None
                and highlight_position.get("current_block_start") is not None
                else config.start_fret
            )

            draw_fretboard_diagram(
                frame,
                config,
                highlight_position=highlight_position,
                total_frets=args.total_frets,
                block_size=args.block_size,
                current_block_start=current_block_start,
                title="FretTone Playback",
                subtitle="recorded sync session",
            )
            draw_playback_overlay(
                frame,
                session_dir,
                elapsed_seconds,
                duration_seconds,
                highlight_position,
                upcoming_notes,
                playing,
            )
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(16) & 0xFF

            if key == ord(" "):
                playing = not playing

                if playing:
                    playback_started_at = time.monotonic() - paused_at_seconds
                    audio_started = start_audio_playback(
                        audio_data,
                        args.no_audio,
                        offset_seconds=paused_at_seconds,
                    )
                else:
                    paused_at_seconds = elapsed_seconds
                    stop_audio_playback()
            elif key == ord("r"):
                stop_audio_playback()
                playing = True
                paused_at_seconds = 0.0
                playback_started_at = time.monotonic()
                audio_started = start_audio_playback(audio_data, args.no_audio)
            elif key == ord("q"):
                break

            if playing and elapsed_seconds >= duration_seconds:
                playing = False
                paused_at_seconds = duration_seconds
                if audio_started:
                    stop_audio_playback()
    finally:
        stop_audio_playback()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_sync_playback()
