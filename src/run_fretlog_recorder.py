import argparse
import json
import sys
import time
from pathlib import Path
from threading import Event

import cv2

from audio_pitch_detector import AudioPitchTracker
from sync_session import SyncGPIOController, SyncSessionConfig, SyncSessionRecorder
from ui_theme import (
    BODY,
    CANVAS_RAISED,
    HAIRLINE,
    INK,
    MUTE,
    PRIMARY,
    RECORD,
    draw_chip,
    draw_hairline_panel,
    draw_label,
    draw_text,
    fit_text,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYNC_OUTPUT_DIR = PROJECT_ROOT / "data" / "sync_sessions"
WINDOW_NAME = "FretLog Recorder"
CANCEL_EXIT_CODE = 20


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record a FretLog camera/audio session for offline analysis.",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--flip",
        choices=("none", "horizontal", "vertical", "both"),
        default="none",
    )
    parser.add_argument("--sync-output-dir", type=Path, default=DEFAULT_SYNC_OUTPUT_DIR)
    parser.add_argument("--video-codec", default="mp4v")
    parser.add_argument("--sync-no-audio", action="store_true")
    parser.add_argument("--audio-device", default=None)
    parser.add_argument("--audio-sample-rate", type=int, default=44100)
    parser.add_argument("--audio-block-size", type=int, default=4096)
    parser.add_argument("--list-audio-devices", action="store_true")
    parser.add_argument("--start-on", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--no-gpio", action="store_true")
    parser.add_argument("--gpio-button-pin", type=int, default=17)
    parser.add_argument("--gpio-led-pin", type=int, default=27)
    return parser.parse_args()


def parse_audio_device(device: str | None):
    if device is None or device == "":
        return None

    return int(device) if device.isdigit() else device


def apply_flip(frame, flip_mode: str):
    if flip_mode == "horizontal":
        return cv2.flip(frame, 1)
    if flip_mode == "vertical":
        return cv2.flip(frame, 0)
    if flip_mode == "both":
        return cv2.flip(frame, -1)
    return frame


def make_metadata(args) -> dict:
    return {
        "recording_mode": "fretlog_recorder",
        "camera": {
            "index": args.camera_index,
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "flip": args.flip,
        },
        "audio": {
            "recorded": not args.sync_no_audio,
            "device": args.audio_device,
            "sample_rate": args.audio_sample_rate,
            "block_size": args.audio_block_size,
        },
        "sync_hint": "Clap once in view of the camera at the beginning of the take.",
    }


def write_take_status(session_dir: Path | None, status: str, reason: str | None = None) -> None:
    if session_dir is None:
        return

    payload = {
        "status": status,
        "reason": reason,
        "written_at_unix": time.time(),
    }

    with (session_dir / "take_status.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def draw_overlay(frame, recorder: SyncSessionRecorder) -> None:
    status = "REC" if recorder.enabled else "IDLE"
    accent = RECORD if recorder.enabled else PRIMARY
    elapsed = recorder.elapsed_seconds() if recorder.enabled else 0.0

    draw_hairline_panel(frame, (16, 16), (530, 116), fill=CANVAS_RAISED, border=HAIRLINE)
    draw_label(frame, "FRETLOG CAPTURE", (32, 42), MUTE)
    draw_chip(frame, status, (32, 56), accent=accent)
    draw_text(
        frame,
        f"{elapsed:06.1f}s",
        (132, 77),
        0.72,
        INK,
        2,
    )
    draw_text(
        frame,
        "o record/stop / c cancel / q finish",
        (32, 102),
        0.5,
        BODY,
        1,
    )

    if recorder.session_dir is not None:
        draw_text(
            frame,
            fit_text(str(recorder.session_dir), 58),
            (246, 77),
            0.45,
            BODY,
            1,
        )


def run_fretlog_recorder() -> int:
    args = parse_args()

    if args.list_audio_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return 0

    recorder = SyncSessionRecorder(
        SyncSessionConfig(
            output_dir=args.sync_output_dir,
            sample_rate=args.audio_sample_rate,
            record_audio=not args.sync_no_audio,
            record_video=True,
            video_fps=args.fps,
            video_codec=args.video_codec,
            video_filename="video.mp4",
        )
    )
    audio_tracker: AudioPitchTracker | None = None
    toggle_requested = Event()
    take_canceled = False

    def start_audio() -> None:
        nonlocal audio_tracker

        if args.sync_no_audio or audio_tracker is not None:
            return

        audio_tracker = AudioPitchTracker(
            device=parse_audio_device(args.audio_device),
            sample_rate=args.audio_sample_rate,
            block_size=args.audio_block_size,
            audio_block_callback=recorder.add_audio_block,
        )
        try:
            audio_tracker.start()
        except Exception as error:
            audio_tracker = None
            print(f"audio recording disabled: {error}")

    def stop_audio() -> None:
        nonlocal audio_tracker

        if audio_tracker is None:
            return

        audio_tracker.stop()
        audio_tracker = None

    def set_recording_enabled(enabled: bool) -> None:
        nonlocal take_canceled

        if enabled == recorder.enabled:
            return

        if enabled:
            take_canceled = False
            start_audio()
            recorder.start(metadata=make_metadata(args))
        else:
            stop_audio()
            recorder.stop()
            if not take_canceled:
                write_take_status(recorder.session_dir, "completed")

        gpio_controller.set_led(recorder.enabled)

    def cancel_take() -> None:
        nonlocal take_canceled

        take_canceled = True

        if recorder.enabled:
            recorder.record_event("take_canceled", {"reason": "user_cancel"})
            stop_audio()
            recorder.stop()

        write_take_status(recorder.session_dir, "canceled", reason="user_cancel")
        gpio_controller.set_led(False)
        print("take canceled. Analysis will be skipped by run_fretlog_take.py.")

    def toggle_recording() -> None:
        toggle_requested.set()

    gpio_controller = SyncGPIOController(
        button_pin=None if args.no_gpio else args.gpio_button_pin,
        led_pin=None if args.no_gpio else args.gpio_led_pin,
        on_toggle=toggle_recording,
        enabled=not args.no_gpio,
    )

    capture = cv2.VideoCapture(args.camera_index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_FPS, args.fps)

    if not capture.isOpened():
        gpio_controller.close()
        raise SystemExit(f"Could not open camera index {args.camera_index}")

    print("FretLog recorder started.")
    print("Clap once at the beginning of each take for AV sync.")
    print("Press o to start/stop recording, c to cancel, q to finish.")
    if args.no_preview:
        print("Preview is disabled. Use GPIO or Ctrl+C to stop.")

    try:
        if args.start_on:
            set_recording_enabled(True)

        while True:
            ok, frame = capture.read()
            if not ok:
                print("camera frame read failed")
                time.sleep(0.05)
                continue

            frame = apply_flip(frame, args.flip)

            if toggle_requested.is_set():
                toggle_requested.clear()
                set_recording_enabled(not recorder.enabled)

            recorder.record_video_frame(frame)

            if not args.no_preview:
                preview = frame.copy()
                draw_overlay(preview, recorder)
                cv2.imshow(WINDOW_NAME, preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("o"):
                    set_recording_enabled(not recorder.enabled)
                elif key == ord("c"):
                    cancel_take()
                    break
                elif key == ord("q") or key == 27:
                    break

    except KeyboardInterrupt:
        pass
    finally:
        set_recording_enabled(False)
        stop_audio()
        capture.release()
        gpio_controller.close()
        if not args.no_preview:
            cv2.destroyAllWindows()

    return CANCEL_EXIT_CODE if take_canceled else 0


if __name__ == "__main__":
    sys.exit(run_fretlog_recorder())
