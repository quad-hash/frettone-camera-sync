import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYNC_OUTPUT_DIR = PROJECT_ROOT / "data" / "sync_sessions"
CANCEL_EXIT_CODE = 20


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record, analyze, report, and replay one FretLog take.",
    )
    parser.add_argument("--sync-root", type=Path, default=DEFAULT_SYNC_OUTPUT_DIR)
    parser.add_argument("--skip-record", action="store_true")
    parser.add_argument("--no-mediapipe-hands", action="store_true")
    parser.add_argument("--hand-viewer", action="store_true")
    parser.add_argument("--no-playback", action="store_true")
    parser.add_argument("--no-audio-playback", action="store_true")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.62)
    parser.add_argument("--notes-file", default="notes.json")

    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--flip",
        choices=("none", "horizontal", "vertical", "both"),
        default="horizontal",
    )
    parser.add_argument("--audio-device", default=None)
    parser.add_argument("--audio-sample-rate", type=int, default=44100)
    parser.add_argument("--audio-block-size", type=int, default=4096)
    parser.add_argument("--sync-no-audio", action="store_true")
    parser.add_argument("--start-on", action="store_true")
    parser.add_argument("--no-gpio", action="store_true")
    parser.add_argument("--gpio-button-pin", type=int, default=17)
    parser.add_argument("--gpio-led-pin", type=int, default=27)
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print()
    print("$ " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT))

    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def run_recorder_command(command: list[str]) -> int:
    print()
    print("$ " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT))

    if completed.returncode not in (0, CANCEL_EXIT_CODE):
        raise SystemExit(completed.returncode)

    return completed.returncode


def session_dirs(sync_root: Path) -> set[Path]:
    return {
        path.resolve()
        for path in sync_root.glob("sync*")
        if path.is_dir()
    }


def find_latest_session(sync_root: Path) -> Path:
    candidates = sorted(
        [path for path in sync_root.glob("sync*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(f"No sync sessions found in {sync_root}")

    return candidates[0]


def find_new_session(sync_root: Path, before: set[Path]) -> Path:
    after = sorted(
        [
            path
            for path in sync_root.glob("sync*")
            if path.is_dir() and path.resolve() not in before
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if after:
        return after[0]

    return find_latest_session(sync_root)


def build_recorder_command(args) -> list[str]:
    command = [
        sys.executable,
        "src/run_fretlog_recorder.py",
        "--sync-output-dir",
        str(args.sync_root),
        "--camera-index",
        str(args.camera_index),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--fps",
        str(args.fps),
        "--flip",
        args.flip,
        "--audio-sample-rate",
        str(args.audio_sample_rate),
        "--audio-block-size",
        str(args.audio_block_size),
        "--gpio-button-pin",
        str(args.gpio_button_pin),
        "--gpio-led-pin",
        str(args.gpio_led_pin),
    ]

    if args.audio_device:
        command.extend(["--audio-device", args.audio_device])

    if args.sync_no_audio:
        command.append("--sync-no-audio")

    if args.start_on:
        command.append("--start-on")

    if args.no_gpio:
        command.append("--no-gpio")

    return command


def build_pipeline_command(args, session_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "src/run_fretlog_pipeline.py",
        str(session_dir),
        "--notes-file",
        args.notes_file,
        "--threshold",
        str(args.threshold),
    ]

    if not args.no_mediapipe_hands:
        command.append("--mediapipe-hands")

    if args.config is not None:
        command.extend(["--config", str(args.config)])

    return command


def run_fretlog_take() -> None:
    args = parse_args()
    args.sync_root.mkdir(parents=True, exist_ok=True)
    before = session_dirs(args.sync_root)

    print("FretLog take")
    print("1. Recorder opens first.")
    print("2. Press o to start/stop recording, c to cancel, q to close the recorder.")
    print("3. This script will then analyze and open playback.")
    print("Remember: clap once at the beginning of the take.")

    if args.skip_record:
        session_dir = find_latest_session(args.sync_root)
    else:
        recorder_returncode = run_recorder_command(build_recorder_command(args))

        if recorder_returncode == CANCEL_EXIT_CODE:
            print()
            print("take canceled. Analysis and playback skipped.")
            return

        session_dir = find_new_session(args.sync_root, before)

    print()
    print(f"session: {session_dir}")
    run_command(build_pipeline_command(args, session_dir))

    if args.hand_viewer:
        run_command(
            [
                sys.executable,
                "src/run_hand_tracks_viewer.py",
                str(session_dir),
            ]
        )

    if not args.no_playback:
        playback_command = [
            sys.executable,
            "src/run_sync_playback.py",
            str(session_dir),
        ]

        if args.no_audio_playback:
            playback_command.append("--no-audio")

        run_command(playback_command)

    print()
    print("take complete")
    print(f"session folder: {session_dir}")


if __name__ == "__main__":
    run_fretlog_take()
