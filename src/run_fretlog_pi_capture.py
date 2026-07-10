import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYNC_OUTPUT_DIR = PROJECT_ROOT / "data" / "sync_sessions"
CANCEL_EXIT_CODE = 20


def parse_args():
    parser = argparse.ArgumentParser(
        description="Raspberry Pi capture workflow: record a take and optionally scp it to a PC.",
    )
    parser.add_argument("--sync-output-dir", type=Path, default=DEFAULT_SYNC_OUTPUT_DIR)
    parser.add_argument("--transfer-to", default=None, help="scp destination, e.g. user@host:/path/to/sync_sessions/")
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
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--no-gpio", action="store_true")
    parser.add_argument("--gpio-button-pin", type=int, default=17)
    parser.add_argument("--gpio-led-pin", type=int, default=27)
    return parser.parse_args()


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
        str(args.sync_output_dir),
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

    if args.no_preview:
        command.append("--no-preview")

    if args.no_gpio:
        command.append("--no-gpio")

    return command


def run_command(command: list[str], allowed_return_codes: set[int] | None = None) -> int:
    if allowed_return_codes is None:
        allowed_return_codes = {0}

    print("$ " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT))

    if completed.returncode not in allowed_return_codes:
        raise SystemExit(completed.returncode)

    return completed.returncode


def load_take_status(session_dir: Path) -> dict:
    status_path = session_dir / "take_status.json"

    if not status_path.exists():
        return {"status": "unknown"}

    with status_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def transfer_session(session_dir: Path, destination: str) -> None:
    command = ["scp", "-r", str(session_dir), destination]
    print("$ " + " ".join(command), flush=True)
    completed = subprocess.run(command)

    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def run_fretlog_pi_capture() -> None:
    args = parse_args()
    args.sync_output_dir.mkdir(parents=True, exist_ok=True)
    before = session_dirs(args.sync_output_dir)

    print("FretLog Pi capture")
    print("o: record/stop, c: cancel, q: finish")
    return_code = run_command(
        build_recorder_command(args),
        allowed_return_codes={0, CANCEL_EXIT_CODE},
    )

    if return_code == CANCEL_EXIT_CODE:
        print("take canceled; transfer skipped")
        return

    session_dir = find_new_session(args.sync_output_dir, before)
    status = load_take_status(session_dir)

    if status.get("status") == "canceled":
        print(f"take canceled: {session_dir}")
        return

    print(f"captured: {session_dir}")

    if args.transfer_to:
        transfer_session(session_dir, args.transfer_to)
        print(f"transferred to: {args.transfer_to}")


if __name__ == "__main__":
    run_fretlog_pi_capture()
