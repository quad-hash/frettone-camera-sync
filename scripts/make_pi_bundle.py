import argparse
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dist" / "fretlog_pi_capture"
FILES_TO_COPY = [
    "README.md",
    "requirements-pi.txt",
    "docs/raspberry_pi_capture.md",
    "src/audio_pitch_detector.py",
    "src/fretboard_note_mapper.py",
    "src/run_audio_pitch.py",
    "src/run_fretlog_pi_capture.py",
    "src/run_fretlog_recorder.py",
    "src/sync_session.py",
    "src/transfer_latest_session.py",
    "src/ui_theme.py",
]
DIRECTORIES_TO_CREATE = [
    "data/sync_sessions",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a lightweight Raspberry Pi capture bundle.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--zip", action="store_true", help="Also create a .zip archive.")
    return parser.parse_args()


def clean_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def copy_file(relative_path: str, output_dir: Path) -> None:
    source = PROJECT_ROOT / relative_path
    destination = output_dir / relative_path

    if not source.exists():
        raise FileNotFoundError(source)

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def make_zip(output_dir: Path) -> Path:
    archive_base = output_dir.parent / output_dir.name
    archive_path = shutil.make_archive(str(archive_base), "zip", output_dir)
    return Path(archive_path)


def make_pi_bundle() -> None:
    args = parse_args()
    output_dir = args.output_dir
    clean_output_dir(output_dir)

    for relative_path in FILES_TO_COPY:
        copy_file(relative_path, output_dir)

    for relative_path in DIRECTORIES_TO_CREATE:
        (output_dir / relative_path).mkdir(parents=True, exist_ok=True)

    print(f"Pi bundle written: {output_dir}")

    if args.zip:
        archive_path = make_zip(output_dir)
        print(f"Pi bundle zip written: {archive_path}")


if __name__ == "__main__":
    make_pi_bundle()
