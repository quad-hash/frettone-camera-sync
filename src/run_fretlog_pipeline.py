import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYNC_OUTPUT_DIR = PROJECT_ROOT / "data" / "sync_sessions"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the FretLog offline pipeline for a sync session.",
    )
    parser.add_argument("session_dir", type=Path, nargs="?", default=None)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--sync-root", type=Path, default=DEFAULT_SYNC_OUTPUT_DIR)
    parser.add_argument("--make-demo", action="store_true")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--skip-av-sync", action="store_true")
    parser.add_argument("--skip-analyze", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--skip-inspect", action="store_true")
    parser.add_argument(
        "--mediapipe-hands",
        action="store_true",
        help="Also run optional MediaPipe Hands tracking into hand_tracks.json.",
    )
    parser.add_argument("--no-video-analysis", action="store_true")
    parser.add_argument("--notes-file", default="notes.json")
    parser.add_argument("--threshold", type=float, default=0.62)
    return parser.parse_args()


def find_latest_session(sync_root: Path) -> Path:
    candidates = sorted(
        [path for path in sync_root.glob("sync*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(f"No sync sessions found in {sync_root}")

    return candidates[0]


def run_command(command: list[str], allow_failure: bool = False) -> int:
    print()
    print("$ " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT))

    if completed.returncode != 0 and not allow_failure:
        raise SystemExit(completed.returncode)

    return completed.returncode


def make_demo_session(sync_root: Path) -> Path:
    before = {
        path.resolve()
        for path in sync_root.glob("sync_demo_*")
        if path.is_dir()
    }
    run_command(
        [
            sys.executable,
            "src/make_demo_sync_session.py",
            "--output-dir",
            str(sync_root),
        ]
    )
    after = sorted(
        [
            path
            for path in sync_root.glob("sync_demo_*")
            if path.is_dir() and path.resolve() not in before
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if after:
        return after[0]

    return find_latest_session(sync_root)


def run_fretlog_pipeline() -> None:
    args = parse_args()
    sync_root = args.sync_root

    if args.make_demo:
        session_dir = make_demo_session(sync_root)
    elif args.latest or args.session_dir is None:
        session_dir = find_latest_session(sync_root)
    else:
        session_dir = args.session_dir

    print(f"session: {session_dir}", flush=True)

    if not args.skip_av_sync:
        av_command = [
            sys.executable,
            "src/estimate_av_sync.py",
            str(session_dir),
            "--update-meta",
        ]
        run_command(av_command, allow_failure=True)

    if args.mediapipe_hands:
        run_command(
            [
                sys.executable,
                "src/run_mediapipe_hands.py",
                str(session_dir),
            ]
        )

    if not args.skip_analyze:
        analyze_command = [
            sys.executable,
            "src/run_offline_analyze.py",
            str(session_dir),
            "--output",
            args.notes_file,
        ]

        if args.config is not None:
            analyze_command.extend(["--config", str(args.config)])

        if args.no_video_analysis:
            analyze_command.append("--no-video-analysis")

        run_command(analyze_command)

    if not args.skip_inspect:
        run_command(
            [
                sys.executable,
                "src/inspect_notes.py",
                str(session_dir),
                "--notes-file",
                args.notes_file,
                "--threshold",
                str(args.threshold),
            ]
        )

    if not args.skip_report:
        run_command(
            [
                sys.executable,
                "src/write_analysis_report.py",
                str(session_dir),
                "--notes-file",
                args.notes_file,
                "--threshold",
                str(args.threshold),
            ]
        )

    print()
    print("pipeline complete")


if __name__ == "__main__":
    run_fretlog_pipeline()
