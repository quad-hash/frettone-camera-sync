import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYNC_OUTPUT_DIR = PROJECT_ROOT / "data" / "sync_sessions"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transfer the latest FretLog sync session with scp.",
    )
    parser.add_argument("destination", help="scp destination, e.g. user@host:/path/to/sync_sessions/")
    parser.add_argument("--sync-root", type=Path, default=DEFAULT_SYNC_OUTPUT_DIR)
    parser.add_argument("--session-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
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


def transfer_session(session_dir: Path, destination: str, dry_run: bool = False) -> None:
    command = ["scp", "-r", str(session_dir), destination]
    print("$ " + " ".join(command), flush=True)

    if dry_run:
        return

    completed = subprocess.run(command)

    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def run_transfer_latest_session() -> None:
    args = parse_args()
    session_dir = args.session_dir if args.session_dir is not None else find_latest_session(args.sync_root)

    if not session_dir.exists():
        raise FileNotFoundError(f"session not found: {session_dir}")

    transfer_session(session_dir, args.destination, dry_run=args.dry_run)
    print(f"transferred: {session_dir}")


if __name__ == "__main__":
    run_transfer_latest_session()
