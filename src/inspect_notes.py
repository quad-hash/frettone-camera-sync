import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYNC_OUTPUT_DIR = PROJECT_ROOT / "data" / "sync_sessions"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect offline FretLog notes.json confidence and candidates.",
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
    parser.add_argument("--notes-file", default="notes.json")
    parser.add_argument("--threshold", type=float, default=0.62)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--note-index", type=int, default=None)
    parser.add_argument("--show-all", action="store_true")
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


def load_notes_file(session_dir: Path, notes_file: str) -> dict:
    notes_path = session_dir / notes_file

    if not notes_path.exists():
        raise FileNotFoundError(f"notes file not found: {notes_path}")

    with notes_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def selected_text(note: dict) -> str:
    string_number = note.get("string")
    fret_number = note.get("fret")

    if string_number is None or fret_number is None:
        return "-"

    return f"{string_number}s/{fret_number}F"


def format_note_line(index: int, note: dict) -> str:
    visual = note.get("visual") or {}
    visual_text = "visual:yes" if visual.get("available") else "visual:no"
    hand_text = ""

    if visual.get("hand_center_fret") is not None:
        hand_text = f" hand~{visual['hand_center_fret']}F"

    return (
        f"{index:04d} "
        f"{note.get('t_onset', 0.0):7.3f}-{note.get('t_offset', 0.0):7.3f}s "
        f"{str(note.get('note_name', '-')):>4} "
        f"{selected_text(note):>7} "
        f"cand={note.get('candidate_count', len(note.get('candidates', []))):>2} "
        f"conf={note.get('confidence', 0.0):.2f} "
        f"{note.get('status', '-'):<34} "
        f"{visual_text}{hand_text}"
    )


def candidate_sort_key(candidate: dict):
    return (
        -float(candidate.get("visual_score", 0.0)),
        candidate.get("string_number", 0),
        candidate.get("fret_number", 0),
    )


def print_candidate_details(note_index: int, note: dict) -> None:
    selected_index = note.get("selected_candidate_index")
    candidates = note.get("candidates", [])

    print(format_note_line(note_index, note))
    print("candidates:")

    for index, candidate in sorted(enumerate(candidates), key=lambda item: candidate_sort_key(item[1])):
        marker = "*" if index == selected_index else " "
        print(
            f" {marker}#{index:02d} "
            f"{candidate.get('string_number')}s/{candidate.get('fret_number')}F "
            f"midi={candidate.get('midi_pitch')} "
            f"visual={float(candidate.get('visual_score', 0.0)):.3f} "
            f"string={float(candidate.get('string_visual_score', 0.0)):.3f} "
            f"hand={float(candidate.get('hand_visual_score', 0.0)):.3f}"
        )

    visual = note.get("visual") or {}

    if visual:
        print("visual:")
        print(json.dumps(visual, indent=2, ensure_ascii=False))


def summarize_notes(notes: list[dict], threshold: float) -> dict:
    low_confidence = [note for note in notes if note.get("confidence", 0.0) < threshold]
    ambiguous = [note for note in notes if note.get("candidate_count", len(note.get("candidates", []))) > 1]
    visual = [note for note in notes if (note.get("visual") or {}).get("available")]
    unresolved = [note for note in notes if note.get("string") is None or note.get("fret") is None]
    return {
        "note_count": len(notes),
        "low_confidence_count": len(low_confidence),
        "ambiguous_count": len(ambiguous),
        "visual_count": len(visual),
        "unresolved_count": len(unresolved),
    }


def run_inspect_notes() -> None:
    args = parse_args()
    session_dir = args.session_dir

    if args.latest or session_dir is None:
        session_dir = find_latest_session(args.sync_root)

    notes_data = load_notes_file(session_dir, args.notes_file)
    notes = notes_data.get("notes", [])
    summary = summarize_notes(notes, args.threshold)

    print(f"session: {session_dir}")
    print(
        "summary: "
        f"notes={summary['note_count']} "
        f"low_conf<{args.threshold:.2f}={summary['low_confidence_count']} "
        f"ambiguous={summary['ambiguous_count']} "
        f"visual={summary['visual_count']} "
        f"unresolved={summary['unresolved_count']}"
    )

    if args.note_index is not None:
        if args.note_index < 0 or args.note_index >= len(notes):
            raise IndexError(f"--note-index out of range: {args.note_index}")

        print_candidate_details(args.note_index, notes[args.note_index])
        return

    selected_notes = [
        (index, note)
        for index, note in enumerate(notes)
        if args.show_all or note.get("confidence", 0.0) < args.threshold
    ]

    if not selected_notes:
        print("no notes matched")
        return

    print("notes:")

    for index, note in selected_notes[: args.limit]:
        print(format_note_line(index, note))

    if len(selected_notes) > args.limit:
        print(f"... {len(selected_notes) - args.limit} more")


if __name__ == "__main__":
    run_inspect_notes()
