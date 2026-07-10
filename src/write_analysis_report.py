import argparse
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYNC_OUTPUT_DIR = PROJECT_ROOT / "data" / "sync_sessions"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Write a Markdown report for an analyzed FretLog session.",
    )
    parser.add_argument("session_dir", type=Path, nargs="?", default=None)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--sync-root", type=Path, default=DEFAULT_SYNC_OUTPUT_DIR)
    parser.add_argument("--notes-file", default="notes.json")
    parser.add_argument("--output", default="analysis_report.md")
    parser.add_argument("--threshold", type=float, default=0.62)
    parser.add_argument("--review-limit", type=int, default=80)
    parser.add_argument("--detail-limit", type=int, default=25)
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


def load_notes_data(session_dir: Path, notes_file: str) -> dict:
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


def visual_value(note: dict, key: str, default="-"):
    visual = note.get("visual") or {}
    value = visual.get(key)
    return default if value is None else value


def hand_summary(note: dict) -> str:
    visual = note.get("visual") or {}
    parts = []

    if visual.get("hand_center_fret") is not None:
        parts.append(
            f"bg~{visual.get('hand_center_fret')}F"
            f"({visual.get('hand_start_fret')}-{visual.get('hand_end_fret')})"
        )

    if visual.get("mediapipe_hand_center_fret") is not None:
        parts.append(
            f"mp~{visual.get('mediapipe_hand_center_fret')}F"
            f"({visual.get('mediapipe_hand_start_fret')}-"
            f"{visual.get('mediapipe_hand_end_fret')})"
            f" c={float(visual.get('mediapipe_hand_confidence', 0.0)):.2f}"
        )

    return ", ".join(parts) if parts else "-"


def selected_candidate(note: dict) -> dict | None:
    candidates = note.get("candidates", [])
    selected_index = note.get("selected_candidate_index")

    if selected_index is None or selected_index < 0 or selected_index >= len(candidates):
        return None

    return candidates[selected_index]


def candidate_position(candidate: dict) -> str:
    return f"{candidate.get('string_number')}s/{candidate.get('fret_number')}F"


def candidate_score_text(candidate: dict) -> str:
    return (
        f"v={float(candidate.get('visual_score', 0.0)):.3f} "
        f"str={float(candidate.get('string_visual_score', 0.0)):.3f} "
        f"bg={float(candidate.get('hand_visual_score', 0.0)):.3f} "
        f"mp={float(candidate.get('mediapipe_hand_visual_score', 0.0)):.3f}"
    )


def candidate_sort_key(item):
    index, candidate = item
    return (
        -float(candidate.get("visual_score", 0.0)),
        candidate.get("string_number", 0),
        candidate.get("fret_number", 0),
        index,
    )


def ranked_candidates(note: dict, limit: int = 6) -> list[tuple[int, dict]]:
    candidates = list(enumerate(note.get("candidates", [])))
    return sorted(candidates, key=candidate_sort_key)[:limit]


def note_warnings(note: dict, threshold: float) -> list[str]:
    warnings = []
    visual = note.get("visual") or {}
    confidence = float(note.get("confidence", 0.0))
    candidate_count = note.get("candidate_count", len(note.get("candidates", [])))
    fret_number = note.get("fret")
    mp_fret = visual.get("mediapipe_hand_center_fret")
    bg_fret = visual.get("hand_center_fret")

    if confidence < threshold:
        warnings.append("low_confidence")

    if candidate_count > 1:
        warnings.append("ambiguous_pitch")

    if note.get("string") is None or fret_number is None:
        warnings.append("unresolved")

    if not visual.get("available"):
        warnings.append("no_visual")

    if visual.get("available") and mp_fret is None:
        warnings.append("no_mediapipe_hand")

    if fret_number is not None and mp_fret is not None and abs(fret_number - mp_fret) >= 4:
        warnings.append("selected_far_from_mediapipe")

    if fret_number is not None and bg_fret is not None and abs(fret_number - bg_fret) >= 4:
        warnings.append("selected_far_from_bg_hand")

    return warnings


def interesting_note_items(notes: list[dict], threshold: float) -> list[tuple[int, dict, list[str]]]:
    items = []

    for index, note in enumerate(notes):
        warnings = note_warnings(note, threshold)

        if warnings:
            items.append((index, note, warnings))

    return items


def append_candidate_details(
    lines: list[str],
    note_index: int,
    note: dict,
    warnings: list[str],
) -> None:
    lines.append(f"### Note {note_index}: {note.get('note_name', '-')}")
    lines.append("")
    lines.append(
        "- "
        f"Time: {note.get('t_onset', 0.0):.3f}-{note.get('t_offset', 0.0):.3f}s"
    )
    lines.append(f"- Selected: `{selected_text(note)}`")
    lines.append(f"- Confidence: {note.get('confidence', 0.0):.3f}")
    lines.append(f"- Status: `{note.get('status', '-')}`")
    lines.append(f"- Hand: {hand_summary(note)}")
    lines.append(f"- Warnings: {', '.join(f'`{warning}`' for warning in warnings)}")
    lines.append("")

    visual = note.get("visual") or {}

    if visual:
        lines.append(
            "- Visual: "
            f"changed_pixels={visual.get('changed_pixels', '-')} "
            f"markers={visual.get('marker_count', '-')} "
            f"inliers={visual.get('marker_inliers', '-')}"
        )
        lines.append("")

    lines.append("| selected | candidate | scores |")
    lines.append("|---:|---|---|")

    selected_index = note.get("selected_candidate_index")

    for candidate_index, candidate in ranked_candidates(note):
        marker = "*" if candidate_index == selected_index else ""
        lines.append(
            "| "
            f"{marker} | "
            f"`#{candidate_index:02d} {candidate_position(candidate)}` | "
            f"{candidate_score_text(candidate)} |"
        )

    lines.append("")


def make_report(
    session_dir: Path,
    notes_data: dict,
    threshold: float,
    review_limit: int = 80,
    detail_limit: int = 25,
) -> str:
    notes = notes_data.get("notes", [])
    visual_analysis = notes_data.get("visual_analysis") or {}
    hand_track_summary = visual_analysis.get("hand_tracks") or {}
    low_confidence = [
        (index, note)
        for index, note in enumerate(notes)
        if note.get("confidence", 0.0) < threshold
    ]
    ambiguous = [
        note
        for note in notes
        if note.get("candidate_count", len(note.get("candidates", []))) > 1
    ]
    visual = [
        note
        for note in notes
        if (note.get("visual") or {}).get("available")
    ]
    unresolved = [
        note
        for note in notes
        if note.get("string") is None or note.get("fret") is None
    ]
    mediapipe_hand_notes = [
        note
        for note in notes
        if (note.get("visual") or {}).get("mediapipe_hand_center_fret") is not None
    ]
    far_from_mediapipe = [
        note
        for note in notes
        if note.get("fret") is not None
        and (note.get("visual") or {}).get("mediapipe_hand_center_fret") is not None
        and abs(note["fret"] - (note.get("visual") or {})["mediapipe_hand_center_fret"]) >= 4
    ]
    interesting_notes = interesting_note_items(notes, threshold)
    candidate_counts = Counter(
        note.get("candidate_count", len(note.get("candidates", [])))
        for note in notes
    )
    status_counts = Counter(note.get("status", "-") for note in notes)
    average_confidence = (
        sum(note.get("confidence", 0.0) for note in notes) / len(notes)
        if notes
        else 0.0
    )
    lines = [
        "# FretLog Analysis Report",
        "",
        f"- Session: `{session_dir}`",
        f"- Notes: {len(notes)}",
        f"- Average confidence: {average_confidence:.3f}",
        f"- Low confidence (< {threshold:.2f}): {len(low_confidence)}",
        f"- Ambiguous candidates: {len(ambiguous)}",
        f"- Visual evidence available: {len(visual)}",
        f"- MediaPipe hand evidence: {len(mediapipe_hand_notes)}",
        f"- Selected far from MediaPipe hand: {len(far_from_mediapipe)}",
        f"- Unresolved: {len(unresolved)}",
        f"- Hand tracks: {hand_track_summary.get('frames_with_hands', 0)} / "
        f"{hand_track_summary.get('frame_count', 0)} frames with hands",
        "",
        "## Candidate Count Distribution",
        "",
    ]

    for candidate_count, count in sorted(candidate_counts.items()):
        lines.append(f"- {candidate_count} candidates: {count}")

    lines.extend(["", "## Status Distribution", ""])

    for status, count in sorted(status_counts.items()):
        lines.append(f"- `{status}`: {count}")

    lines.extend(["", "## Review Queue", ""])

    if not interesting_notes:
        lines.append("No notes need review.")
    else:
        lines.append("| # | time | note | selected | hand | confidence | warnings |")
        lines.append("|---:|---:|---:|---:|---|---:|---|")

        for index, note, warnings in interesting_notes[:review_limit]:
            lines.append(
                "| "
                f"{index} | "
                f"{note.get('t_onset', 0.0):.3f}-{note.get('t_offset', 0.0):.3f} | "
                f"{note.get('note_name', '-')} | "
                f"`{selected_text(note)}` | "
                f"{hand_summary(note)} | "
                f"{note.get('confidence', 0.0):.3f} | "
                f"{', '.join(f'`{warning}`' for warning in warnings)} |"
            )

        if len(interesting_notes) > review_limit:
            lines.append("")
            lines.append(f"... {len(interesting_notes) - review_limit} more review notes omitted.")

    lines.extend(["", "## Low Confidence Notes", ""])

    if not low_confidence:
        lines.append("No low-confidence notes.")
    else:
        lines.append("| # | time | note | selected | hand | candidates | confidence | status |")
        lines.append("|---:|---:|---:|---:|---|---:|---:|---|")

        for index, note in low_confidence[:50]:
            lines.append(
                "| "
                f"{index} | "
                f"{note.get('t_onset', 0.0):.3f}-{note.get('t_offset', 0.0):.3f} | "
                f"{note.get('note_name', '-')} | "
                f"{selected_text(note)} | "
                f"{hand_summary(note)} | "
                f"{note.get('candidate_count', len(note.get('candidates', [])))} | "
                f"{note.get('confidence', 0.0):.3f} | "
                f"{note.get('status', '-')} |"
            )

        if len(low_confidence) > 50:
            lines.append("")
            lines.append(f"... {len(low_confidence) - 50} more low-confidence notes omitted.")

    lines.extend(["", "## Candidate Details", ""])

    if not interesting_notes:
        lines.append("No candidate details needed.")
    else:
        for index, note, warnings in interesting_notes[:detail_limit]:
            append_candidate_details(lines, index, note, warnings)

        if len(interesting_notes) > detail_limit:
            lines.append(f"... {len(interesting_notes) - detail_limit} more detailed notes omitted.")
            lines.append("")

    lines.append("")
    return "\n".join(lines)


def run_write_analysis_report() -> None:
    args = parse_args()
    session_dir = args.session_dir

    if args.latest or session_dir is None:
        session_dir = find_latest_session(args.sync_root)

    notes_data = load_notes_data(session_dir, args.notes_file)
    report = make_report(
        session_dir=session_dir,
        notes_data=notes_data,
        threshold=args.threshold,
        review_limit=args.review_limit,
        detail_limit=args.detail_limit,
    )
    output_path = session_dir / args.output
    output_path.write_text(report, encoding="utf-8")
    print(f"analysis report written: {output_path}")


if __name__ == "__main__":
    run_write_analysis_report()
