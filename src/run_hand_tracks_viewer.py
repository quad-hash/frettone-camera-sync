import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from run_offline_analyze import (
    WARP_SIZE,
    load_frame_times,
    mediapipe_hand_evidence_from_track,
    nearest_hand_track_frame,
    warp_fretboard_frame,
)
from ui_theme import (
    BODY,
    CANVAS_RAISED,
    HAIRLINE,
    INK,
    MUTE,
    PRIMARY,
    PRIMARY_SOFT,
    WARNING,
    draw_chip,
    draw_hairline_panel,
    draw_label,
    draw_text,
    fit_text,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYNC_OUTPUT_DIR = PROJECT_ROOT / "data" / "sync_sessions"
WINDOW_NAME = "FretLog - Hand Tracks"
HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preview MediaPipe hand tracks over a recorded FretLog video.",
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
    parser.add_argument("--video-file", default="video.mp4")
    parser.add_argument("--hand-tracks", default="hand_tracks.json")
    parser.add_argument("--frame-times", default="frame_times.json")
    parser.add_argument("--marker-min-area", type=float, default=45.0)
    parser.add_argument("--hand-window", type=float, default=0.08)
    parser.add_argument("--output-video", default=None)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--playback-speed", type=float, default=1.0)
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


def load_hand_tracks(session_dir: Path, hand_tracks_name: str) -> list[dict]:
    hand_tracks_path = session_dir / hand_tracks_name

    if not hand_tracks_path.exists():
        raise SystemExit(
            f"hand tracks not found: {hand_tracks_path}\n"
            "Create it first with:\n"
            f"  python src/run_mediapipe_hands.py {session_dir}\n"
            "or run the full pipeline with:\n"
            "  python src/run_fretlog_pipeline.py --latest --mediapipe-hands"
        )

    with hand_tracks_path.open("r", encoding="utf-8") as file:
        raw_tracks = json.load(file)

    return sorted(
        raw_tracks.get("frames", []),
        key=lambda item: float(item.get("time_seconds", 0.0)),
    )


def frame_time_for_index(frame_index: int, frame_times: list[float], fps: float) -> float:
    if 0 <= frame_index < len(frame_times):
        return float(frame_times[frame_index])

    if fps > 0:
        return frame_index / fps

    return 0.0


def make_video_writer(output_path: Path, fps: float, frame_size: tuple[int, int]):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, max(1.0, fps), frame_size)

    if not writer.isOpened():
        writer.release()
        raise SystemExit(f"Could not open VideoWriter: {output_path}")

    return writer


def draw_hand_landmarks(frame, hand: dict, hand_index: int) -> None:
    landmarks = hand.get("landmarks", [])
    points = []

    for landmark in landmarks:
        px = landmark.get("px")
        py = landmark.get("py")

        if px is None or py is None:
            points.append(None)
        else:
            points.append((int(round(px)), int(round(py))))

    for start, end in HAND_CONNECTIONS:
        if start >= len(points) or end >= len(points):
            continue

        point_a = points[start]
        point_b = points[end]

        if point_a is None or point_b is None:
            continue

        cv2.line(frame, point_a, point_b, PRIMARY, 2, cv2.LINE_AA)

    for index, point in enumerate(points):
        if point is None:
            continue

        radius = 5 if index in (4, 8, 12, 16, 20) else 3
        cv2.circle(frame, point, radius, PRIMARY_SOFT, -1, cv2.LINE_AA)
        cv2.circle(frame, point, radius + 2, (20, 20, 20), 1, cv2.LINE_AA)

    bbox = hand.get("bbox", {})
    x_min = bbox.get("x_min")
    y_min = bbox.get("y_min")
    x_max = bbox.get("x_max")
    y_max = bbox.get("y_max")

    if None not in (x_min, y_min, x_max, y_max):
        top_left = (int(round(x_min)), int(round(y_min)))
        bottom_right = (int(round(x_max)), int(round(y_max)))
        cv2.rectangle(frame, top_left, bottom_right, PRIMARY, 1, cv2.LINE_AA)

    label = hand.get("label") or f"hand {hand_index + 1}"
    score = hand.get("score")

    if score is not None:
        label = f"{label} {float(score):.2f}"

    center = hand.get("center", {})
    center_px = center.get("px")
    center_py = center.get("py")

    if center_px is not None and center_py is not None:
        draw_text(
            frame,
            label,
            (int(center_px) + 10, int(center_py) - 10),
            0.46,
            INK,
            1,
        )


def canonical_rect_to_image_polygon(
    x_min: float,
    x_max: float,
    homography,
) -> np.ndarray | None:
    if homography is None:
        return None

    try:
        inverse_homography = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return None

    _, height = WARP_SIZE
    canonical_points = np.array(
        [
            [x_min, 0.0],
            [x_max, 0.0],
            [x_max, float(height)],
            [x_min, float(height)],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    image_points = cv2.perspectiveTransform(canonical_points, inverse_homography)
    return image_points.reshape(-1, 2).astype(np.int32)


def draw_fretboard_projection(frame, homography) -> None:
    width, height = WARP_SIZE
    polygon = canonical_rect_to_image_polygon(0.0, float(width), homography)

    if polygon is None:
        return

    cv2.polylines(frame, [polygon], True, HAIRLINE, 2, cv2.LINE_AA)
    cv2.line(frame, tuple(polygon[0]), tuple(polygon[1]), PRIMARY, 2, cv2.LINE_AA)


def draw_hand_fret_band(frame, hand_evidence: dict, homography) -> None:
    if not hand_evidence.get("available"):
        return

    start_fret = hand_evidence.get("start_fret")
    end_fret = hand_evidence.get("end_fret")

    if start_fret is None or end_fret is None:
        return

    width, _ = WARP_SIZE
    start_fret = max(1, min(12, int(start_fret)))
    end_fret = max(start_fret, min(12, int(end_fret)))
    x_min = (start_fret - 1) / 12.0 * width
    x_max = end_fret / 12.0 * width
    polygon = canonical_rect_to_image_polygon(x_min, x_max, homography)

    if polygon is None:
        return

    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], PRIMARY, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
    cv2.polylines(frame, [polygon], True, PRIMARY, 2, cv2.LINE_AA)


def draw_status_overlay(
    frame,
    session_dir: Path,
    frame_index: int,
    frame_time: float,
    hand_frame: dict | None,
    hand_evidence: dict,
    warp_ok: bool,
) -> None:
    hand_count = len(hand_frame.get("hands", [])) if hand_frame is not None else 0
    status_text = "HAND OK" if hand_count else "NO HAND"
    status_accent = PRIMARY if hand_count else WARNING
    warp_text = "WARP OK" if warp_ok else "WARP LOST"

    draw_hairline_panel(frame, (16, 16), (560, 124), fill=CANVAS_RAISED)
    draw_label(frame, "HAND TRACK REVIEW", (32, 42), MUTE)
    draw_chip(frame, status_text, (32, 56), accent=status_accent)
    draw_chip(frame, warp_text, (152, 56), accent=PRIMARY if warp_ok else WARNING)
    draw_text(
        frame,
        f"frame {frame_index}  {frame_time:07.3f}s  hands {hand_count}",
        (32, 104),
        0.5,
        BODY,
        1,
    )

    if hand_evidence.get("available"):
        fret_text = (
            f"hand~{hand_evidence.get('center_fret')}F "
            f"range {hand_evidence.get('start_fret')}-{hand_evidence.get('end_fret')}F "
            f"conf {float(hand_evidence.get('confidence', 0.0)):.2f}"
        )
        draw_text(frame, fret_text, (300, 76), 0.48, PRIMARY, 1)
    else:
        draw_text(frame, "hand fret: -", (300, 76), 0.48, BODY, 1)

    draw_text(frame, fit_text(str(session_dir), 70), (300, 104), 0.4, BODY, 1)


def run_hand_tracks_viewer() -> None:
    args = parse_args()

    if args.latest or args.session_dir is None:
        session_dir = find_latest_session(args.sync_root)
    else:
        session_dir = args.session_dir

    video_path = session_dir / args.video_file

    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    hand_frames = load_hand_tracks(session_dir, args.hand_tracks)
    frame_times = load_frame_times(session_dir, args.frame_times)
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    writer = None

    if args.output_video is not None:
        output_path = session_dir / args.output_video
        writer = make_video_writer(output_path, fps, (frame_width, frame_height))
        print(f"writing hand track preview: {output_path}")

    if not args.no_preview:
        cv2.namedWindow(WINDOW_NAME)

    print(f"reviewing hand tracks: {session_dir}")
    print("controls: space pause / n next frame / q quit")

    paused = False
    frame_index = 0
    processed_count = 0

    try:
        while True:
            if not paused:
                ok, frame = capture.read()

                if not ok:
                    break

                frame_time = frame_time_for_index(frame_index, frame_times, fps)
                hand_frame, hand_delta = nearest_hand_track_frame(
                    hand_frames,
                    frame_time,
                    args.hand_window,
                )
                warp = warp_fretboard_frame(frame, marker_min_area=args.marker_min_area)
                hand_evidence = (
                    mediapipe_hand_evidence_from_track(
                        hand_frame,
                        warp["homography"] if warp is not None else None,
                        time_delta_seconds=hand_delta,
                        max_delta_seconds=args.hand_window,
                    )
                    if hand_frame is not None
                    else {
                        "available": False,
                        "center_fret": None,
                        "start_fret": None,
                        "end_fret": None,
                        "confidence": 0.0,
                    }
                )

                if warp is not None:
                    draw_fretboard_projection(frame, warp["homography"])
                    draw_hand_fret_band(frame, hand_evidence, warp["homography"])

                if hand_frame is not None:
                    for hand_index, hand in enumerate(hand_frame.get("hands", [])):
                        draw_hand_landmarks(frame, hand, hand_index)

                draw_status_overlay(
                    frame,
                    session_dir=session_dir,
                    frame_index=frame_index,
                    frame_time=frame_time,
                    hand_frame=hand_frame,
                    hand_evidence=hand_evidence,
                    warp_ok=warp is not None,
                )

                if writer is not None:
                    writer.write(frame)

                processed_count += 1
                frame_index += 1

                if args.max_frames > 0 and processed_count >= args.max_frames:
                    break

            if args.no_preview:
                continue

            cv2.imshow(WINDOW_NAME, frame)
            delay_ms = max(1, int(1000 / max(1.0, fps * max(0.01, args.playback_speed))))
            key = cv2.waitKey(0 if paused else delay_ms) & 0xFF

            if key == ord("q") or key == 27:
                break
            if key == ord(" "):
                paused = not paused
            elif key == ord("n"):
                paused = False

    finally:
        capture.release()

        if writer is not None:
            writer.release()

        if not args.no_preview:
            cv2.destroyAllWindows()

    print(f"hand track review complete: frames={processed_count}")


if __name__ == "__main__":
    run_hand_tracks_viewer()
