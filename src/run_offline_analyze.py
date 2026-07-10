import argparse
import json
import wave
from bisect import bisect_left
from pathlib import Path

import cv2
import numpy as np

from audio_pitch_detector import detect_pitch_from_samples
from fretboard_note_mapper import find_positions_for_midi_pitch, midi_pitch_to_note_name
from fretboard_marker_detector import detect_markers_for_color


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYNC_OUTPUT_DIR = PROJECT_ROOT / "data" / "sync_sessions"
EDGE_PAIR_MARKER_COUNTS = {
    "green": 2,
    "blue": 2,
    "yellow": 4,
    "pink": 4,
}
WARP_SIZE = (960, 180)
STRING_ORDER_TOP_TO_BOTTOM = (6, 5, 4, 3, 2, 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline analysis for a recorded FretLog sync session.",
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
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON analysis config. CLI options override config values.",
    )
    parser.add_argument("--output", default="notes.json")
    parser.add_argument("--max-fret", type=int, default=24)
    parser.add_argument("--frame-size", type=int, default=2048)
    parser.add_argument("--hop-size", type=int, default=512)
    parser.add_argument("--onset-threshold", type=float, default=3.0)
    parser.add_argument("--min-note-gap", type=float, default=0.09)
    parser.add_argument("--min-note-duration", type=float, default=0.08)
    parser.add_argument("--pitch-window", type=float, default=0.32)
    parser.add_argument("--pitch-skip", type=float, default=0.025)
    parser.add_argument("--rms-threshold", type=float, default=0.006)
    parser.add_argument("--min-frequency", type=float, default=70.0)
    parser.add_argument("--max-frequency", type=float, default=1000.0)
    parser.add_argument("--no-video-analysis", action="store_true")
    parser.add_argument("--video-file", default="video.mp4")
    parser.add_argument("--frame-times", default="frame_times.json")
    parser.add_argument("--av-sync-file", default="av_sync.json")
    parser.add_argument("--av-offset-ms", type=float, default=None)
    parser.add_argument("--video-marker-min-area", type=float, default=45.0)
    parser.add_argument("--video-baseline-offset", type=float, default=0.075)
    parser.add_argument("--video-onset-offset", type=float, default=0.045)
    parser.add_argument("--visual-weight", type=float, default=1.1)
    parser.add_argument("--hand-tracks", default="hand_tracks.json")
    parser.add_argument("--no-hand-tracks", action="store_true")
    parser.add_argument("--hand-track-window", type=float, default=0.14)
    preliminary_args, _ = parser.parse_known_args()

    if preliminary_args.config is not None:
        config_defaults = load_analysis_config(preliminary_args.config, parser)
        parser.set_defaults(**config_defaults)

    return parser.parse_args()


def load_analysis_config(config_path: Path, parser: argparse.ArgumentParser) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        raw_config = json.load(file)

    config = raw_config.get("analysis", raw_config)
    valid_dests = {
        action.dest
        for action in parser._actions
        if action.dest not in (None, argparse.SUPPRESS)
    }
    defaults = {}

    for key, value in config.items():
        normalized_key = key.replace("-", "_")

        if normalized_key in valid_dests and normalized_key != "config":
            defaults[normalized_key] = value

    return defaults


def find_latest_session(sync_root: Path) -> Path:
    candidates = sorted(
        [path for path in sync_root.glob("sync_*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(f"No sync sessions found in {sync_root}")

    return candidates[0]


def serializable_analysis_args(args) -> dict:
    values = {}

    for key, value in vars(args).items():
        if isinstance(value, Path):
            values[key] = str(value)
        else:
            values[key] = value

    return values


def load_audio(audio_path: Path) -> tuple[np.ndarray, int]:
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
        audio = audio.reshape(-1, channels).mean(axis=1)

    return audio.reshape(-1), sample_rate


def load_frame_times(session_dir: Path, frame_times_name: str) -> list[float]:
    frame_times_path = session_dir / frame_times_name

    if not frame_times_path.exists():
        return []

    with frame_times_path.open("r", encoding="utf-8") as file:
        raw_frame_times = json.load(file)

    if not raw_frame_times:
        return []

    times_by_index = {}

    for item in raw_frame_times:
        frame_index = item.get("frame_index")
        time_seconds = item.get("time_seconds")

        if frame_index is None or time_seconds is None:
            continue

        times_by_index[int(frame_index)] = float(time_seconds)

    if not times_by_index:
        return []

    max_index = max(times_by_index)
    return [times_by_index.get(index, 0.0) for index in range(max_index + 1)]


def load_hand_tracks(session_dir: Path, hand_tracks_name: str) -> tuple[list[dict], dict]:
    hand_tracks_path = session_dir / hand_tracks_name

    if not hand_tracks_path.exists():
        return [], {"enabled": False, "reason": f"missing {hand_tracks_name}"}

    with hand_tracks_path.open("r", encoding="utf-8") as file:
        hand_tracks = json.load(file)

    frames = sorted(
        hand_tracks.get("frames", []),
        key=lambda frame: float(frame.get("time_seconds", 0.0)),
    )
    frames_with_hands = sum(1 for frame in frames if frame.get("hands"))

    return frames, {
        "enabled": True,
        "hand_tracks_ref": hand_tracks_name,
        "frame_count": len(frames),
        "frames_with_hands": frames_with_hands,
    }


def nearest_hand_track_frame(
    hand_frames: list[dict],
    target_time: float,
    max_delta_seconds: float,
) -> tuple[dict | None, float | None]:
    if not hand_frames:
        return None, None

    times = [float(frame.get("time_seconds", 0.0)) for frame in hand_frames]
    insert_index = bisect_left(times, target_time)
    candidate_indices = []

    if insert_index < len(times):
        candidate_indices.append(insert_index)
    if insert_index > 0:
        candidate_indices.append(insert_index - 1)

    best_frame = None
    best_delta = None

    for index in candidate_indices:
        delta = abs(times[index] - target_time)

        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_frame = hand_frames[index]

    if best_delta is None or best_delta > max_delta_seconds:
        return None, best_delta

    return best_frame, best_delta


def load_av_offset_seconds(session_dir: Path, args) -> tuple[float, dict | None]:
    if args.av_offset_ms is not None:
        return args.av_offset_ms / 1000.0, {
            "source": "cli",
            "audio_to_video_offset_ms": args.av_offset_ms,
        }

    av_sync_path = session_dir / args.av_sync_file

    if not av_sync_path.exists():
        return 0.0, None

    with av_sync_path.open("r", encoding="utf-8") as file:
        av_sync = json.load(file)

    offset_ms = av_sync.get("audio_to_video_offset_ms")

    if offset_ms is None:
        return 0.0, av_sync

    return float(offset_ms) / 1000.0, av_sync


def nearest_frame_index(frame_times: list[float], target_time: float, fps: float) -> int:
    if frame_times:
        return int(np.argmin(np.abs(np.array(frame_times, dtype=np.float32) - target_time)))

    return max(0, int(round(target_time * fps)))


def detect_edge_markers(frame, min_area: float) -> list[dict]:
    marker_points = []

    for color_name, expected_count in EDGE_PAIR_MARKER_COUNTS.items():
        detections = detect_markers_for_color(
            frame,
            color_name=color_name,
            min_area=min_area,
        )
        detections = sorted(
            detections[:expected_count],
            key=lambda detection: (detection.center[1], detection.center[0]),
        )

        for index, detection in enumerate(detections, start=1):
            marker_points.append(
                {
                    "color": color_name,
                    "index": index,
                    "center": detection.center,
                }
            )

    return marker_points


def make_canonical_points(marker_points: list[dict]) -> dict[tuple[str, int], tuple[float, float]]:
    if len(marker_points) < 4:
        return {}

    width, height = WARP_SIZE
    padding = 10.0
    points = np.array(
        [marker_point["center"] for marker_point in marker_points],
        dtype=np.float32,
    )
    centered_points = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered_points, full_matrices=False)
    fret_axis = vh[0]
    green_points = np.array(
        [
            marker_point["center"]
            for marker_point in marker_points
            if marker_point["color"] == "green"
        ],
        dtype=np.float32,
    )

    if len(green_points) and float((green_points @ fret_axis).mean()) > float((points @ fret_axis).mean()):
        fret_axis = -fret_axis

    string_axis = np.array([-fret_axis[1], fret_axis[0]], dtype=np.float32)
    fret_projections = points @ fret_axis
    string_projections = points @ string_axis
    fret_span = float(fret_projections.max() - fret_projections.min())
    string_span = float(string_projections.max() - string_projections.min())

    if fret_span <= 1.0 or string_span <= 1.0:
        return {}

    canonical_points = {}
    usable_width = width - padding * 2
    usable_height = height - padding * 2

    for marker_point in marker_points:
        point = np.array(marker_point["center"], dtype=np.float32)
        x_ratio = (float(point @ fret_axis) - float(fret_projections.min())) / fret_span
        y_ratio = (float(point @ string_axis) - float(string_projections.min())) / string_span
        canonical_points[(marker_point["color"], marker_point["index"])] = (
            padding + max(0.0, min(1.0, x_ratio)) * usable_width,
            padding + max(0.0, min(1.0, y_ratio)) * usable_height,
        )

    return canonical_points


def warp_fretboard_frame(frame, marker_min_area: float):
    marker_points = detect_edge_markers(frame, min_area=marker_min_area)

    if len(marker_points) < 8:
        return None

    canonical_points = make_canonical_points(marker_points)
    source_points = []
    destination_points = []

    for marker_point in marker_points:
        marker_key = (marker_point["color"], marker_point["index"])
        canonical_point = canonical_points.get(marker_key)

        if canonical_point is None:
            continue

        source_points.append(marker_point["center"])
        destination_points.append(canonical_point)

    if len(source_points) < 4:
        return None

    homography, inlier_mask = cv2.findHomography(
        np.array(source_points, dtype=np.float32),
        np.array(destination_points, dtype=np.float32),
        cv2.RANSAC,
        8.0,
    )

    if homography is None:
        return None

    inlier_count = int(inlier_mask.sum()) if inlier_mask is not None else len(source_points)
    warped = cv2.warpPerspective(frame, homography, WARP_SIZE)
    return {
        "warped": warped,
        "homography": homography,
        "marker_count": len(marker_points),
        "inlier_count": inlier_count,
    }


def transform_points_with_homography(points: list[tuple[float, float]], homography) -> np.ndarray:
    if not points:
        return np.empty((0, 2), dtype=np.float32)

    source = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(source, homography)
    return transformed.reshape(-1, 2)


def mediapipe_hand_evidence_from_track(
    hand_frame: dict | None,
    homography,
    time_delta_seconds: float | None = None,
    max_delta_seconds: float = 0.14,
) -> dict:
    empty_evidence = {
        "available": False,
        "center_fret": None,
        "start_fret": None,
        "end_fret": None,
        "confidence": 0.0,
    }

    if hand_frame is None or homography is None:
        return empty_evidence

    width, height = WARP_SIZE
    tip_indices = {4, 8, 12, 16, 20}
    support_indices = {5, 9, 13, 17}
    hand_evidence_items = []

    for hand in hand_frame.get("hands", []):
        landmarks = hand.get("landmarks", [])
        indexed_points = []

        for index, landmark in enumerate(landmarks):
            if "px" not in landmark or "py" not in landmark:
                continue

            indexed_points.append((index, (float(landmark["px"]), float(landmark["py"]))))

        if not indexed_points:
            continue

        transformed_points = transform_points_with_homography(
            [point for _, point in indexed_points],
            homography,
        )
        points_by_index = {
            landmark_index: transformed_points[index]
            for index, (landmark_index, _) in enumerate(indexed_points)
        }
        in_fretboard_points = [
            (landmark_index, point)
            for landmark_index, point in points_by_index.items()
            if -width * 0.12 <= point[0] <= width * 1.12
            and -height * 0.25 <= point[1] <= height * 1.25
        ]
        preferred_points = [
            point
            for landmark_index, point in in_fretboard_points
            if landmark_index in tip_indices
        ]

        if not preferred_points:
            preferred_points = [
                point
                for landmark_index, point in in_fretboard_points
                if landmark_index in support_indices
            ]

        if not preferred_points:
            preferred_points = [point for _, point in in_fretboard_points]

        if not preferred_points:
            continue

        x_values = np.array([point[0] for point in preferred_points], dtype=np.float32)
        score = hand.get("score")
        confidence = float(score) if score is not None else 0.7
        hand_evidence_items.append(
            {
                "center_x": float(np.median(x_values)),
                "start_x": float(np.min(x_values)),
                "end_x": float(np.max(x_values)),
                "confidence": max(0.0, min(1.0, confidence)),
                "landmark_count": len(preferred_points),
            }
        )

    if not hand_evidence_items:
        return empty_evidence

    best_item = max(
        hand_evidence_items,
        key=lambda item: (item["confidence"], item["landmark_count"]),
    )
    delta_confidence = 1.0

    if time_delta_seconds is not None and max_delta_seconds > 0:
        delta_confidence = max(0.0, 1.0 - time_delta_seconds / max_delta_seconds)

    confidence = best_item["confidence"] * delta_confidence
    center_ratio = max(0.0, min(0.999, best_item["center_x"] / max(1, width)))
    start_ratio = max(0.0, min(0.999, best_item["start_x"] / max(1, width)))
    end_ratio = max(0.0, min(0.999, best_item["end_x"] / max(1, width)))

    return {
        "available": True,
        "center_fret": 1 + int(center_ratio * 12),
        "start_fret": 1 + int(start_ratio * 12),
        "end_fret": 1 + int(end_ratio * 12),
        "confidence": round(confidence, 4),
        "frame_index": hand_frame.get("frame_index"),
        "time_seconds": hand_frame.get("time_seconds"),
        "time_delta_seconds": round(float(time_delta_seconds), 4)
        if time_delta_seconds is not None
        else None,
        "source": "mediapipe_hands",
    }


def skin_or_change_mask(warped_frame, diff_frame):
    hsv_frame = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2HSV)
    skin_mask_a = cv2.inRange(hsv_frame, np.array((0, 25, 40)), np.array((28, 190, 255)))
    skin_mask_b = cv2.inRange(hsv_frame, np.array((160, 20, 40)), np.array((179, 180, 255)))
    skin_mask = cv2.bitwise_or(skin_mask_a, skin_mask_b)
    _, diff_mask = cv2.threshold(diff_frame, 18, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_or(skin_mask, diff_mask)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []

    max_value = max(values)

    if max_value <= 1e-9:
        return [0.0 for _ in values]

    return [float(value / max_value) for value in values]


def make_rms_envelope(
    audio: np.ndarray,
    frame_size: int,
    hop_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(audio) < frame_size:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int64)

    starts = np.arange(0, len(audio) - frame_size + 1, hop_size, dtype=np.int64)
    envelope = []

    for start in starts:
        frame = audio[start : start + frame_size]
        envelope.append(float(np.sqrt(np.mean(frame * frame))))

    return np.array(envelope, dtype=np.float32), starts


def robust_zscore(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1e-6, mad * 1.4826)
    return (values - median) / scale


def detect_onsets(
    audio: np.ndarray,
    sample_rate: int,
    frame_size: int,
    hop_size: int,
    threshold: float,
    min_gap_seconds: float,
) -> list[float]:
    envelope, starts = make_rms_envelope(audio, frame_size=frame_size, hop_size=hop_size)

    if len(envelope) < 3:
        return []

    flux = np.maximum(0.0, envelope[1:] - envelope[:-1])
    zscores = robust_zscore(flux)
    candidate_indices = np.where(zscores >= threshold)[0] + 1
    onsets = []
    last_onset = -999.0
    last_onset_index = None

    for index in candidate_indices:
        if index <= 0 or index >= len(envelope) - 1:
            continue

        if envelope[index] < max(0.004, float(np.percentile(envelope, 65))):
            continue

        onset_time = float(starts[index] / sample_rate)

        if onset_time - last_onset < min_gap_seconds:
            if (
                onsets
                and last_onset_index is not None
                and envelope[index] > envelope[last_onset_index]
            ):
                onsets[-1] = onset_time
                last_onset = onset_time
                last_onset_index = index
            continue

        onsets.append(onset_time)
        last_onset = onset_time
        last_onset_index = index

    return onsets


def sample_range(
    start_seconds: float,
    end_seconds: float,
    sample_rate: int,
    sample_count: int,
) -> tuple[int, int]:
    start_index = max(0, min(sample_count, int(round(start_seconds * sample_rate))))
    end_index = max(start_index, min(sample_count, int(round(end_seconds * sample_rate))))
    return start_index, end_index


def detect_note_pitch(
    audio: np.ndarray,
    sample_rate: int,
    onset: float,
    offset: float,
    pitch_skip: float,
    pitch_window: float,
    rms_threshold: float,
    min_frequency: float,
    max_frequency: float,
):
    start = onset + pitch_skip
    end = min(offset, start + pitch_window)
    start_index, end_index = sample_range(start, end, sample_rate, len(audio))

    if end_index - start_index < int(sample_rate * 0.04):
        start_index, end_index = sample_range(onset, offset, sample_rate, len(audio))

    return detect_pitch_from_samples(
        audio[start_index:end_index],
        sample_rate=sample_rate,
        min_frequency_hz=min_frequency,
        max_frequency_hz=max_frequency,
        rms_threshold=rms_threshold,
        confidence_threshold=0.14,
    )


def make_f0_curve(
    audio: np.ndarray,
    sample_rate: int,
    onset: float,
    offset: float,
    rms_threshold: float,
    min_frequency: float,
    max_frequency: float,
) -> list[dict]:
    window_seconds = 0.11
    hop_seconds = 0.055
    curve = []
    current = onset

    while current + window_seconds <= offset:
        start_index, end_index = sample_range(
            current,
            current + window_seconds,
            sample_rate,
            len(audio),
        )
        detection = detect_pitch_from_samples(
            audio[start_index:end_index],
            sample_rate=sample_rate,
            min_frequency_hz=min_frequency,
            max_frequency_hz=max_frequency,
            rms_threshold=rms_threshold,
            confidence_threshold=0.12,
        )

        if detection is not None:
            curve.append(
                {
                    "t": round(current - onset, 4),
                    "frequency_hz": round(detection.frequency_hz, 3),
                    "midi_pitch": detection.midi_pitch,
                    "confidence": round(detection.confidence, 4),
                }
            )

        current += hop_seconds

    return curve


def note_velocity(audio: np.ndarray, sample_rate: int, onset: float, offset: float) -> float:
    start_index, end_index = sample_range(onset, min(offset, onset + 0.12), sample_rate, len(audio))

    if end_index <= start_index:
        return 0.0

    peak = float(np.max(np.abs(audio[start_index:end_index])))
    return round(max(0.0, min(1.0, peak)), 4)


def visual_evidence_from_warped_frames(baseline_warped, onset_warped) -> dict:
    gray_baseline = cv2.cvtColor(baseline_warped, cv2.COLOR_BGR2GRAY)
    gray_onset = cv2.cvtColor(onset_warped, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(
        cv2.GaussianBlur(gray_baseline, (5, 5), 0),
        cv2.GaussianBlur(gray_onset, (5, 5), 0),
    )
    height, width = diff.shape[:2]
    raw_string_scores = []

    for string_number in STRING_ORDER_TOP_TO_BOTTOM:
        string_index = STRING_ORDER_TOP_TO_BOTTOM.index(string_number)
        center_y = int((string_index + 0.5) / 6.0 * height)
        band_half_height = max(4, int(height / 42))
        top = max(0, center_y - band_half_height)
        bottom = min(height, center_y + band_half_height + 1)
        band = diff[top:bottom, :]
        raw_string_scores.append(float(np.percentile(band, 92)))

    normalized_string_scores = normalize_scores(raw_string_scores)
    string_scores = {
        string_number: normalized_string_scores[index]
        for index, string_number in enumerate(STRING_ORDER_TOP_TO_BOTTOM)
    }
    mask = skin_or_change_mask(onset_warped, diff)
    changed_pixels = int(cv2.countNonZero(mask))
    hand_center_fret = None
    hand_start_fret = None
    hand_end_fret = None
    hand_confidence = 0.0

    if changed_pixels >= 80:
        nonzero_points = cv2.findNonZero(mask)

        if nonzero_points is not None:
            x, _, rect_width, _ = cv2.boundingRect(nonzero_points)
            center_x = x + rect_width / 2.0
            left_ratio = max(0.0, min(0.999, x / max(1, width)))
            center_ratio = max(0.0, min(0.999, center_x / max(1, width)))
            right_ratio = max(0.0, min(0.999, (x + rect_width) / max(1, width)))
            hand_center_fret = 1 + int(center_ratio * 12)
            hand_start_fret = 1 + int(left_ratio * 12)
            hand_end_fret = 1 + int(right_ratio * 12)
            hand_confidence = min(1.0, changed_pixels / float(width * height * 0.16))

    return {
        "string_scores": string_scores,
        "hand_center_fret": hand_center_fret,
        "hand_start_fret": hand_start_fret,
        "hand_end_fret": hand_end_fret,
        "hand_confidence": round(hand_confidence, 4),
        "changed_pixels": changed_pixels,
    }


def candidate_hand_score(candidate: dict, hand_center_fret: int | None) -> float:
    if hand_center_fret is None:
        return 0.0

    distance = abs(candidate["fret_number"] - hand_center_fret)
    return float(np.exp(-0.5 * (distance / 2.1) ** 2))


def apply_visual_evidence_to_note(note: dict, evidence: dict) -> None:
    string_scores = evidence["string_scores"]
    hand_center_fret = evidence["hand_center_fret"]
    mediapipe_hand = evidence.get("mediapipe_hand", {})
    mediapipe_hand_center_fret = mediapipe_hand.get("center_fret")
    mediapipe_hand_confidence = float(mediapipe_hand.get("confidence", 0.0))

    for candidate in note["candidates"]:
        string_score = float(string_scores.get(candidate["string_number"], 0.0))
        hand_score = candidate_hand_score(candidate, hand_center_fret)
        mediapipe_hand_score = candidate_hand_score(candidate, mediapipe_hand_center_fret)

        if mediapipe_hand_center_fret is None:
            visual_score = 0.62 * string_score + 0.38 * hand_score * evidence["hand_confidence"]
        else:
            visual_score = (
                0.52 * string_score
                + 0.2 * hand_score * evidence["hand_confidence"]
                + 0.28 * mediapipe_hand_score * mediapipe_hand_confidence
            )

        candidate["string_visual_score"] = round(string_score, 4)
        candidate["hand_visual_score"] = round(hand_score, 4)
        candidate["mediapipe_hand_visual_score"] = round(mediapipe_hand_score, 4)
        candidate["visual_score"] = round(visual_score, 4)

    note["visual"] = {
        "available": True,
        "string_scores": {
            str(string_number): round(score, 4)
            for string_number, score in string_scores.items()
        },
        "hand_center_fret": hand_center_fret,
        "hand_start_fret": evidence["hand_start_fret"],
        "hand_end_fret": evidence["hand_end_fret"],
        "hand_confidence": evidence["hand_confidence"],
        "mediapipe_hand_center_fret": mediapipe_hand_center_fret,
        "mediapipe_hand_start_fret": mediapipe_hand.get("start_fret"),
        "mediapipe_hand_end_fret": mediapipe_hand.get("end_fret"),
        "mediapipe_hand_confidence": mediapipe_hand_confidence,
        "mediapipe_hand_frame_index": mediapipe_hand.get("frame_index"),
        "mediapipe_hand_time_delta_seconds": mediapipe_hand.get("time_delta_seconds"),
        "changed_pixels": evidence["changed_pixels"],
    }


def attach_video_evidence(session_dir: Path, notes: list[dict], args) -> dict:
    if args.no_video_analysis:
        return {"enabled": False, "reason": "disabled"}

    video_path = session_dir / args.video_file

    if not video_path.exists():
        return {"enabled": False, "reason": f"missing {args.video_file}"}

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        return {"enabled": False, "reason": f"could not open {args.video_file}"}

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_times = load_frame_times(session_dir, args.frame_times)
    av_offset_seconds, av_sync = load_av_offset_seconds(session_dir, args)
    frame_cache: dict[int, np.ndarray | None] = {}
    hand_frames = []
    hand_track_summary = {"enabled": False, "reason": "disabled"}
    analyzed_count = 0
    visual_count = 0
    mediapipe_hand_note_count = 0

    if not args.no_hand_tracks:
        hand_frames, hand_track_summary = load_hand_tracks(session_dir, args.hand_tracks)

    def read_frame(frame_index: int):
        if frame_index in frame_cache:
            return frame_cache[frame_index]

        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
        ok, frame = capture.read()
        frame_cache[frame_index] = frame if ok else None
        return frame_cache[frame_index]

    try:
        for note in notes:
            video_onset_time = note["t_onset"] + av_offset_seconds
            baseline_time = max(0.0, video_onset_time - args.video_baseline_offset)
            onset_time = video_onset_time + args.video_onset_offset
            baseline_index = nearest_frame_index(frame_times, baseline_time, fps)
            onset_index = nearest_frame_index(frame_times, onset_time, fps)
            baseline_frame = read_frame(baseline_index)
            onset_frame = read_frame(onset_index)

            if baseline_frame is None or onset_frame is None:
                continue

            analyzed_count += 1
            baseline_warp = warp_fretboard_frame(
                baseline_frame,
                marker_min_area=args.video_marker_min_area,
            )
            onset_warp = warp_fretboard_frame(
                onset_frame,
                marker_min_area=args.video_marker_min_area,
            )

            if baseline_warp is None or onset_warp is None:
                note["visual"] = {
                    "available": False,
                    "reason": "marker_warp_failed",
                }
                continue

            evidence = visual_evidence_from_warped_frames(
                baseline_warp["warped"],
                onset_warp["warped"],
            )
            hand_track_frame, hand_track_delta = nearest_hand_track_frame(
                hand_frames,
                onset_time,
                args.hand_track_window,
            )
            mediapipe_hand_evidence = mediapipe_hand_evidence_from_track(
                hand_track_frame,
                onset_warp["homography"],
                time_delta_seconds=hand_track_delta,
                max_delta_seconds=args.hand_track_window,
            )
            evidence["baseline_frame_index"] = baseline_index
            evidence["onset_frame_index"] = onset_index
            evidence["marker_count"] = min(
                baseline_warp["marker_count"],
                onset_warp["marker_count"],
            )
            evidence["marker_inliers"] = min(
                baseline_warp["inlier_count"],
                onset_warp["inlier_count"],
            )
            evidence["mediapipe_hand"] = mediapipe_hand_evidence
            apply_visual_evidence_to_note(note, evidence)
            note["visual"]["baseline_frame_index"] = baseline_index
            note["visual"]["onset_frame_index"] = onset_index
            note["visual"]["marker_count"] = evidence["marker_count"]
            note["visual"]["marker_inliers"] = evidence["marker_inliers"]
            if mediapipe_hand_evidence.get("available"):
                mediapipe_hand_note_count += 1
            visual_count += 1
    finally:
        capture.release()

    return {
        "enabled": True,
        "video_ref": args.video_file,
        "frame_times_ref": args.frame_times if frame_times else None,
        "av_sync_ref": args.av_sync_file if av_sync is not None else None,
        "audio_to_video_offset_ms": round(av_offset_seconds * 1000.0, 3),
        "fps": fps,
        "analyzed_note_count": analyzed_count,
        "visual_note_count": visual_count,
        "mediapipe_hand_note_count": mediapipe_hand_note_count,
        "hand_tracks": hand_track_summary,
    }


def transition_cost(previous: dict, current: dict, gap_seconds: float) -> float:
    fret_distance = abs(current["fret_number"] - previous["fret_number"])
    string_distance = abs(current["string_number"] - previous["string_number"])
    movement = fret_distance * 0.42 + string_distance * 0.72

    if fret_distance >= 7:
        movement += 0.9

    if string_distance >= 3:
        movement += 0.45

    if gap_seconds > 0.55:
        movement *= 0.35
    elif gap_seconds > 0.28:
        movement *= 0.65

    return movement


def candidate_prior(candidate: dict) -> float:
    fret = candidate["fret_number"]

    if fret == 0:
        return 0.05

    if 1 <= fret <= 12:
        return 0.0

    return 0.12 + (fret - 12) * 0.02


def choose_positions_with_viterbi(note_items: list[dict], visual_weight: float = 1.1) -> list[dict]:
    if not note_items:
        return []

    costs: list[list[float]] = []
    backrefs: list[list[int | None]] = []

    for note_index, note in enumerate(note_items):
        candidates = note["candidates"]

        if not candidates:
            costs.append([])
            backrefs.append([])
            continue

        current_costs = []
        current_backrefs = []

        for candidate in candidates:
            visual_score = float(candidate.get("visual_score", 0.0))
            emission_cost = (
                candidate_prior(candidate)
                - note["pitch_confidence"] * 0.18
                - visual_score * visual_weight
            )

            if note_index == 0 or not costs[note_index - 1]:
                current_costs.append(emission_cost)
                current_backrefs.append(None)
                continue

            previous_note = note_items[note_index - 1]
            gap_seconds = max(0.0, note["t_onset"] - previous_note["t_offset"])
            best_previous_index = 0
            best_cost = float("inf")

            for previous_index, previous_candidate in enumerate(previous_note["candidates"]):
                cost = (
                    costs[note_index - 1][previous_index]
                    + transition_cost(previous_candidate, candidate, gap_seconds)
                    + emission_cost
                )

                if cost < best_cost:
                    best_cost = cost
                    best_previous_index = previous_index

            current_costs.append(best_cost)
            current_backrefs.append(best_previous_index)

        costs.append(current_costs)
        backrefs.append(current_backrefs)

    selected_indices = [None] * len(note_items)
    last_valid_index = None

    for index in range(len(note_items) - 1, -1, -1):
        if costs[index]:
            last_valid_index = index
            break

    if last_valid_index is None:
        for note in note_items:
            note["string"] = None
            note["fret"] = None
            note["confidence"] = 0.0
            note["status"] = "no_pitch_candidate"
        return note_items

    selected_indices[last_valid_index] = int(np.argmin(costs[last_valid_index]))

    for index in range(last_valid_index, 0, -1):
        current_selection = selected_indices[index]

        if current_selection is None:
            continue

        previous_index = backrefs[index][current_selection]

        if previous_index is not None:
            selected_indices[index - 1] = previous_index

    for index, note in enumerate(note_items):
        candidates = note["candidates"]

        if not candidates:
            note["string"] = None
            note["fret"] = None
            note["confidence"] = 0.0
            note["status"] = "no_pitch_candidate"
            continue

        selected_index = selected_indices[index]

        if selected_index is None:
            selected_index = int(np.argmin([candidate_prior(candidate) for candidate in candidates]))

        selected_candidate = candidates[selected_index]
        sorted_costs = sorted(costs[index]) if costs[index] else []
        score_gap = (
            sorted_costs[1] - sorted_costs[0]
            if len(sorted_costs) >= 2
            else 3.0
        )
        ambiguity_penalty = 0.22 if len(candidates) > 1 else 0.0
        confidence = max(
            0.05,
            min(
                0.99,
                0.42
                + note["pitch_confidence"] * 0.38
                + float(selected_candidate.get("visual_score", 0.0)) * 0.18
                + min(0.25, score_gap * 0.12)
                - ambiguity_penalty,
            ),
        )
        note["string"] = selected_candidate["string_number"]
        note["fret"] = selected_candidate["fret_number"]
        note["selected_candidate_index"] = selected_index
        note["confidence"] = round(confidence, 4)
        note["status"] = (
            "single_candidate"
            if len(candidates) == 1
            else "visual_viterbi_ambiguous_candidate"
            if any("visual_score" in candidate for candidate in candidates) and confidence < 0.62
            else "visual_viterbi_selected"
            if any("visual_score" in candidate for candidate in candidates)
            else "viterbi_ambiguous_candidate"
            if confidence < 0.62
            else "viterbi_selected"
        )

    return note_items


def build_notes(
    audio: np.ndarray,
    sample_rate: int,
    onsets: list[float],
    args,
) -> list[dict]:
    notes = []
    audio_duration = len(audio) / sample_rate

    for index, onset in enumerate(onsets):
        next_onset = onsets[index + 1] if index + 1 < len(onsets) else audio_duration
        offset = max(onset + args.min_note_duration, min(next_onset, onset + 1.25))
        detection = detect_note_pitch(
            audio=audio,
            sample_rate=sample_rate,
            onset=onset,
            offset=offset,
            pitch_skip=args.pitch_skip,
            pitch_window=args.pitch_window,
            rms_threshold=args.rms_threshold,
            min_frequency=args.min_frequency,
            max_frequency=args.max_frequency,
        )

        if detection is None:
            continue

        candidates = find_positions_for_midi_pitch(
            midi_pitch=detection.midi_pitch,
            start_fret=0,
            visible_frets=args.max_fret + 1,
        )
        notes.append(
            {
                "t_onset": round(onset, 4),
                "t_offset": round(offset, 4),
                "midi_pitch": detection.midi_pitch,
                "note_name": detection.note_name,
                "frequency_hz": round(detection.frequency_hz, 3),
                "pitch_confidence": round(detection.confidence, 4),
                "velocity": note_velocity(audio, sample_rate, onset, offset),
                "candidate_count": len(candidates),
                "candidates": candidates,
                "f0_curve": make_f0_curve(
                    audio=audio,
                    sample_rate=sample_rate,
                    onset=onset,
                    offset=offset,
                    rms_threshold=args.rms_threshold,
                    min_frequency=args.min_frequency,
                    max_frequency=args.max_frequency,
                ),
                "corrected": False,
            }
        )

    return notes


def run_offline_analyze() -> None:
    args = parse_args()
    session_dir = args.session_dir

    if args.latest or session_dir is None:
        session_dir = find_latest_session(args.sync_root)

    audio_path = session_dir / "audio.wav"

    if not audio_path.exists():
        raise FileNotFoundError(f"audio.wav not found: {audio_path}")

    audio, sample_rate = load_audio(audio_path)
    onsets = detect_onsets(
        audio=audio,
        sample_rate=sample_rate,
        frame_size=args.frame_size,
        hop_size=args.hop_size,
        threshold=args.onset_threshold,
        min_gap_seconds=args.min_note_gap,
    )
    notes = build_notes(audio=audio, sample_rate=sample_rate, onsets=onsets, args=args)
    visual_summary = attach_video_evidence(session_dir=session_dir, notes=notes, args=args)
    notes = choose_positions_with_viterbi(notes, visual_weight=args.visual_weight)
    output = {
        "format_version": "0.1",
        "session": {
            "session_dir": str(session_dir),
            "audio_ref": "audio.wav",
            "sample_rate": sample_rate,
            "analysis": "audio_video_onset_f0_viterbi_v0",
            "tuning": ["E2", "A2", "D3", "G3", "B3", "E4"],
            "capo": 0,
        },
        "summary": {
            "onset_count": len(onsets),
            "note_count": len(notes),
            "ambiguous_note_count": sum(1 for note in notes if note["candidate_count"] > 1),
            "low_confidence_count": sum(1 for note in notes if note.get("confidence", 0.0) < 0.62),
            "visual_note_count": sum(
                1
                for note in notes
                if note.get("visual", {}).get("available")
            ),
        },
        "visual_analysis": visual_summary,
        "analysis_config": serializable_analysis_args(args),
        "notes": notes,
    }
    output_path = session_dir / args.output

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print(f"offline analysis written: {output_path}")
    print(
        "summary: "
        f"onsets={output['summary']['onset_count']} "
        f"notes={output['summary']['note_count']} "
        f"ambiguous={output['summary']['ambiguous_note_count']} "
        f"low_conf={output['summary']['low_confidence_count']} "
        f"visual={output['summary']['visual_note_count']}"
    )

    for note in notes[:12]:
        selected = (
            f"{note['string']}s/{note['fret']}F"
            if note.get("string") is not None
            else "-"
        )
        print(
            f"{note['t_onset']:7.3f}s {note['note_name']:>3} "
            f"{selected:>7} candidates={note['candidate_count']} "
            f"conf={note.get('confidence', 0.0):.2f} {note['status']}"
        )


if __name__ == "__main__":
    run_offline_analyze()
