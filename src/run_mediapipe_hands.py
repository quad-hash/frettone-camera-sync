import argparse
import json
import urllib.request
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYNC_OUTPUT_DIR = PROJECT_ROOT / "data" / "sync_sessions"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "hand_landmarker.task"
DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run offline MediaPipe Hands tracking for a recorded FretLog video.",
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
    parser.add_argument("--frame-times", default="frame_times.json")
    parser.add_argument("--output", default="hand_tracks.json")
    parser.add_argument("--sample-step", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-num-hands", type=int, default=2)
    parser.add_argument("--model-complexity", type=int, default=1)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-url", default=DEFAULT_MODEL_URL)
    parser.add_argument(
        "--no-download-model",
        action="store_true",
        help="Do not download hand_landmarker.task automatically for the new MediaPipe Tasks API.",
    )
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


def load_frame_times(session_dir: Path, frame_times_name: str) -> list[float]:
    frame_times_path = session_dir / frame_times_name

    if not frame_times_path.exists():
        return []

    with frame_times_path.open("r", encoding="utf-8") as file:
        raw_frame_times = json.load(file)

    frame_times = []
    for item in raw_frame_times:
        if isinstance(item, dict):
            frame_times.append(float(item.get("time_seconds", 0.0)))
        else:
            frame_times.append(float(item))

    return frame_times


def frame_time_for_index(frame_index: int, frame_times: list[float], fps: float) -> float:
    if 0 <= frame_index < len(frame_times):
        return float(frame_times[frame_index])

    if fps > 0:
        return frame_index / fps

    return 0.0


def import_mediapipe():
    try:
        import mediapipe as mp
    except ImportError as error:
        raise SystemExit(
            "MediaPipe is not installed. Install it with: pip install mediapipe"
        ) from error

    return mp


def ensure_task_model(model_path: Path, model_url: str, no_download_model: bool) -> Path:
    if model_path.exists():
        return model_path

    if no_download_model:
        raise SystemExit(
            f"MediaPipe task model not found: {model_path}\n"
            "Download hand_landmarker.task and pass it with --model-path."
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading MediaPipe hand model: {model_url}")

    try:
        urllib.request.urlretrieve(model_url, model_path)
    except Exception as error:
        raise SystemExit(
            f"Could not download MediaPipe hand model: {error}\n"
            f"Download it manually to: {model_path}\n"
            f"URL: {model_url}"
        ) from error

    return model_path


def landmark_to_dict(landmark, width: int, height: int) -> dict:
    return {
        "x": round(float(landmark.x), 6),
        "y": round(float(landmark.y), 6),
        "z": round(float(landmark.z), 6),
        "px": round(float(landmark.x) * width, 2),
        "py": round(float(landmark.y) * height, 2),
    }


def hand_to_dict(hand_landmarks, handedness, width: int, height: int) -> dict:
    landmarks = [
        landmark_to_dict(landmark, width, height)
        for landmark in hand_landmarks.landmark
    ]
    xs = [landmark["px"] for landmark in landmarks]
    ys = [landmark["py"] for landmark in landmarks]
    classification = handedness.classification[0] if handedness.classification else None

    return {
        "label": classification.label if classification is not None else None,
        "score": round(float(classification.score), 4)
        if classification is not None
        else None,
        "center": {
            "px": round(sum(xs) / len(xs), 2) if xs else None,
            "py": round(sum(ys) / len(ys), 2) if ys else None,
        },
        "bbox": {
            "x_min": round(min(xs), 2) if xs else None,
            "y_min": round(min(ys), 2) if ys else None,
            "x_max": round(max(xs), 2) if xs else None,
            "y_max": round(max(ys), 2) if ys else None,
        },
        "landmarks": landmarks,
    }


def task_hand_to_dict(hand_landmarks, handedness, width: int, height: int) -> dict:
    landmarks = [
        landmark_to_dict(landmark, width, height)
        for landmark in hand_landmarks
    ]
    xs = [landmark["px"] for landmark in landmarks]
    ys = [landmark["py"] for landmark in landmarks]
    classification = handedness[0] if handedness else None

    return {
        "label": classification.category_name if classification is not None else None,
        "score": round(float(classification.score), 4)
        if classification is not None
        else None,
        "center": {
            "px": round(sum(xs) / len(xs), 2) if xs else None,
            "py": round(sum(ys) / len(ys), 2) if ys else None,
        },
        "bbox": {
            "x_min": round(min(xs), 2) if xs else None,
            "y_min": round(min(ys), 2) if ys else None,
            "x_max": round(max(xs), 2) if xs else None,
            "y_max": round(max(ys), 2) if ys else None,
        },
        "landmarks": landmarks,
    }


class SolutionsHandsAdapter:
    def __init__(self, mp, args) -> None:
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=args.max_num_hands,
            model_complexity=args.model_complexity,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )

    def process(self, rgb_frame, frame_time_seconds: float, width: int, height: int) -> list[dict]:
        result = self.hands.process(rgb_frame)

        if not result.multi_hand_landmarks or not result.multi_handedness:
            return []

        return [
            hand_to_dict(hand_landmarks, handedness, width, height)
            for hand_landmarks, handedness in zip(
                result.multi_hand_landmarks,
                result.multi_handedness,
            )
        ]

    def close(self) -> None:
        self.hands.close()


class TasksHandLandmarkerAdapter:
    def __init__(self, mp, args) -> None:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        model_path = ensure_task_model(
            args.model_path,
            args.model_url,
            args.no_download_model,
        )
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=args.max_num_hands,
            min_hand_detection_confidence=args.min_detection_confidence,
            min_hand_presence_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )
        self.mp = mp
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.last_timestamp_ms = -1

    def process(self, rgb_frame, frame_time_seconds: float, width: int, height: int) -> list[dict]:
        timestamp_ms = max(
            self.last_timestamp_ms + 1,
            int(round(frame_time_seconds * 1000.0)),
        )
        self.last_timestamp_ms = timestamp_ms
        image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB,
            data=rgb_frame,
        )
        result = self.landmarker.detect_for_video(image, timestamp_ms)

        if not result.hand_landmarks or not result.handedness:
            return []

        return [
            task_hand_to_dict(hand_landmarks, handedness, width, height)
            for hand_landmarks, handedness in zip(
                result.hand_landmarks,
                result.handedness,
            )
        ]

    def close(self) -> None:
        self.landmarker.close()


def create_hand_detector(mp, args):
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        print("MediaPipe backend: solutions.hands")
        return SolutionsHandsAdapter(mp, args)

    if hasattr(mp, "tasks"):
        print("MediaPipe backend: tasks HandLandmarker")
        return TasksHandLandmarkerAdapter(mp, args)

    raise SystemExit(
        "Unsupported MediaPipe package: neither mp.solutions.hands nor mp.tasks is available."
    )


def run_mediapipe_hands() -> None:
    args = parse_args()
    mp = import_mediapipe()

    if args.latest or args.session_dir is None:
        session_dir = find_latest_session(args.sync_root)
    else:
        session_dir = args.session_dir

    video_path = session_dir / args.video_file
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    frame_times = load_frame_times(session_dir, args.frame_times)
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    output_frames = []
    processed_count = 0
    frames_with_hands = 0

    hand_detector = create_hand_detector(mp, args)

    try:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            should_process = frame_index % max(1, args.sample_step) == 0
            if should_process:
                height, width = frame.shape[:2]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_time_seconds = frame_time_for_index(frame_index, frame_times, fps)
                hand_items = hand_detector.process(
                    rgb_frame,
                    frame_time_seconds=frame_time_seconds,
                    width=width,
                    height=height,
                )

                if hand_items:
                    frames_with_hands += 1

                output_frames.append(
                    {
                        "frame_index": frame_index,
                        "time_seconds": round(
                            frame_time_seconds,
                            6,
                        ),
                        "hands": hand_items,
                    }
                )
                processed_count += 1

                if args.max_frames > 0 and processed_count >= args.max_frames:
                    break

            frame_index += 1
    finally:
        hand_detector.close()
        capture.release()

    output = {
        "format_version": "0.1",
        "source": {
            "session_dir": str(session_dir),
            "video_ref": args.video_file,
            "frame_times_ref": args.frame_times if frame_times else None,
            "sample_step": max(1, args.sample_step),
        },
        "summary": {
            "processed_frame_count": processed_count,
            "frames_with_hands": frames_with_hands,
            "fps": fps,
        },
        "frames": output_frames,
    }

    output_path = session_dir / args.output
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print(f"MediaPipe Hands complete: {output_path}")
    print(f"processed={processed_count} frames_with_hands={frames_with_hands}")


if __name__ == "__main__":
    run_mediapipe_hands()
