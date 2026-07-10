import argparse
import json
import wave
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYNC_OUTPUT_DIR = PROJECT_ROOT / "data" / "sync_sessions"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate audio/video offset from a clap in a sync session.",
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
    parser.add_argument("--audio-file", default="audio.wav")
    parser.add_argument("--video-file", default="video.mp4")
    parser.add_argument("--frame-times", default="frame_times.json")
    parser.add_argument("--output", default="av_sync.json")
    parser.add_argument("--scan-start", type=float, default=0.0)
    parser.add_argument("--scan-duration", type=float, default=8.0)
    parser.add_argument("--audio-frame-size", type=int, default=2048)
    parser.add_argument("--audio-hop-size", type=int, default=512)
    parser.add_argument("--update-meta", action="store_true")
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

    times_by_index = {}

    for item in raw_frame_times:
        frame_index = item.get("frame_index")
        time_seconds = item.get("time_seconds")

        if frame_index is None or time_seconds is None:
            continue

        times_by_index[int(frame_index)] = float(time_seconds)

    if not times_by_index:
        return []

    return [times_by_index.get(index, 0.0) for index in range(max(times_by_index) + 1)]


def robust_peak_confidence(values: np.ndarray, peak_index: int) -> float:
    if len(values) == 0:
        return 0.0

    baseline = float(np.median(values))
    mad = float(np.median(np.abs(values - baseline)))
    scale = max(1e-8, mad * 1.4826)
    return float(max(0.0, (values[peak_index] - baseline) / scale))


def pick_peak_time(values: np.ndarray, times: np.ndarray) -> dict | None:
    if len(values) == 0 or len(times) == 0:
        return None

    peak_index = int(np.argmax(values))
    return {
        "time_seconds": float(times[peak_index]),
        "peak_value": float(values[peak_index]),
        "confidence": robust_peak_confidence(values, peak_index),
        "peak_index": peak_index,
    }


def estimate_audio_clap(
    audio: np.ndarray,
    sample_rate: int,
    scan_start: float,
    scan_duration: float,
    frame_size: int,
    hop_size: int,
) -> dict | None:
    start_sample = max(0, int(round(scan_start * sample_rate)))
    end_sample = min(len(audio), int(round((scan_start + scan_duration) * sample_rate)))

    if end_sample - start_sample < frame_size:
        return None

    scan_audio = audio[start_sample:end_sample]
    starts = np.arange(0, len(scan_audio) - frame_size + 1, hop_size, dtype=np.int64)
    peaks = []

    for start in starts:
        frame = scan_audio[start : start + frame_size]
        peaks.append(float(np.max(np.abs(frame))))

    times = (starts + start_sample) / sample_rate
    result = pick_peak_time(np.array(peaks, dtype=np.float32), times.astype(np.float32))

    if result is not None:
        result["method"] = "audio_peak"

    return result


def frame_time_for_index(frame_index: int, frame_times: list[float], fps: float) -> float:
    if 0 <= frame_index < len(frame_times):
        return float(frame_times[frame_index])

    return frame_index / max(1e-6, fps)


def estimate_video_clap(
    video_path: Path,
    frame_times: list[float],
    scan_start: float,
    scan_duration: float,
) -> dict | None:
    if not video_path.exists():
        return None

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        return None

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    start_frame = max(0, int(round(scan_start * fps)))
    end_frame = max(start_frame + 2, int(round((scan_start + scan_duration) * fps)))
    motion_values = []
    motion_times = []

    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, previous_frame = capture.read()

        if not ok:
            return None

        previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        frame_index = start_frame + 1

        while frame_index <= end_frame:
            ok, frame = capture.read()

            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(
                cv2.GaussianBlur(previous_gray, (5, 5), 0),
                cv2.GaussianBlur(gray, (5, 5), 0),
            )
            motion_values.append(float(np.percentile(diff, 97)))
            motion_times.append(frame_time_for_index(frame_index, frame_times, fps))
            previous_gray = gray
            frame_index += 1
    finally:
        capture.release()

    result = pick_peak_time(
        np.array(motion_values, dtype=np.float32),
        np.array(motion_times, dtype=np.float32),
    )

    if result is not None:
        result["method"] = "video_frame_difference"
        result["fps"] = fps

    return result


def update_meta(session_dir: Path, sync_result: dict) -> None:
    meta_path = session_dir / "meta.json"

    if not meta_path.exists():
        return

    with meta_path.open("r", encoding="utf-8") as file:
        meta = json.load(file)

    meta["av_sync"] = {
        "ref": "av_sync.json",
        "audio_to_video_offset_ms": sync_result.get("audio_to_video_offset_ms"),
    }

    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2, ensure_ascii=False)


def run_estimate_av_sync() -> None:
    args = parse_args()
    session_dir = args.session_dir

    if args.latest or session_dir is None:
        session_dir = find_latest_session(args.sync_root)

    audio, sample_rate = load_audio(session_dir / args.audio_file)
    frame_times = load_frame_times(session_dir, args.frame_times)
    audio_clap = estimate_audio_clap(
        audio=audio,
        sample_rate=sample_rate,
        scan_start=args.scan_start,
        scan_duration=args.scan_duration,
        frame_size=args.audio_frame_size,
        hop_size=args.audio_hop_size,
    )
    video_clap = estimate_video_clap(
        video_path=session_dir / args.video_file,
        frame_times=frame_times,
        scan_start=args.scan_start,
        scan_duration=args.scan_duration,
    )
    audio_to_video_offset_ms = None

    if audio_clap is not None and video_clap is not None:
        audio_to_video_offset_ms = round(
            (video_clap["time_seconds"] - audio_clap["time_seconds"]) * 1000.0,
            3,
        )

    result = {
        "format_version": "0.1",
        "session_dir": str(session_dir),
        "audio_ref": args.audio_file,
        "video_ref": args.video_file,
        "frame_times_ref": args.frame_times if frame_times else None,
        "scan_start": args.scan_start,
        "scan_duration": args.scan_duration,
        "audio_clap": audio_clap,
        "video_clap": video_clap,
        "audio_to_video_offset_ms": audio_to_video_offset_ms,
    }
    output_path = session_dir / args.output

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    if args.update_meta:
        update_meta(session_dir, result)

    print(f"AV sync written: {output_path}")

    if audio_clap is None:
        print("audio clap: not found")
    else:
        print(
            "audio clap: "
            f"{audio_clap['time_seconds']:.4f}s "
            f"confidence={audio_clap['confidence']:.2f}"
        )

    if video_clap is None:
        print("video clap: not found")
    else:
        print(
            "video clap: "
            f"{video_clap['time_seconds']:.4f}s "
            f"confidence={video_clap['confidence']:.2f}"
        )

    if audio_to_video_offset_ms is not None:
        print(f"audio_to_video_offset_ms: {audio_to_video_offset_ms:.3f}")


if __name__ == "__main__":
    run_estimate_av_sync()
