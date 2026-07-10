import argparse
import json
import math
import time
import wave
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYNC_OUTPUT_DIR = PROJECT_ROOT / "data" / "sync_sessions"


DEMO_NOTES = (
    {"t_onset": 0.50, "duration": 0.34, "frequency_hz": 196.00, "name": "G3"},
    {"t_onset": 1.00, "duration": 0.34, "frequency_hz": 220.00, "name": "A3"},
    {"t_onset": 1.50, "duration": 0.34, "frequency_hz": 246.94, "name": "B3"},
    {"t_onset": 2.00, "duration": 0.36, "frequency_hz": 293.66, "name": "D4"},
    {"t_onset": 2.55, "duration": 0.40, "frequency_hz": 329.63, "name": "E4"},
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a synthetic sync session for offline-analysis smoke tests.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SYNC_OUTPUT_DIR)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--duration", type=float, default=3.4)
    return parser.parse_args()


def make_unique_session_dir(output_dir: Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_dir = output_dir / f"sync_demo_{timestamp}"

    if not base_dir.exists():
        return base_dir

    for index in range(1, 1000):
        candidate = output_dir / f"sync_demo_{timestamp}_{index:03d}"

        if not candidate.exists():
            return candidate

    return output_dir / f"sync_demo_{timestamp}_{time.time_ns()}"


def synthesize_audio(sample_rate: int, duration: float) -> np.ndarray:
    sample_count = int(round(sample_rate * duration))
    audio = np.zeros(sample_count, dtype=np.float32)

    for note in DEMO_NOTES:
        start_index = int(round(note["t_onset"] * sample_rate))
        end_index = min(
            sample_count,
            start_index + int(round(note["duration"] * sample_rate)),
        )
        note_samples = end_index - start_index

        if note_samples <= 0:
            continue

        t = np.arange(note_samples, dtype=np.float32) / sample_rate
        attack_samples = max(1, int(0.018 * sample_rate))
        release_samples = max(1, int(0.055 * sample_rate))
        envelope = np.ones(note_samples, dtype=np.float32)
        envelope[:attack_samples] = np.linspace(0.0, 1.0, attack_samples)
        envelope[-release_samples:] *= np.linspace(1.0, 0.0, release_samples)
        decay = np.exp(-2.4 * t)
        fundamental = np.sin(2.0 * math.pi * note["frequency_hz"] * t)
        harmonic = 0.28 * np.sin(2.0 * math.pi * note["frequency_hz"] * 2.0 * t)
        audio[start_index:end_index] += 0.48 * envelope * decay * (fundamental + harmonic)

    return np.clip(audio, -1.0, 1.0)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(sample_rate)
        file.writeframes(pcm.tobytes())


def run_make_demo_sync_session() -> None:
    args = parse_args()
    session_dir = make_unique_session_dir(args.output_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    audio = synthesize_audio(sample_rate=args.sample_rate, duration=args.duration)
    write_wav(session_dir / "audio.wav", audio, args.sample_rate)

    events = [
        {"time_seconds": 0.0, "event_type": "sync_started"},
        {"time_seconds": round(args.duration, 4), "event_type": "sync_stopped"},
    ]
    (session_dir / "events.json").write_text(
        json.dumps(events, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    meta = {
        "format_version": "0.1",
        "files": {
            "audio_ref": "audio.wav",
            "events_ref": "events.json",
        },
        "audio": {
            "recorded": True,
            "sample_rate": args.sample_rate,
            "channels": 1,
            "sample_width_bytes": 2,
        },
        "metadata": {
            "demo_notes": DEMO_NOTES,
            "recording_mode": "synthetic_demo",
        },
    }
    (session_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"demo sync session written: {session_dir}")


if __name__ == "__main__":
    run_make_demo_sync_session()
