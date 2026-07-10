from __future__ import annotations

import json
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


@dataclass
class SyncSessionConfig:
    output_dir: Path
    sample_rate: int
    record_audio: bool = True
    record_video: bool = False
    video_fps: float = 30.0
    video_codec: str = "mp4v"
    video_filename: str = "video.mp4"


class SyncSessionRecorder:
    def __init__(self, config: SyncSessionConfig) -> None:
        self.config = config
        self.enabled = False
        self.session_dir: Path | None = None
        self.started_at_monotonic = 0.0
        self.events: list[dict] = []
        self.audio_queue: Queue[np.ndarray] = Queue()
        self.wave_file = None
        self.video_writer = None
        self.video_frame_size: tuple[int, int] | None = None
        self.video_frame_count = 0
        self.frame_times: list[dict] = []
        self.metadata: dict = {}
        self.started_at_wall_time = 0.0
        self.video_error_reported = False
        self.last_highlight_key = None

    def make_unique_session_dir(self) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_dir = self.config.output_dir / f"sync_{timestamp}"

        if not base_dir.exists():
            return base_dir

        for index in range(1, 1000):
            candidate = self.config.output_dir / f"sync_{timestamp}_{index:03d}"

            if not candidate.exists():
                return candidate

        return self.config.output_dir / f"sync_{timestamp}_{time.time_ns()}"

    def start(self, metadata: dict | None = None) -> None:
        if self.enabled:
            return

        self.session_dir = self.make_unique_session_dir()
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.started_at_monotonic = time.monotonic()
        self.started_at_wall_time = time.time()
        self.events = []
        self.frame_times = []
        self.video_frame_count = 0
        self.video_frame_size = None
        self.video_error_reported = False
        self.metadata = dict(metadata or {})
        self.last_highlight_key = None

        if self.config.record_audio:
            self.wave_file = wave.open(str(self.session_dir / "audio.wav"), "wb")
            self.wave_file.setnchannels(1)
            self.wave_file.setsampwidth(2)
            self.wave_file.setframerate(self.config.sample_rate)

        self.enabled = True
        self.record_event("sync_started", {})
        self.write_meta()
        print(f"sync ON: {self.session_dir}")

    def stop(self) -> None:
        if not self.enabled:
            return

        self.drain_audio()
        self.record_event("sync_stopped", {})

        if self.session_dir is not None:
            with (self.session_dir / "events.json").open("w", encoding="utf-8") as file:
                json.dump(self.events, file, indent=2, ensure_ascii=False)

            with (self.session_dir / "frame_times.json").open("w", encoding="utf-8") as file:
                json.dump(self.frame_times, file, indent=2, ensure_ascii=False)

        if self.wave_file is not None:
            self.wave_file.close()
            self.wave_file = None

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        self.write_meta()
        print(f"sync OFF: {self.session_dir}")
        self.enabled = False

    def toggle(self) -> None:
        if self.enabled:
            self.stop()
        else:
            self.start()

    def elapsed_seconds(self) -> float:
        if not self.enabled:
            return 0.0

        return time.monotonic() - self.started_at_monotonic

    def record_event(self, event_type: str, payload: dict) -> None:
        if not self.enabled:
            return

        event = {
            "time_seconds": round(self.elapsed_seconds(), 4),
            "event_type": event_type,
            **payload,
        }
        self.events.append(event)

    def make_meta(self) -> dict:
        files = {
            "events_ref": "events.json",
            "frame_times_ref": "frame_times.json",
        }

        if self.config.record_audio:
            files["audio_ref"] = "audio.wav"

        if self.config.record_video:
            files["video_ref"] = self.config.video_filename

        return {
            "format_version": "0.1",
            "started_at_unix": self.started_at_wall_time,
            "started_at_local": (
                time.strftime(
                    "%Y-%m-%dT%H:%M:%S%z",
                    time.localtime(self.started_at_wall_time),
                )
                if self.started_at_wall_time
                else None
            ),
            "files": files,
            "audio": {
                "recorded": self.config.record_audio,
                "sample_rate": self.config.sample_rate,
                "channels": 1,
                "sample_width_bytes": 2,
            },
            "video": {
                "recorded": self.config.record_video,
                "fps": self.config.video_fps,
                "codec": self.config.video_codec,
                "frame_count": self.video_frame_count,
                "frame_size": self.video_frame_size,
            },
            "metadata": self.metadata,
        }

    def write_meta(self) -> None:
        if self.session_dir is None:
            return

        with (self.session_dir / "meta.json").open("w", encoding="utf-8") as file:
            json.dump(self.make_meta(), file, indent=2, ensure_ascii=False)

    def ensure_video_writer(self, frame: np.ndarray) -> bool:
        if self.video_writer is not None:
            return True

        if cv2 is None:
            if not self.video_error_reported:
                print("video recording disabled: cv2 is not available")
                self.video_error_reported = True
            return False

        if self.session_dir is None:
            return False

        frame_height, frame_width = frame.shape[:2]
        self.video_frame_size = (int(frame_width), int(frame_height))
        fourcc = cv2.VideoWriter_fourcc(*self.config.video_codec[:4])
        self.video_writer = cv2.VideoWriter(
            str(self.session_dir / self.config.video_filename),
            fourcc,
            float(self.config.video_fps),
            self.video_frame_size,
        )

        if not self.video_writer.isOpened():
            self.video_writer.release()
            self.video_writer = None
            if not self.video_error_reported:
                print("video recording disabled: could not open VideoWriter")
                self.video_error_reported = True
            return False

        self.write_meta()
        return True

    def record_video_frame(self, frame: np.ndarray) -> None:
        if not self.enabled or not self.config.record_video:
            return

        if not self.ensure_video_writer(frame):
            return

        if len(frame.shape) == 2:
            frame_to_write = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_to_write = frame

        self.video_writer.write(frame_to_write)
        self.frame_times.append(
            {
                "frame_index": self.video_frame_count,
                "time_seconds": round(self.elapsed_seconds(), 6),
            }
        )
        self.video_frame_count += 1

    def add_audio_block(self, audio_block: np.ndarray) -> None:
        if not self.enabled or not self.config.record_audio:
            return

        self.audio_queue.put(np.asarray(audio_block, dtype=np.float32).copy())

    def drain_audio(self) -> None:
        if self.wave_file is None:
            return

        while True:
            try:
                audio_block = self.audio_queue.get_nowait()
            except Empty:
                break

            clipped = np.clip(audio_block, -1.0, 1.0)
            pcm = (clipped * 32767.0).astype(np.int16)
            self.wave_file.writeframes(pcm.tobytes())

    def record_highlight(
        self,
        highlight_position: dict | None,
        midi_pitch: int | None,
        note_name: str | None,
        pitch_source: str,
        current_block_start: int,
        match_result: dict | None = None,
    ) -> None:
        if not self.enabled or highlight_position is None:
            return

        string_number = highlight_position.get("string_number")
        fret_number = highlight_position.get("fret_number")

        if string_number is None or fret_number is None:
            return

        match_status = match_result.get("status") if match_result is not None else None
        highlight_key = (
            string_number,
            fret_number,
            midi_pitch,
            pitch_source,
            current_block_start,
            match_status,
        )

        if highlight_key == self.last_highlight_key:
            return

        self.last_highlight_key = highlight_key
        self.record_event(
            "fretboard_highlight",
            {
                "string_number": string_number,
                "fret_number": fret_number,
                "midi_pitch": midi_pitch,
                "note_name": note_name,
                "pitch_source": pitch_source,
                "current_block_start": current_block_start,
                "match_status": match_status,
            },
        )


class SyncGPIOController:
    def __init__(
        self,
        button_pin: int | None,
        led_pin: int | None,
        on_toggle,
        enabled: bool = True,
    ) -> None:
        self.button = None
        self.led = None

        if not enabled or (button_pin is None and led_pin is None):
            return

        try:
            from gpiozero import Button, LED
        except ImportError:
            print("gpiozero is not installed. GPIO sync controls are disabled.")
            return

        if led_pin is not None:
            self.led = LED(led_pin)
            self.led.off()

        if button_pin is not None:
            self.button = Button(button_pin, pull_up=True, bounce_time=0.08)
            self.button.when_pressed = on_toggle
            print(f"GPIO sync button enabled: pin={button_pin}")

        if led_pin is not None:
            print(f"GPIO sync LED enabled: pin={led_pin}")

    def set_led(self, enabled: bool) -> None:
        if self.led is None:
            return

        if enabled:
            self.led.on()
        else:
            self.led.off()

    def close(self) -> None:
        if self.button is not None:
            self.button.close()
        if self.led is not None:
            self.led.off()
            self.led.close()
