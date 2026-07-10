from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Queue

import numpy as np

from fretboard_note_mapper import midi_pitch_to_note_name


@dataclass(frozen=True)
class PitchDetection:
    frequency_hz: float
    midi_pitch: int
    note_name: str
    confidence: float
    rms: float


def frequency_to_midi_pitch(frequency_hz: float) -> int:
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be greater than 0")

    return int(round(69 + 12 * np.log2(frequency_hz / 440.0)))


def midi_pitch_to_frequency(midi_pitch: int) -> float:
    return float(440.0 * (2 ** ((midi_pitch - 69) / 12)))


def parabolic_peak_offset(values: np.ndarray, peak_index: int) -> float:
    if peak_index <= 0 or peak_index >= len(values) - 1:
        return 0.0

    left = values[peak_index - 1]
    center = values[peak_index]
    right = values[peak_index + 1]
    denominator = left - 2 * center + right

    if abs(denominator) < 1e-12:
        return 0.0

    offset = float(0.5 * (left - right) / denominator)

    if not np.isfinite(offset):
        return 0.0

    return float(np.clip(offset, -0.5, 0.5))


def detect_pitch_from_samples(
    samples: np.ndarray,
    sample_rate: int,
    min_frequency_hz: float = 70.0,
    max_frequency_hz: float = 1000.0,
    rms_threshold: float = 0.015,
    confidence_threshold: float = 0.22,
) -> PitchDetection | None:
    mono_samples = np.asarray(samples, dtype=np.float32).reshape(-1)

    if len(mono_samples) < 2:
        return None

    mono_samples = mono_samples - float(np.mean(mono_samples))
    rms = float(np.sqrt(np.mean(mono_samples * mono_samples)))

    if rms < rms_threshold:
        return None

    windowed_samples = mono_samples * np.hanning(len(mono_samples))
    autocorrelation = np.correlate(windowed_samples, windowed_samples, mode="full")
    autocorrelation = autocorrelation[len(autocorrelation) // 2 :]

    if autocorrelation[0] <= 0:
        return None

    min_lag = max(1, int(sample_rate / max_frequency_hz))
    max_lag = min(len(autocorrelation) - 1, int(sample_rate / min_frequency_hz))

    if min_lag >= max_lag:
        return None

    search_window = autocorrelation[min_lag : max_lag + 1]
    peak_index = int(np.argmax(search_window))
    peak_lag = min_lag + peak_index
    confidence = float(search_window[peak_index] / autocorrelation[0])

    if confidence < confidence_threshold:
        return None

    refined_lag = peak_lag + parabolic_peak_offset(autocorrelation, peak_lag)

    if not np.isfinite(refined_lag) or refined_lag <= 0:
        return None

    frequency_hz = float(sample_rate / refined_lag)

    if (
        not np.isfinite(frequency_hz)
        or frequency_hz < min_frequency_hz
        or frequency_hz > max_frequency_hz
    ):
        return None

    midi_pitch = frequency_to_midi_pitch(frequency_hz)

    return PitchDetection(
        frequency_hz=frequency_hz,
        midi_pitch=midi_pitch,
        note_name=midi_pitch_to_note_name(midi_pitch),
        confidence=confidence,
        rms=rms,
    )


class AudioPitchTracker:
    def __init__(
        self,
        device: int | str | None = None,
        sample_rate: int = 44100,
        block_size: int = 4096,
        min_frequency_hz: float = 70.0,
        max_frequency_hz: float = 1000.0,
        rms_threshold: float = 0.015,
        audio_block_callback=None,
    ) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.min_frequency_hz = min_frequency_hz
        self.max_frequency_hz = max_frequency_hz
        self.rms_threshold = rms_threshold
        self.audio_block_callback = audio_block_callback
        self.audio_blocks: Queue[np.ndarray] = Queue(maxsize=4)
        self.latest_detection: PitchDetection | None = None
        self.stream = None

    def start(self) -> None:
        import sounddevice as sd

        def handle_audio_input(indata, frames, time, status) -> None:
            if status:
                print(f"audio input status: {status}")

            audio_block = np.asarray(indata[:, 0], dtype=np.float32).copy()

            if self.audio_block_callback is not None:
                self.audio_block_callback(audio_block)

            if self.audio_blocks.full():
                try:
                    self.audio_blocks.get_nowait()
                except Empty:
                    pass

            self.audio_blocks.put_nowait(audio_block)

        self.stream = sd.InputStream(
            device=self.device,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            dtype="float32",
            callback=handle_audio_input,
        )
        self.stream.start()

    def poll(self) -> PitchDetection | None:
        latest_block = None

        while True:
            try:
                latest_block = self.audio_blocks.get_nowait()
            except Empty:
                break

        if latest_block is None:
            return None

        detection = detect_pitch_from_samples(
            latest_block,
            sample_rate=self.sample_rate,
            min_frequency_hz=self.min_frequency_hz,
            max_frequency_hz=self.max_frequency_hz,
            rms_threshold=self.rms_threshold,
        )

        if detection is not None:
            self.latest_detection = detection

        return detection

    def stop(self) -> None:
        if self.stream is None:
            return

        self.stream.stop()
        self.stream.close()
        self.stream = None
