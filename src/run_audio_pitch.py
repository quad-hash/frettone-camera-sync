import argparse
import time

import sounddevice as sd

from audio_pitch_detector import AudioPitchTracker


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read microphone input and estimate a monophonic guitar pitch.",
    )
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--rms-threshold", type=float, default=0.015)
    parser.add_argument("--duration", type=float, default=0.0)
    return parser.parse_args()


def run_audio_pitch_monitor() -> None:
    args = parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    device = int(args.device) if args.device and args.device.isdigit() else args.device
    tracker = AudioPitchTracker(
        device=device,
        sample_rate=args.sample_rate,
        block_size=args.block_size,
        rms_threshold=args.rms_threshold,
    )
    tracker.start()

    print("audio pitch monitor started")
    print("Play one note at a time. Press Ctrl+C to stop.")

    last_midi_pitch = None
    start_time = time.monotonic()

    try:
        while True:
            if args.duration > 0 and time.monotonic() - start_time >= args.duration:
                break

            detection = tracker.poll()

            if detection is not None and detection.midi_pitch != last_midi_pitch:
                last_midi_pitch = detection.midi_pitch
                print(
                    f"{detection.note_name} MIDI {detection.midi_pitch} "
                    f"{detection.frequency_hz:.1f} Hz "
                    f"confidence={detection.confidence:.2f} "
                    f"rms={detection.rms:.3f}"
                )

            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        tracker.stop()


if __name__ == "__main__":
    run_audio_pitch_monitor()
