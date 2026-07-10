import argparse
import time
from pathlib import Path

from sync_session import SyncGPIOController, SyncSessionConfig, SyncSessionRecorder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYNC_OUTPUT_DIR = PROJECT_ROOT / "data" / "sync_sessions"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test Raspberry Pi tact switch and LED sync recording without camera.",
    )
    parser.add_argument("--gpio-button-pin", type=int, default=17)
    parser.add_argument("--gpio-led-pin", type=int, default=27)
    parser.add_argument("--sync-output-dir", type=Path, default=DEFAULT_SYNC_OUTPUT_DIR)
    parser.add_argument(
        "--demo-events",
        action="store_true",
        help="Write sample fretboard highlight events while recording is ON.",
    )
    return parser.parse_args()


def run_sync_gpio_test() -> None:
    args = parse_args()
    recorder = SyncSessionRecorder(
        SyncSessionConfig(
            output_dir=args.sync_output_dir,
            sample_rate=44100,
            record_audio=False,
        )
    )
    gpio_controller = None
    demo_positions = [
        {"inside_fretboard": True, "string_number": 3, "fret_number": 5},
        {"inside_fretboard": True, "string_number": 3, "fret_number": 7},
        {"inside_fretboard": True, "string_number": 2, "fret_number": 8},
        {"inside_fretboard": True, "string_number": 4, "fret_number": 10},
    ]
    demo_index = 0
    last_demo_event_at = 0.0

    def set_recording_enabled(enabled: bool) -> None:
        if enabled:
            recorder.start()
        else:
            recorder.stop()

        if gpio_controller is not None:
            gpio_controller.set_led(recorder.enabled)

    def toggle_recording() -> None:
        set_recording_enabled(not recorder.enabled)

    gpio_controller = SyncGPIOController(
        button_pin=args.gpio_button_pin,
        led_pin=args.gpio_led_pin,
        on_toggle=toggle_recording,
        enabled=True,
    )

    print("GPIO sync test started")
    print(f"button pin: {args.gpio_button_pin}")
    print(f"LED pin: {args.gpio_led_pin}")
    print("Press the tact switch to toggle recording ON/OFF.")
    print("Recording ON should turn the LED on. Press Ctrl+C to quit.")

    try:
        while True:
            if args.demo_events and recorder.enabled:
                now = time.monotonic()

                if now - last_demo_event_at >= 1.0:
                    position = demo_positions[demo_index % len(demo_positions)]
                    demo_index += 1
                    last_demo_event_at = now
                    recorder.record_highlight(
                        highlight_position=position,
                        midi_pitch=None,
                        note_name=None,
                        pitch_source="gpio_test",
                        current_block_start=position["fret_number"],
                    )
                    print(
                        "demo event: "
                        f'{position["string_number"]} string '
                        f'{position["fret_number"]} fret'
                    )

            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        set_recording_enabled(False)
        gpio_controller.close()


if __name__ == "__main__":
    run_sync_gpio_test()
