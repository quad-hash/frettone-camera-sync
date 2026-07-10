import argparse

from fretboard_note_mapper import (
    match_camera_estimate_with_midi_pitch,
    find_positions_for_midi_pitch,
    generate_positions_for_fret_window,
    midi_pitch_to_note_name,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="List string/fret candidates for a MIDI pitch in the current fret window.",
    )
    parser.add_argument("--midi-pitch", type=int, default=62)
    parser.add_argument("--start-fret", type=int, default=5)
    parser.add_argument("--visible-frets", type=int, default=4)
    parser.add_argument("--estimated-string", type=int, default=None)
    parser.add_argument("--estimated-fret", type=int, default=None)
    parser.add_argument(
        "--show-window",
        action="store_true",
        help="Print every note in the current fret window before candidates.",
    )

    return parser.parse_args()


def format_position(position: dict) -> str:
    return (
        f'{position["string_number"]} string '
        f'{position["fret_number"]} fret '
        f'-> {position["note_name"]} '
        f'(MIDI {position["midi_pitch"]})'
    )


def main() -> None:
    args = parse_args()

    if args.show_window:
        print(
            f"Fret window: {args.start_fret}"
            f"-{args.start_fret + args.visible_frets - 1}"
        )

        for position in generate_positions_for_fret_window(
            start_fret=args.start_fret,
            visible_frets=args.visible_frets,
        ):
            print(format_position(position))

        print()

    candidates = find_positions_for_midi_pitch(
        midi_pitch=args.midi_pitch,
        start_fret=args.start_fret,
        visible_frets=args.visible_frets,
    )

    print(
        f"Candidates for {midi_pitch_to_note_name(args.midi_pitch)} "
        f"(MIDI {args.midi_pitch}) in fret window "
        f"{args.start_fret}-{args.start_fret + args.visible_frets - 1}:"
    )

    if not candidates:
        print("No candidates.")
    else:
        for candidate in candidates:
            print(format_position(candidate))

    if args.estimated_string is None and args.estimated_fret is None:
        return

    if args.estimated_string is None or args.estimated_fret is None:
        raise ValueError(
            "--estimated-string and --estimated-fret must be specified together"
        )

    match_result = match_camera_estimate_with_midi_pitch(
        estimated_position={
            "string_number": args.estimated_string,
            "fret_number": args.estimated_fret,
        },
        midi_pitch=args.midi_pitch,
        start_fret=args.start_fret,
        visible_frets=args.visible_frets,
    )
    selected_position = match_result["selected_position"]

    print()
    print(
        "Camera estimate: "
        f"{args.estimated_string} string {args.estimated_fret} fret"
    )

    if selected_position is None:
        print("Matched position: none")
        return

    print(
        "Matched position: "
        f'{selected_position["string_number"]} string '
        f'{selected_position["fret_number"]} fret '
        f'[{match_result["status"]}]'
    )


if __name__ == "__main__":
    main()
