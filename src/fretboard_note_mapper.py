from dataclasses import dataclass


STANDARD_TUNING_OPEN_MIDI: dict[int, int] = {
    6: 40,  # E2
    5: 45,  # A2
    4: 50,  # D3
    3: 55,  # G3
    2: 59,  # B3
    1: 64,  # E4
}

NOTE_NAMES_SHARP = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)


@dataclass(frozen=True)
class FretboardNotePosition:
    string_number: int
    fret_number: int
    midi_pitch: int
    note_name: str

    def to_dict(self) -> dict:
        return {
            "string_number": self.string_number,
            "fret_number": self.fret_number,
            "midi_pitch": self.midi_pitch,
            "note_name": self.note_name,
        }


def midi_pitch_to_note_name(midi_pitch: int) -> str:
    note_name = NOTE_NAMES_SHARP[midi_pitch % 12]
    octave = (midi_pitch // 12) - 1

    return f"{note_name}{octave}"


def string_fret_to_midi_pitch(
    string_number: int,
    fret_number: int,
    tuning_open_midi: dict[int, int] | None = None,
) -> int:
    tuning = tuning_open_midi or STANDARD_TUNING_OPEN_MIDI

    if string_number not in tuning:
        raise ValueError(f"Unknown string number: {string_number}")

    if fret_number < 0:
        raise ValueError("fret_number must be 0 or greater")

    return tuning[string_number] + fret_number


def make_note_position(
    string_number: int,
    fret_number: int,
    tuning_open_midi: dict[int, int] | None = None,
) -> FretboardNotePosition:
    midi_pitch = string_fret_to_midi_pitch(
        string_number=string_number,
        fret_number=fret_number,
        tuning_open_midi=tuning_open_midi,
    )

    return FretboardNotePosition(
        string_number=string_number,
        fret_number=fret_number,
        midi_pitch=midi_pitch,
        note_name=midi_pitch_to_note_name(midi_pitch),
    )


def generate_positions_for_fret_window(
    start_fret: int,
    visible_frets: int,
    tuning_open_midi: dict[int, int] | None = None,
) -> list[dict]:
    if start_fret < 0:
        raise ValueError("start_fret must be 0 or greater")

    if visible_frets <= 0:
        raise ValueError("visible_frets must be greater than 0")

    positions = []

    for string_number in sorted(STANDARD_TUNING_OPEN_MIDI.keys(), reverse=True):
        for fret_number in range(start_fret, start_fret + visible_frets):
            position = make_note_position(
                string_number=string_number,
                fret_number=fret_number,
                tuning_open_midi=tuning_open_midi,
            )
            positions.append(position.to_dict())

    return positions


def find_positions_for_midi_pitch(
    midi_pitch: int,
    start_fret: int,
    visible_frets: int,
    tuning_open_midi: dict[int, int] | None = None,
) -> list[dict]:
    positions = generate_positions_for_fret_window(
        start_fret=start_fret,
        visible_frets=visible_frets,
        tuning_open_midi=tuning_open_midi,
    )

    return [
        position
        for position in positions
        if position["midi_pitch"] == midi_pitch
    ]


def calculate_position_distance(
    estimated_position: dict,
    candidate_position: dict,
) -> float:
    estimated_string = estimated_position.get("string_number")
    estimated_fret = estimated_position.get("fret_number")

    if estimated_string is None or estimated_fret is None:
        raise ValueError("estimated_position must include string_number and fret_number")

    string_distance = abs(estimated_string - candidate_position["string_number"])
    fret_distance = abs(estimated_fret - candidate_position["fret_number"])

    return float(string_distance + fret_distance)


def choose_nearest_position_candidate(
    estimated_position: dict,
    candidate_positions: list[dict],
) -> dict | None:
    if not candidate_positions:
        return None

    scored_candidates = []

    for candidate_position in candidate_positions:
        distance = calculate_position_distance(
            estimated_position=estimated_position,
            candidate_position=candidate_position,
        )
        scored_candidate = dict(candidate_position)
        scored_candidate["distance_from_camera_estimate"] = distance
        scored_candidates.append(scored_candidate)

    return min(
        scored_candidates,
        key=lambda position: (
            position["distance_from_camera_estimate"],
            position["string_number"],
            position["fret_number"],
        ),
    )


def match_camera_estimate_with_midi_pitch(
    estimated_position: dict,
    midi_pitch: int,
    start_fret: int,
    visible_frets: int,
    tuning_open_midi: dict[int, int] | None = None,
) -> dict:
    candidate_positions = find_positions_for_midi_pitch(
        midi_pitch=midi_pitch,
        start_fret=start_fret,
        visible_frets=visible_frets,
        tuning_open_midi=tuning_open_midi,
    )
    selected_position = choose_nearest_position_candidate(
        estimated_position=estimated_position,
        candidate_positions=candidate_positions,
    )

    if selected_position is None:
        status = "no_pitch_candidate"
    elif len(candidate_positions) == 1:
        status = "single_pitch_candidate"
    else:
        status = "nearest_camera_candidate"

    return {
        "status": status,
        "midi_pitch": midi_pitch,
        "note_name": midi_pitch_to_note_name(midi_pitch),
        "camera_estimate": estimated_position,
        "candidate_positions": candidate_positions,
        "selected_position": selected_position,
    }
