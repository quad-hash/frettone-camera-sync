import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from run_offline_analyze import (
    apply_visual_evidence_to_note,
    choose_positions_with_viterbi,
    mediapipe_hand_evidence_from_track,
    nearest_hand_track_frame,
)
from run_sync_playback import get_note_highlight_at_time, load_notes
from estimate_av_sync import estimate_audio_clap, pick_peak_time

import numpy as np


def make_note(onset, offset, candidates, pitch_confidence=0.8):
    return {
        "t_onset": onset,
        "t_offset": offset,
        "pitch_confidence": pitch_confidence,
        "candidate_count": len(candidates),
        "candidates": [dict(candidate) for candidate in candidates],
    }


class OfflineViterbiTests(unittest.TestCase):
    def test_context_prefers_playable_path_across_ambiguous_candidates(self):
        notes = [
            make_note(
                0.00,
                0.20,
                [
                    {"string_number": 6, "fret_number": 8},
                    {"string_number": 3, "fret_number": 3},
                ],
            ),
            make_note(
                0.24,
                0.44,
                [
                    {"string_number": 6, "fret_number": 10},
                    {"string_number": 2, "fret_number": 15},
                ],
            ),
        ]

        result = choose_positions_with_viterbi(notes, visual_weight=0.0)

        self.assertEqual((result[0]["string"], result[0]["fret"]), (6, 8))
        self.assertEqual((result[1]["string"], result[1]["fret"]), (6, 10))
        self.assertIn(result[0]["status"], {"viterbi_selected", "viterbi_ambiguous_candidate"})

    def test_visual_score_can_resolve_same_pitch_candidates(self):
        notes = [
            make_note(
                0.00,
                0.18,
                [
                    {"string_number": 4, "fret_number": 12, "visual_score": 0.05},
                    {"string_number": 3, "fret_number": 7, "visual_score": 0.92},
                    {"string_number": 2, "fret_number": 3, "visual_score": 0.02},
                ],
            ),
            make_note(
                0.22,
                0.40,
                [
                    {"string_number": 4, "fret_number": 14, "visual_score": 0.05},
                    {"string_number": 3, "fret_number": 9, "visual_score": 0.9},
                    {"string_number": 2, "fret_number": 5, "visual_score": 0.02},
                    {"string_number": 1, "fret_number": 0, "visual_score": 0.0},
                ],
            ),
        ]

        result = choose_positions_with_viterbi(notes, visual_weight=1.2)

        self.assertEqual((result[0]["string"], result[0]["fret"]), (3, 7))
        self.assertEqual((result[1]["string"], result[1]["fret"]), (3, 9))
        self.assertEqual(result[0]["status"], "visual_viterbi_selected")
        self.assertGreater(result[0]["confidence"], 0.6)

    def test_note_without_candidates_is_marked_unresolved(self):
        notes = [make_note(0.0, 0.2, [])]

        result = choose_positions_with_viterbi(notes)

        self.assertIsNone(result[0]["string"])
        self.assertIsNone(result[0]["fret"])
        self.assertEqual(result[0]["confidence"], 0.0)
        self.assertEqual(result[0]["status"], "no_pitch_candidate")


class MediaPipeHandEvidenceTests(unittest.TestCase):
    def test_nearest_hand_track_frame_respects_window(self):
        frames = [
            {"time_seconds": 0.1, "hands": []},
            {"time_seconds": 0.3, "hands": [{"score": 0.9}]},
        ]

        frame, delta = nearest_hand_track_frame(frames, 0.26, 0.08)

        self.assertEqual(frame["time_seconds"], 0.3)
        self.assertAlmostEqual(delta, 0.04)
        missing_frame, _ = nearest_hand_track_frame(frames, 0.7, 0.08)
        self.assertIsNone(missing_frame)

    def test_mediapipe_hand_score_can_boost_matching_fret_candidate(self):
        landmarks = [
            {"px": 100.0, "py": 90.0, "x": 0.0, "y": 0.0, "z": 0.0}
            for _ in range(21)
        ]

        for index in (4, 8, 12, 16, 20):
            landmarks[index]["px"] = 480.0

        hand_frame = {
            "frame_index": 12,
            "time_seconds": 1.0,
            "hands": [{"score": 0.95, "landmarks": landmarks}],
        }
        evidence = mediapipe_hand_evidence_from_track(
            hand_frame,
            np.eye(3, dtype=np.float32),
            time_delta_seconds=0.01,
            max_delta_seconds=0.14,
        )
        note = make_note(
            1.0,
            1.2,
            [
                {"string_number": 6, "fret_number": 2},
                {"string_number": 3, "fret_number": 7},
            ],
        )

        apply_visual_evidence_to_note(
            note,
            {
                "string_scores": {6: 0.0, 3: 0.0},
                "hand_center_fret": None,
                "hand_start_fret": None,
                "hand_end_fret": None,
                "hand_confidence": 0.0,
                "changed_pixels": 0,
                "mediapipe_hand": evidence,
            },
        )

        self.assertEqual(evidence["center_fret"], 7)
        self.assertGreater(
            note["candidates"][1]["visual_score"],
            note["candidates"][0]["visual_score"],
        )


class OfflinePlaybackNotesTests(unittest.TestCase):
    def test_load_notes_and_get_highlight_from_notes_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = Path(temp_dir)
            notes_path = session_dir / "notes.json"
            notes_path.write_text(
                json.dumps(
                    {
                        "format_version": "0.1",
                        "notes": [
                            {
                                "t_onset": 1.0,
                                "t_offset": 1.5,
                                "string": 3,
                                "fret": 7,
                                "midi_pitch": 62,
                                "note_name": "D4",
                                "confidence": 0.86,
                                "status": "visual_viterbi_selected",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            notes = load_notes(session_dir)
            highlight = get_note_highlight_at_time(notes, 1.2)

        self.assertEqual(len(notes), 1)
        self.assertEqual(highlight["string_number"], 3)
        self.assertEqual(highlight["fret_number"], 7)
        self.assertEqual(highlight["note_name"], "D4")
        self.assertEqual(highlight["confidence"], 0.86)
        self.assertIsNone(get_note_highlight_at_time(notes, 1.8))


class AvSyncTests(unittest.TestCase):
    def test_pick_peak_time_returns_strongest_peak(self):
        values = np.array([0.1, 0.2, 1.3, 0.3], dtype=np.float32)
        times = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32)

        result = pick_peak_time(values, times)

        self.assertAlmostEqual(result["time_seconds"], 0.2, places=4)
        self.assertEqual(result["peak_index"], 2)
        self.assertGreater(result["confidence"], 0.0)

    def test_estimate_audio_clap_finds_synthetic_spike(self):
        sample_rate = 1000
        audio = np.zeros(sample_rate * 2, dtype=np.float32)
        audio[730:735] = 0.95

        result = estimate_audio_clap(
            audio=audio,
            sample_rate=sample_rate,
            scan_start=0.0,
            scan_duration=2.0,
            frame_size=64,
            hop_size=16,
        )

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["time_seconds"], 0.688, delta=0.04)
        self.assertEqual(result["method"], "audio_peak")


if __name__ == "__main__":
    unittest.main()
