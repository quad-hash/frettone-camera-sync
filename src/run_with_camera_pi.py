import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from audio_pitch_detector import AudioPitchTracker
import fretboard_display
from fretboard_position_mapper import (
    FretboardMappingConfig,
    build_perspective_matrix,
    estimate_string_and_fret,
    load_calibration_data,
    map_image_point_to_virtual_point,
)
from run_with_image import (
    draw_calibration_frame,
    draw_estimation_result,
    draw_fret_boundaries,
    format_pitch_match_result,
)
from fretboard_note_mapper import (
    STANDARD_TUNING_OPEN_MIDI,
    find_positions_for_midi_pitch,
    match_camera_estimate_with_midi_pitch,
    midi_pitch_to_note_name,
)
from fretboard_marker_detector import (
    COLOR_RANGES,
    DEFAULT_BLOCK_MARKERS,
    FretboardBlockDetection,
    MarkerDetection,
    choose_block_by_hand_markers,
    detect_all_markers,
    detect_fretboard_block,
    detect_markers_for_color,
    draw_marker_detection,
    draw_marker_polygon,
    draw_marker_roi,
    make_block_rect,
    parse_roi,
)
from sync_session import SyncGPIOController, SyncSessionConfig, SyncSessionRecorder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION_JSON_PATH = PROJECT_ROOT / "data" / "camera_calibration_points.json"
DEFAULT_SYNC_OUTPUT_DIR = PROJECT_ROOT / "data" / "sync_sessions"
WINDOW_NAME = "FretTone Camera Sync - Camera Test"

FLIP_MODES = ("none", "horizontal", "vertical", "both")
DISPLAY_MODES = ("fretboard", "camera")
FRETBOARD_DISPLAY_WIDTH = fretboard_display.DEFAULT_DISPLAY_WIDTH
FRETBOARD_DISPLAY_HEIGHT = fretboard_display.DEFAULT_DISPLAY_HEIGHT
EDGE_PAIR_MARKER_COUNTS = {
    "green": 2,
    "blue": 2,
    "yellow": 4,
    "pink": 4,
}
BACKGROUND_HAND_WARP_SIZE = (960, 180)


def apply_frame_flip(frame, flip_mode: str):
    if flip_mode == "horizontal":
        return cv2.flip(frame, 1)
    if flip_mode == "vertical":
        return cv2.flip(frame, 0)
    if flip_mode == "both":
        return cv2.flip(frame, -1)

    return frame


def next_flip_mode(flip_mode: str) -> str:
    index = FLIP_MODES.index(flip_mode)
    return FLIP_MODES[(index + 1) % len(FLIP_MODES)]


def parse_finger_marker_specs(spec_text: str) -> dict[str, str]:
    specs = {}

    if not spec_text:
        return specs

    for item in spec_text.split(","):
        item = item.strip()

        if not item:
            continue

        if ":" not in item:
            raise argparse.ArgumentTypeError(
                f"finger marker must be name:color, got {item!r}"
            )

        finger_name, color_name = (part.strip().lower() for part in item.split(":", 1))

        if not finger_name:
            raise argparse.ArgumentTypeError("finger marker name cannot be empty")

        if color_name not in COLOR_RANGES:
            choices = ", ".join(sorted(COLOR_RANGES.keys()))
            raise argparse.ArgumentTypeError(
                f"unknown marker color {color_name!r}. choices: {choices}"
            )

        specs[finger_name] = color_name

    return specs


def parse_marker_color_list(spec_text: str) -> tuple[str, ...]:
    color_names = []

    for item in spec_text.split(","):
        color_name = item.strip().lower()

        if not color_name:
            continue

        if color_name not in COLOR_RANGES:
            choices = ", ".join(sorted(COLOR_RANGES.keys()))
            raise argparse.ArgumentTypeError(
                f"unknown marker color {color_name!r}. choices: {choices}"
            )

        color_names.append(color_name)

    if len(color_names) < 2:
        raise argparse.ArgumentTypeError("at least 2 marker colors are required")

    if len(set(color_names)) != len(color_names):
        raise argparse.ArgumentTypeError("marker colors must be unique")

    return tuple(color_names)


def copy_marker_points(marker_points: list[dict]) -> list[dict]:
    return [
        {
            "color": marker_point["color"],
            "index": marker_point["index"],
            "center": tuple(marker_point["center"]),
        }
        for marker_point in marker_points
    ]


def make_marker_canonical_points(
    marker_points: list[dict],
    size: tuple[int, int] = BACKGROUND_HAND_WARP_SIZE,
    padding: float = 10.0,
) -> dict[tuple[str, int], tuple[float, float]]:
    if len(marker_points) < 4:
        return {}

    width, height = size
    points = np.array(
        [marker_point["center"] for marker_point in marker_points],
        dtype=np.float32,
    )
    centered_points = points - points.mean(axis=0)

    if len(centered_points) < 2:
        return {}

    _, _, vh = np.linalg.svd(centered_points, full_matrices=False)
    fret_axis = vh[0]
    green_points = np.array(
        [
            marker_point["center"]
            for marker_point in marker_points
            if marker_point["color"] == "green"
        ],
        dtype=np.float32,
    )

    if len(green_points) > 0 and float((green_points @ fret_axis).mean()) > float((points @ fret_axis).mean()):
        fret_axis = -fret_axis

    string_axis = np.array([-fret_axis[1], fret_axis[0]], dtype=np.float32)
    fret_projections = points @ fret_axis
    string_projections = points @ string_axis
    fret_min = float(fret_projections.min())
    fret_max = float(fret_projections.max())
    string_min = float(string_projections.min())
    string_max = float(string_projections.max())
    fret_span = fret_max - fret_min
    string_span = string_max - string_min

    if fret_span <= 1.0 or string_span <= 1.0:
        return {}

    canonical_points = {}
    usable_width = max(1.0, width - padding * 2)
    usable_height = max(1.0, height - padding * 2)

    for marker_point in marker_points:
        point = np.array(marker_point["center"], dtype=np.float32)
        x_ratio = (float(point @ fret_axis) - fret_min) / fret_span
        y_ratio = (float(point @ string_axis) - string_min) / string_span
        canonical_points[(marker_point["color"], marker_point["index"])] = (
            padding + max(0.0, min(1.0, x_ratio)) * usable_width,
            padding + max(0.0, min(1.0, y_ratio)) * usable_height,
        )

    return canonical_points


def build_marker_to_canonical_homography(
    marker_points: list[dict],
    canonical_points: dict[tuple[str, int], tuple[float, float]],
) -> np.ndarray | None:
    source_points = []
    destination_points = []

    for marker_point in marker_points:
        marker_key = (marker_point["color"], marker_point["index"])
        canonical_point = canonical_points.get(marker_key)

        if canonical_point is None:
            continue

        source_points.append(marker_point["center"])
        destination_points.append(canonical_point)

    if len(source_points) < 4:
        return None

    homography, _ = cv2.findHomography(
        np.array(source_points, dtype=np.float32),
        np.array(destination_points, dtype=np.float32),
        cv2.RANSAC,
        6.0,
    )
    return homography


def estimate_similarity_from_two_points(
    reference_points: tuple[tuple[int, int], tuple[int, int]],
    current_points: tuple[tuple[int, int], tuple[int, int]],
) -> np.ndarray | None:
    ref_a = np.array(reference_points[0], dtype=np.float32)
    ref_b = np.array(reference_points[1], dtype=np.float32)
    cur_a = np.array(current_points[0], dtype=np.float32)
    cur_b = np.array(current_points[1], dtype=np.float32)
    ref_vector = ref_b - ref_a
    cur_vector = cur_b - cur_a
    ref_length = float(np.linalg.norm(ref_vector))
    cur_length = float(np.linalg.norm(cur_vector))

    if ref_length < 1.0 or cur_length < 1.0:
        return None

    scale = cur_length / ref_length
    ref_angle = float(np.arctan2(ref_vector[1], ref_vector[0]))
    cur_angle = float(np.arctan2(cur_vector[1], cur_vector[0]))
    angle = cur_angle - ref_angle
    cos_value = float(np.cos(angle) * scale)
    sin_value = float(np.sin(angle) * scale)
    rotation_scale = np.array(
        [[cos_value, -sin_value], [sin_value, cos_value]],
        dtype=np.float32,
    )
    translation = cur_a - rotation_scale @ ref_a
    return np.array(
        [
            [rotation_scale[0, 0], rotation_scale[0, 1], translation[0]],
            [rotation_scale[1, 0], rotation_scale[1, 1], translation[1]],
        ],
        dtype=np.float32,
    )


def expand_roi_for_frame(
    roi: tuple[int, int, int, int] | None,
    margin: int,
    frame_shape,
) -> tuple[int, int, int, int] | None:
    if roi is None:
        return None

    frame_height, frame_width = frame_shape[:2]
    x, y, width, height = roi
    left = max(0, x - margin)
    top = max(0, y - margin)
    right = min(frame_width, x + width + margin)
    bottom = min(frame_height, y + height + margin)
    return left, top, right - left, bottom - top


def make_roi_from_points(
    points: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    frame_shape,
    margin: int,
) -> tuple[int, int, int, int] | None:
    if not points:
        return None

    frame_height, frame_width = frame_shape[:2]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    left = max(0, int(min(xs)) - margin)
    top = max(0, int(min(ys)) - margin)
    right = min(frame_width, int(max(xs)) + margin)
    bottom = min(frame_height, int(max(ys)) + margin)

    if right <= left or bottom <= top:
        return None

    return left, top, right - left, bottom - top


def order_quad_points(points: tuple[tuple[int, int], ...] | list[tuple[int, int]]):
    point_array = np.array(points, dtype=np.float32)
    sums = point_array.sum(axis=1)
    diffs = np.diff(point_array, axis=1).reshape(-1)

    return np.array(
        [
            point_array[int(np.argmin(sums))],
            point_array[int(np.argmin(diffs))],
            point_array[int(np.argmax(sums))],
            point_array[int(np.argmax(diffs))],
        ],
        dtype=np.float32,
    )


def get_compatible_fret_boundaries(fret_boundaries, config: FretboardMappingConfig):
    if fret_boundaries is None:
        return None

    if len(fret_boundaries) == config.visible_frets + 1:
        return fret_boundaries

    print(
        "Saved fret boundaries do not match current visible_frets. "
        "Using equal fret split instead."
    )
    return None


class CameraMapperState:
    def __init__(
        self,
        config: FretboardMappingConfig,
        calibration_json_path: Path,
    ) -> None:
        self.config = config
        self.calibration_json_path = calibration_json_path
        self.mode = "mapping"
        self.corner_points = []
        self.internal_fret_points = []
        self.calibration_points = None
        self.fret_boundaries = None
        self.perspective_matrix = None
        self.last_click = None
        self.last_result = None
        self.last_match_result = None
        self.flip_mode = "none"
        self.tracking_enabled = True
        self.tracking_lost = False
        self.previous_gray_frame = None
        self.tracking_points = None
        self.midi_pitch = None
        self.pitch_source = "manual"
        self.last_audio_detection = None
        self.display_mode = "fretboard"
        self.display_total_frets = fretboard_display.DEFAULT_TOTAL_FRETS
        self.display_block_size = fretboard_display.DEFAULT_BLOCK_SIZE
        self.sync_enabled = False
        self.marker_detection_enabled = False
        self.hand_marker_color = None
        self.hand_marker_count = 2
        self.hand_marker_min_area = 30.0
        self.raw_hand_markers = ()
        self.raw_block_marker_counts = {}
        self.finger_detection_enabled = False
        self.finger_marker_specs = {
            "index": "pink",
            "middle": "green",
            "ring": "red",
            "little": "purple",
        }
        self.finger_marker_min_area = 24.0
        self.raw_finger_markers = {}
        self.finger_fret_estimates = {}
        self.last_marker_detection = None
        self.last_marker_block_start = None
        self.raw_marker_detection = None
        self.stable_marker_detection = None
        self.marker_candidate_key = None
        self.marker_candidate_frames = 0
        self.marker_missing_frames = 0
        self.marker_stable_frames = 8
        self.marker_min_confidence = 0.25
        self.marker_hold_frames = 30
        self.marker_scan_pending = False
        self.detected_marker_colors = set()
        self.marker_scan_completed = False
        self.marker_scan_confirm_frames = 0
        self.marker_only = False
        self.marker_roi = None
        self.marker_roi_polygon = None
        self.marker_roi_user_defined = False
        self.marker_auto_roi_enabled = True
        self.marker_auto_roi_margin = 90
        self.marker_corner_count = 4
        self.marker_roi_selection_pending = False
        self.marker_roi_points = []
        self.marker_setup_pending = False
        self.marker_setup_last_signature = None
        self.marker_points = []
        self.drag_marker_index = None
        self.marker_point_tracking_requested = False
        self.marker_points_locked = False
        self.reference_marker_points = []
        self.marker_homography_enabled = True
        self.marker_homography_active = False
        self.marker_homography_inliers = 0
        self.marker_homography_min_points = 4
        self.marker_homography_min_inliers = 6
        self.marker_homography_max_median_jump = 70.0
        self.marker_homography_smoothing = 0.45
        self.marker_homography_roi_margin = 60
        self.occlusion_hand_enabled = True
        self.occlusion_hand_min_hidden = 2
        self.occlusion_hand_fret_radius = 1
        self.hidden_marker_points = []
        self.occlusion_hand_estimate = None
        self.background_hand_enabled = True
        self.background_hand_reference = None
        self.background_hand_canonical_points = {}
        self.background_hand_homography = None
        self.background_hand_estimate = None
        self.background_hand_threshold = 10
        self.background_hand_min_area = 80.0
        self.background_hand_fret_radius = 1
        self.background_hand_last_area = 0.0
        self.background_hand_largest_area = 0.0
        self.background_hand_changed_pixels = 0
        self.audio_hand_match_enabled = True
        self.audio_hand_match_result = None
        self.previous_audio_hand_position = None
        self.marker_tracking_enabled = False
        self.previous_marker_gray_frame = None
        self.marker_tracking_lost_frames = 0
        self.nut_anchor_enabled = False
        self.nut_anchor_colors = ()
        self.nut_anchor_block_color = "green"
        self.nut_anchor_min_area = 30.0
        self.nut_anchor_roi_margin = 80
        self.nut_anchor_smoothing = 0.28
        self.nut_anchor_max_step = 45.0
        self.raw_nut_anchor_markers = {}
        self.reference_nut_anchor_points = {}
        self.nut_anchor_transform_active = False
        self.nut_anchor_transform_delta = (0.0, 0.0)
        self.pick_detection_enabled = False
        self.pick_marker_color = "pink"
        self.pick_marker_min_area = 20.0
        self.raw_pick_marker = None
        self.pick_roi = None
        self.pick_roi_polygon = None
        self.pick_roi_points = []
        self.pick_roi_selection_pending = False
        self.pick_string_number = None
        self.pick_string_confidence = 0.0
        self.pick_audio_position = None

    def refresh_pitch_match_result(self) -> None:
        self.last_match_result = None

        if (
            self.midi_pitch is None
            or self.last_result is None
            or not self.last_result["inside_fretboard"]
        ):
            return

        self.last_match_result = match_camera_estimate_with_midi_pitch(
            estimated_position=self.last_result,
            midi_pitch=self.midi_pitch,
            start_fret=self.config.start_fret,
            visible_frets=self.config.visible_frets,
        )

    def get_active_hand_estimate(self) -> dict | None:
        if self.background_hand_enabled and self.background_hand_estimate is not None:
            return self.background_hand_estimate

        if self.occlusion_hand_estimate is not None:
            return {
                "fret_number": self.occlusion_hand_estimate["fret_number"],
                "start_fret": max(
                    0,
                    self.occlusion_hand_estimate["fret_number"]
                    - self.occlusion_hand_fret_radius,
                ),
                "end_fret": min(
                    24,
                    self.occlusion_hand_estimate["fret_number"]
                    + self.occlusion_hand_fret_radius,
                ),
                "source": "occlusion",
            }

        return None

    def update_audio_hand_match(self) -> None:
        self.audio_hand_match_result = None

        if not self.audio_hand_match_enabled or self.midi_pitch is None:
            return

        candidates = find_positions_for_midi_pitch(
            midi_pitch=self.midi_pitch,
            start_fret=0,
            visible_frets=25,
        )
        if not candidates:
            self.previous_audio_hand_position = None
            return

        hand_fret = None
        start_fret = None
        end_fret = None
        in_range_candidates = []
        hand_estimate = self.get_active_hand_estimate()
        if hand_estimate is not None:
            hand_fret = hand_estimate["fret_number"]
            start_fret = hand_estimate["start_fret"]
            end_fret = hand_estimate["end_fret"]
            in_range_candidates = [
                candidate
                for candidate in candidates
                if start_fret <= candidate["fret_number"] <= end_fret
            ]

        previous_position = self.previous_audio_hand_position

        def continuity_score(candidate):
            if previous_position is None:
                return (999, 999, 999)

            return (
                abs(candidate["fret_number"] - previous_position["fret_number"]),
                abs(candidate["string_number"] - previous_position["string_number"]),
                abs(candidate["fret_number"] - hand_fret) if hand_fret is not None else 0,
            )

        def hand_score(candidate):
            if hand_fret is None:
                return (0, candidate["string_number"], candidate["fret_number"])

            return (
                abs(candidate["fret_number"] - hand_fret),
                candidate["string_number"],
                candidate["fret_number"],
            )

        def select_candidate(candidate_pool):
            if previous_position is not None:
                return min(
                    candidate_pool,
                    key=lambda candidate: (continuity_score(candidate), hand_score(candidate)),
                )

            return min(candidate_pool, key=hand_score)

        selected_position = None
        status = "no_hand_range_candidate"

        if in_range_candidates:
            selected_position = select_candidate(in_range_candidates)
            if len(in_range_candidates) == 1:
                status = "single_hand_range_candidate"
            elif previous_position is not None:
                status = "nearest_previous_in_hand_range"
            else:
                status = "nearest_hand_range_candidate"
        elif previous_position is not None:
            selected_position = select_candidate(candidates)
            if hand_fret is None:
                status = "nearest_previous_no_hand"
            else:
                status = "nearest_previous_outside_hand_range"
        elif hand_fret is not None:
            selected_position = select_candidate(candidates)
            status = "nearest_outside_hand_range"

        self.audio_hand_match_result = {
            "status": status,
            "midi_pitch": self.midi_pitch,
            "note_name": midi_pitch_to_note_name(self.midi_pitch),
            "hand_start_fret": start_fret,
            "hand_end_fret": end_fret,
            "hand_fret": hand_fret,
            "candidate_positions": candidates,
            "in_range_candidates": in_range_candidates,
            "selected_position": selected_position,
        }

        if selected_position is not None:
            self.previous_audio_hand_position = dict(selected_position)
            self.last_result = {
                "inside_fretboard": True,
                "string_number": selected_position["string_number"],
                "fret_number": selected_position["fret_number"],
                "source": "audio_hand",
            }
            self.config.start_fret = fretboard_display.get_current_block_start(
                selected_position["fret_number"],
                block_size=self.display_block_size,
            )
            self.config.visible_frets = self.display_block_size

    def adjust_midi_pitch(self, semitone_delta: int) -> None:
        self.pitch_source = "manual"

        if self.midi_pitch is None:
            self.midi_pitch = 62
        else:
            self.midi_pitch = max(0, min(127, self.midi_pitch + semitone_delta))

        self.refresh_pitch_match_result()
        self.update_pick_audio_position()
        self.update_audio_hand_match()

        print(
            "manual MIDI pitch: "
            f"{midi_pitch_to_note_name(self.midi_pitch)} (MIDI {self.midi_pitch})"
        )

        if self.last_match_result is not None:
            print(format_pitch_match_result(self.last_match_result))

    def update_audio_pitch(self, detection) -> None:
        if detection is None:
            return

        self.pitch_source = "audio"
        self.last_audio_detection = detection

        if self.midi_pitch == detection.midi_pitch:
            return

        self.midi_pitch = detection.midi_pitch
        self.refresh_pitch_match_result()
        self.update_pick_audio_position()
        self.update_audio_hand_match()
        print(
            "audio pitch: "
            f"{detection.note_name} (MIDI {detection.midi_pitch}) "
            f"{detection.frequency_hz:.1f} Hz "
            f"confidence={detection.confidence:.2f}"
        )

        if self.last_match_result is not None:
            print(format_pitch_match_result(self.last_match_result))

    def set_virtual_result(self, virtual_x: float, virtual_y: float) -> None:
        result = estimate_string_and_fret(
            virtual_x=virtual_x,
            virtual_y=virtual_y,
            config=self.config,
            fret_boundaries=self.fret_boundaries,
        )

        self.last_click = None
        self.last_result = result
        self.refresh_pitch_match_result()

        if self.last_match_result is not None:
            print(format_pitch_match_result(self.last_match_result))

        if result["inside_fretboard"]:
            print(
                f'virtual fretboard: {result["string_number"]} string '
                f'{result["fret_number"]} fret'
            )

    def set_display_position_result(self, position: dict) -> None:
        self.last_click = None
        self.last_result = position

        if position["inside_fretboard"]:
            self.config.start_fret = fretboard_display.get_current_block_start(
                position["fret_number"],
                block_size=self.display_block_size,
            )

        self.refresh_pitch_match_result()

        if self.last_match_result is not None:
            print(format_pitch_match_result(self.last_match_result))

        if position["inside_fretboard"]:
            print(
                f'virtual fretboard: {position["string_number"]} string '
                f'{position["fret_number"]} fret'
            )

    def toggle_display_mode(self) -> None:
        current_index = DISPLAY_MODES.index(self.display_mode)
        self.display_mode = DISPLAY_MODES[(current_index + 1) % len(DISPLAY_MODES)]
        print(f"display mode: {self.display_mode}")

    def reset_marker_stability(self) -> None:
        self.raw_marker_detection = None
        self.stable_marker_detection = None
        self.last_marker_detection = None
        self.last_marker_block_start = None
        self.marker_candidate_key = None
        self.marker_candidate_frames = 0
        self.marker_missing_frames = 0
        self.raw_hand_markers = ()
        self.raw_block_marker_counts = {}
        self.raw_finger_markers = {}
        self.finger_fret_estimates = {}

    def reset_marker_tracking(self) -> None:
        self.marker_setup_pending = False
        self.marker_points = []
        self.drag_marker_index = None
        self.marker_points_locked = False
        self.reference_marker_points = []
        self.marker_homography_active = False
        self.marker_homography_inliers = 0
        self.marker_tracking_enabled = False
        self.previous_marker_gray_frame = None
        self.background_hand_reference = None
        self.background_hand_canonical_points = {}
        self.background_hand_homography = None
        self.background_hand_estimate = None
        self.background_hand_last_area = 0.0
        self.background_hand_largest_area = 0.0
        self.background_hand_changed_pixels = 0
        self.marker_setup_last_signature = None
        self.reference_nut_anchor_points = {}
        self.nut_anchor_transform_active = False
        self.nut_anchor_transform_delta = (0.0, 0.0)

    def begin_pick_roi_selection(self) -> None:
        self.pick_roi = None
        self.pick_roi_polygon = None
        self.pick_roi_points = []
        self.pick_roi_selection_pending = True
        self.raw_pick_marker = None
        self.pick_string_number = None
        self.pick_audio_position = None
        print("pick ROI: click 4 corners around the picking-string area")

    def add_pick_roi_point(self, point: tuple[int, int]) -> None:
        if not self.pick_roi_selection_pending or len(self.pick_roi_points) >= 4:
            return

        self.pick_roi_points.append(point)
        print(f"pick ROI point {len(self.pick_roi_points)}/4: {point}")

        if len(self.pick_roi_points) == 4:
            self.confirm_pick_roi_selection()

    def confirm_pick_roi_selection(self) -> None:
        if not self.pick_roi_selection_pending:
            return

        if len(self.pick_roi_points) != 4:
            print("pick ROI needs 4 points")
            return

        ordered_points = order_quad_points(self.pick_roi_points)
        self.pick_roi_polygon = tuple(
            (int(point[0]), int(point[1])) for point in ordered_points
        )
        xs = [point[0] for point in self.pick_roi_polygon]
        ys = [point[1] for point in self.pick_roi_polygon]
        self.pick_roi = (
            min(xs),
            min(ys),
            max(xs) - min(xs),
            max(ys) - min(ys),
        )
        self.pick_roi_selection_pending = False
        print(f"pick ROI set: {self.pick_roi_polygon}")

    def is_pick_setup_active(self) -> bool:
        return self.pick_detection_enabled and self.pick_roi_selection_pending

    def update_pick_detection(self, frame) -> None:
        self.raw_pick_marker = None
        self.pick_string_number = None
        self.pick_string_confidence = 0.0

        if (
            not self.pick_detection_enabled
            or self.pick_roi_selection_pending
            or self.pick_roi_polygon is None
        ):
            return

        detections = detect_markers_for_color(
            frame,
            color_name=self.pick_marker_color,
            min_area=self.pick_marker_min_area,
            roi=self.pick_roi,
            roi_polygon=self.pick_roi_polygon,
        )

        if not detections:
            return

        self.raw_pick_marker = detections[0]
        self.pick_string_number, self.pick_string_confidence = (
            self.estimate_pick_string(self.raw_pick_marker.center)
        )
        self.update_pick_audio_position()

    def estimate_pick_string(self, point: tuple[int, int]) -> tuple[int | None, float]:
        if self.pick_roi_polygon is None:
            return None, 0.0

        src_points = order_quad_points(self.pick_roi_polygon)
        dst_width = 1000.0
        dst_height = 600.0
        dst_points = np.array(
            [
                [0.0, 0.0],
                [dst_width, 0.0],
                [dst_width, dst_height],
                [0.0, dst_height],
            ],
            dtype=np.float32,
        )
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        warped_point = cv2.perspectiveTransform(
            np.array([[[point[0], point[1]]]], dtype=np.float32),
            perspective_matrix,
        )[0][0]
        y_ratio = float(np.clip(warped_point[1] / dst_height, 0.0, 0.999))
        string_index = min(max(int(y_ratio * 6), 0), 5)
        string_number = self.config.string_order_top_to_bottom[string_index]
        lane_center = (string_index + 0.5) / 6.0
        distance_to_center = abs(y_ratio - lane_center)
        confidence = float(max(0.0, 1.0 - distance_to_center / (0.5 / 6.0)))
        return string_number, confidence

    def update_pick_audio_position(self) -> None:
        self.pick_audio_position = None

        if self.pick_string_number is None or self.midi_pitch is None:
            return

        open_pitch = STANDARD_TUNING_OPEN_MIDI.get(self.pick_string_number)

        if open_pitch is None:
            return

        fret_number = self.midi_pitch - open_pitch

        if fret_number < 0 or fret_number > 24:
            return

        self.pick_audio_position = {
            "inside_fretboard": True,
            "string_number": self.pick_string_number,
            "fret_number": fret_number,
            "display_fret_index": None,
            "source": "pick_audio",
            "confidence": self.pick_string_confidence,
        }
        self.last_click = None
        self.last_result = self.pick_audio_position
        self.config.start_fret = fretboard_display.get_current_block_start(
            fret_number,
            block_size=self.display_block_size,
        )
        self.config.visible_frets = self.display_block_size
        self.refresh_pitch_match_result()

    def marker_detection_key(self, detection) -> tuple | None:
        if detection is None:
            return None

        return (
            detection.block_name,
            detection.start_fret,
            detection.marker_color,
        )

    def accept_stable_marker_detection(self, detection) -> None:
        self.stable_marker_detection = detection
        self.last_marker_detection = detection
        self.marker_missing_frames = 0

        if detection.start_fret != self.last_marker_block_start:
            print(
                "marker block: "
                f"{detection.block_name} "
                f"start_fret={detection.start_fret} "
                f"color={detection.marker_color} "
                f"confidence={detection.confidence:.2f}"
            )

        self.last_marker_block_start = detection.start_fret
        self.config.start_fret = detection.start_fret
        self.config.visible_frets = detection.visible_frets
        self.refresh_pitch_match_result()

    def update_stable_marker_detection(self, detection) -> None:
        self.raw_marker_detection = detection

        if detection is None or detection.confidence < self.marker_min_confidence:
            self.marker_candidate_key = None
            self.marker_candidate_frames = 0
            self.marker_missing_frames += 1

            if self.marker_missing_frames > self.marker_hold_frames:
                self.stable_marker_detection = None
                self.last_marker_detection = None
            else:
                self.last_marker_detection = self.stable_marker_detection

            return

        detection_key = self.marker_detection_key(detection)

        if detection_key == self.marker_candidate_key:
            self.marker_candidate_frames += 1
        else:
            self.marker_candidate_key = detection_key
            self.marker_candidate_frames = 1

        if self.marker_candidate_frames >= self.marker_stable_frames:
            self.accept_stable_marker_detection(detection)
        else:
            self.last_marker_detection = self.stable_marker_detection

    def update_marker_detection(self, frame, min_area: float) -> None:
        self.raw_marker_detection = None

        if not self.marker_detection_enabled or self.mode != "mapping":
            return

        if self.marker_roi_selection_pending or self.marker_setup_pending:
            return

        if self.hand_marker_color is not None:
            self.raw_hand_markers = tuple(
                detect_markers_for_color(
                    frame,
                    color_name=self.hand_marker_color,
                    min_area=self.hand_marker_min_area,
                    roi=self.marker_roi,
                    roi_polygon=self.marker_roi_polygon,
                )[: self.hand_marker_count]
            )

        if self.finger_detection_enabled:
            self.update_finger_markers(frame)

        self.raw_block_marker_counts = {
            color_name: len(markers)
            for color_name, markers in detect_all_markers(
                frame,
                color_names=tuple(sorted(self.required_marker_colors())),
                min_area=min_area,
                roi=self.marker_roi,
                roi_polygon=self.marker_roi_polygon,
            ).items()
        }

        if self.marker_points_locked and self.marker_points:
            detection = self.detect_block_from_tracked_markers(frame)
            self.raw_marker_detection = detection
            self.stable_marker_detection = detection
            self.last_marker_detection = detection

            if detection is not None:
                self.last_marker_block_start = detection.start_fret
                self.config.start_fret = detection.start_fret
                self.config.visible_frets = detection.visible_frets
        else:
            self.update_marker_scan(frame, min_area=min_area)
            detection = detect_fretboard_block(
                frame,
                hand_marker_color=self.hand_marker_color,
                required_hand_marker_count=self.hand_marker_count,
                hand_marker_min_area=self.hand_marker_min_area,
                min_area=min_area,
                roi=self.marker_roi,
                roi_polygon=self.marker_roi_polygon,
                required_corner_count=self.marker_corner_count,
                current_start_fret=self.last_marker_block_start or self.config.start_fret,
            )
            self.update_stable_marker_detection(detection)
        self.update_finger_fret_estimates()

    def update_nut_anchor_detection(self, frame) -> None:
        self.raw_nut_anchor_markers = {}
        self.nut_anchor_transform_active = False

        if not self.nut_anchor_enabled:
            return

        anchor_roi = expand_roi_for_frame(
            self.marker_roi,
            self.nut_anchor_roi_margin,
            frame.shape,
        )

        if self.nut_anchor_colors:
            for color_name in self.nut_anchor_colors:
                detections = detect_markers_for_color(
                    frame,
                    color_name=color_name,
                    min_area=self.nut_anchor_min_area,
                    roi=anchor_roi,
                    roi_polygon=None,
                )

                if detections:
                    self.raw_nut_anchor_markers[color_name] = detections[0]
        else:
            detections = detect_markers_for_color(
                frame,
                color_name=self.nut_anchor_block_color,
                min_area=self.nut_anchor_min_area,
                roi=anchor_roi,
                roi_polygon=None,
            )
            anchor_markers = self.choose_same_color_anchor_markers(detections, count=2)

            for index, marker in enumerate(anchor_markers, start=1):
                self.raw_nut_anchor_markers[f"{self.nut_anchor_block_color}_{index}"] = marker

        if self.marker_points_locked:
            homography_updated = self.update_marker_points_from_visible_markers(frame)

            if not homography_updated:
                self.update_marker_points_from_nut_anchor(frame.shape)

    def choose_same_color_anchor_markers(self, markers, count: int = 2):
        largest_markers = sorted(markers, key=lambda marker: -marker.area)[:count]
        return tuple(
            sorted(
                largest_markers,
                key=lambda marker: (
                    marker.center[0],
                    marker.center[1],
                ),
            )
        )

    def get_reference_marker_by_key(self) -> dict[tuple[str, int], tuple[int, int]]:
        reference_points = self.reference_marker_points or self.marker_points
        return {
            (marker_point["color"], marker_point["index"]): marker_point["center"]
            for marker_point in reference_points
        }

    def detect_edge_pair_markers_for_homography(self, frame):
        marker_roi = expand_roi_for_frame(
            self.marker_roi,
            self.marker_homography_roi_margin,
            frame.shape,
        )
        marker_map = {}

        for color_name, expected_count in EDGE_PAIR_MARKER_COUNTS.items():
            detections = detect_markers_for_color(
                frame,
                color_name=color_name,
                min_area=self.nut_anchor_min_area,
                roi=marker_roi,
                roi_polygon=None,
            )
            selected_detections = sorted(
                detections[:expected_count],
                key=lambda detection: (
                    detection.center[1],
                    detection.center[0],
                ),
            )

            for index, detection in enumerate(selected_detections, start=1):
                marker_map[(color_name, index)] = detection

        return marker_map

    def estimate_fret_from_marker_center(self, center: tuple[float, float]) -> dict | None:
        if len(self.marker_points) < 2:
            return None

        points = np.array(
            [marker_point["center"] for marker_point in self.marker_points],
            dtype=np.float32,
        )
        centered_points = points - points.mean(axis=0)

        if len(centered_points) < 2:
            return None

        _, _, vh = np.linalg.svd(centered_points, full_matrices=False)
        axis = vh[0]
        projections = points @ axis
        min_projection = float(projections.min())
        max_projection = float(projections.max())
        span = max_projection - min_projection

        if span <= 1.0:
            return None

        center_projection = float(np.array(center, dtype=np.float32) @ axis)
        ratio = (center_projection - min_projection) / span
        ratio = max(0.0, min(0.999, ratio))
        fret_number = 1 + int(ratio * 12)
        return {
            "inside_fretboard": True,
            "fret_number": fret_number,
            "x_ratio": ratio,
        }

    def update_occlusion_hand_estimate(self, current_by_key: dict) -> None:
        self.hidden_marker_points = []
        self.occlusion_hand_estimate = None

        if not self.occlusion_hand_enabled or not self.marker_points_locked:
            self.update_audio_hand_match()
            return

        visible_keys = set(current_by_key.keys())

        for marker_point in self.marker_points:
            marker_key = (marker_point["color"], marker_point["index"])

            if marker_key not in visible_keys:
                self.hidden_marker_points.append(marker_point)

        if len(self.hidden_marker_points) < self.occlusion_hand_min_hidden:
            self.update_audio_hand_match()
            return

        hidden_centers = np.array(
            [marker_point["center"] for marker_point in self.hidden_marker_points],
            dtype=np.float32,
        )
        center = hidden_centers.mean(axis=0)
        spread = float(
            np.mean(
                np.linalg.norm(
                    hidden_centers - center,
                    axis=1,
                )
            )
        )
        fret_estimate = self.estimate_fret_from_marker_center(
            (float(center[0]), float(center[1]))
        )

        if fret_estimate is None:
            self.update_audio_hand_match()
            return

        confidence = min(
            1.0,
            0.25 + 0.18 * len(self.hidden_marker_points) + max(0.0, 0.35 - spread / 240.0),
        )
        self.occlusion_hand_estimate = {
            "center": (int(round(float(center[0]))), int(round(float(center[1])))),
            "fret_number": fret_estimate["fret_number"],
            "hidden_count": len(self.hidden_marker_points),
            "confidence": confidence,
            "spread": spread,
        }
        self.update_audio_hand_match()

    def warp_fretboard_for_background_hand(self, frame):
        if not self.background_hand_canonical_points:
            return None, None

        homography = build_marker_to_canonical_homography(
            self.marker_points,
            self.background_hand_canonical_points,
        )

        if homography is None:
            return None, None

        width, height = BACKGROUND_HAND_WARP_SIZE
        warped_frame = cv2.warpPerspective(frame, homography, (width, height))
        return warped_frame, homography

    def capture_background_hand_reference(self, frame) -> None:
        if not self.marker_points_locked or len(self.marker_points) < 4:
            print("background hand: lock marker points first.")
            return

        canonical_points = make_marker_canonical_points(self.marker_points)

        if len(canonical_points) < 4:
            print("background hand: could not build canonical fretboard points.")
            return

        self.background_hand_canonical_points = canonical_points
        warped_frame, homography = self.warp_fretboard_for_background_hand(frame)

        if warped_frame is None:
            print("background hand: could not warp fretboard.")
            return

        self.background_hand_reference = cv2.GaussianBlur(warped_frame, (7, 7), 0)
        self.background_hand_homography = homography
        self.background_hand_estimate = None
        self.background_hand_last_area = 0.0
        self.background_hand_largest_area = 0.0
        self.background_hand_changed_pixels = 0
        print("background hand: reference captured. Keep playing; changed area will be tracked.")

    def update_background_hand_estimate(self, frame) -> None:
        self.background_hand_estimate = None

        if (
            not self.background_hand_enabled
            or self.background_hand_reference is None
            or not self.marker_points_locked
        ):
            self.update_audio_hand_match()
            return

        warped_frame, homography = self.warp_fretboard_for_background_hand(frame)

        if warped_frame is None:
            self.update_audio_hand_match()
            return

        blurred_frame = cv2.GaussianBlur(warped_frame, (7, 7), 0)
        if len(self.background_hand_reference.shape) == 2:
            gray_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)
            diff_frame = cv2.absdiff(gray_frame, self.background_hand_reference)
        else:
            color_diff = cv2.absdiff(blurred_frame, self.background_hand_reference)
            diff_frame = np.max(color_diff, axis=2).astype(np.uint8)
        _, mask = cv2.threshold(
            diff_frame,
            int(self.background_hand_threshold),
            255,
            cv2.THRESH_BINARY,
        )
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        self.background_hand_changed_pixels = int(cv2.countNonZero(mask))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contour_areas = [float(cv2.contourArea(contour)) for contour in contours]
        self.background_hand_largest_area = max(all_contour_areas, default=0.0)
        contours = [
            contour
            for contour in contours
            if cv2.contourArea(contour) >= self.background_hand_min_area
        ]
        self.background_hand_last_area = float(
            sum(cv2.contourArea(contour) for contour in contours)
        )

        if not contours:
            nonzero_points = cv2.findNonZero(mask)

            if (
                nonzero_points is not None
                and self.background_hand_changed_pixels >= self.background_hand_min_area
            ):
                x, y, width, height = cv2.boundingRect(nonzero_points)
                area = float(self.background_hand_changed_pixels)
                contours = None
            else:
                self.background_hand_homography = homography
                self.update_audio_hand_match()
                return
        else:
            all_points = np.vstack(contours)
            x, y, width, height = cv2.boundingRect(all_points)
            area = self.background_hand_last_area

        if area <= 0.0:
            self.background_hand_homography = homography
            self.update_audio_hand_match()
            return

        warp_width, warp_height = BACKGROUND_HAND_WARP_SIZE
        center_x = x + width / 2.0
        center_y = y + height / 2.0
        x_ratio = max(0.0, min(0.999, center_x / max(1, warp_width)))
        left_ratio = max(0.0, min(0.999, x / max(1, warp_width)))
        right_ratio = max(0.0, min(0.999, (x + width) / max(1, warp_width)))
        fret_number = 1 + int(x_ratio * 12)
        detected_start_fret = 1 + int(left_ratio * 12)
        detected_end_fret = 1 + int(right_ratio * 12)
        start_fret = max(0, detected_start_fret - self.background_hand_fret_radius)
        end_fret = min(24, detected_end_fret + self.background_hand_fret_radius)
        confidence = min(1.0, area / (warp_width * warp_height * 0.18))
        self.background_hand_homography = homography
        self.background_hand_estimate = {
            "source": "background",
            "fret_number": fret_number,
            "start_fret": start_fret,
            "end_fret": end_fret,
            "detected_start_fret": detected_start_fret,
            "detected_end_fret": detected_end_fret,
            "confidence": confidence,
            "area": area,
            "canonical_rect": (x, y, x + width, y + height),
            "canonical_center": (center_x, center_y),
        }
        self.update_audio_hand_match()

    def update_marker_points_from_visible_markers(self, frame) -> bool:
        self.marker_homography_active = False
        self.marker_homography_inliers = 0

        if (
            not self.marker_homography_enabled
            or not self.reference_marker_points
            or len(self.reference_marker_points) < self.marker_homography_min_points
        ):
            return False

        reference_by_key = self.get_reference_marker_by_key()
        current_by_key = self.detect_edge_pair_markers_for_homography(frame)
        self.update_occlusion_hand_estimate(current_by_key)
        reference_points = []
        current_points = []

        for marker_key, reference_center in reference_by_key.items():
            current_detection = current_by_key.get(marker_key)

            if current_detection is None:
                continue

            reference_points.append(reference_center)
            current_points.append(current_detection.center)

        if len(reference_points) < self.marker_homography_min_points:
            return False

        homography, inlier_mask = cv2.findHomography(
            np.array(reference_points, dtype=np.float32),
            np.array(current_points, dtype=np.float32),
            cv2.RANSAC,
            8.0,
        )

        if homography is None:
            return False

        inlier_count = int(inlier_mask.sum()) if inlier_mask is not None else len(reference_points)

        if inlier_count < self.marker_homography_min_inliers:
            return False

        reference_centers = np.array(
            [
                [[marker_point["center"][0], marker_point["center"][1]]]
                for marker_point in self.reference_marker_points
            ],
            dtype=np.float32,
        )
        transformed_centers = cv2.perspectiveTransform(reference_centers, homography)
        jump_distances = []

        for marker_point, transformed_center in zip(self.marker_points, transformed_centers):
            current_x, current_y = marker_point["center"]
            target_x = float(transformed_center[0][0])
            target_y = float(transformed_center[0][1])
            jump_distances.append(float(np.hypot(target_x - current_x, target_y - current_y)))

        median_jump = float(np.median(jump_distances)) if jump_distances else 0.0

        if median_jump > self.marker_homography_max_median_jump:
            return False

        blend = self.marker_homography_smoothing

        for marker_point, transformed_center in zip(self.marker_points, transformed_centers):
            current_x, current_y = marker_point["center"]
            target_x = float(transformed_center[0][0])
            target_y = float(transformed_center[0][1])
            marker_point["center"] = (
                int(round(current_x + (target_x - current_x) * blend)),
                int(round(current_y + (target_y - current_y) * blend)),
            )

        self.marker_homography_active = True
        self.marker_homography_inliers = inlier_count
        if len(self.marker_points) >= sum(EDGE_PAIR_MARKER_COUNTS.values()):
            self.update_marker_auto_roi(frame.shape)
        return True

    def nut_anchor_expected_count(self) -> int:
        return len(self.nut_anchor_colors) if self.nut_anchor_colors else 2

    def nut_anchor_description(self) -> str:
        if self.nut_anchor_colors:
            return ",".join(self.nut_anchor_colors)

        return f"{self.nut_anchor_block_color} same-color x2"

    def current_nut_anchor_points(self) -> dict[str, tuple[int, int]]:
        return {
            color_name: marker.center
            for color_name, marker in self.raw_nut_anchor_markers.items()
        }

    def capture_nut_anchor_reference(self) -> None:
        self.reference_marker_points = copy_marker_points(self.marker_points)
        self.reference_nut_anchor_points = {}

        if not self.nut_anchor_enabled:
            return

        current_points = self.current_nut_anchor_points()

        if len(current_points) < 2:
            print("nut anchor: reference not captured. Need at least 2 visible anchors.")
            return

        self.reference_nut_anchor_points = dict(current_points)
        colors = ",".join(self.reference_nut_anchor_points.keys())
        print(f"nut anchor: reference captured ({colors})")

    def make_nut_anchor_transform(self) -> np.ndarray | None:
        if len(self.reference_nut_anchor_points) < 2 or len(self.raw_nut_anchor_markers) < 2:
            return None

        anchor_keys = (
            list(self.nut_anchor_colors)
            if self.nut_anchor_colors
            else list(self.reference_nut_anchor_points.keys())
        )
        common_colors = [
            color_name
            for color_name in anchor_keys
            if color_name in self.reference_nut_anchor_points
            and color_name in self.raw_nut_anchor_markers
        ]

        if len(common_colors) < 2:
            return None

        reference_points = np.array(
            [self.reference_nut_anchor_points[color_name] for color_name in common_colors],
            dtype=np.float32,
        )
        current_points = np.array(
            [self.raw_nut_anchor_markers[color_name].center for color_name in common_colors],
            dtype=np.float32,
        )
        reference_center = reference_points.mean(axis=0)
        current_center = current_points.mean(axis=0)
        self.nut_anchor_transform_delta = (
            float(current_center[0] - reference_center[0]),
            float(current_center[1] - reference_center[1]),
        )

        if len(common_colors) == 2:
            return estimate_similarity_from_two_points(
                (tuple(reference_points[0]), tuple(reference_points[1])),
                (tuple(current_points[0]), tuple(current_points[1])),
            )

        affine_matrix, inliers = cv2.estimateAffinePartial2D(
            reference_points,
            current_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=6.0,
        )

        if affine_matrix is None:
            return None

        if inliers is not None and int(inliers.sum()) < 2:
            return None

        return affine_matrix.astype(np.float32)

    def update_marker_points_from_nut_anchor(self, frame_shape=None) -> None:
        if not self.reference_marker_points:
            return

        affine_matrix = self.make_nut_anchor_transform()

        if affine_matrix is None:
            return

        target_centers = []

        for reference_marker_point in self.reference_marker_points:
            reference_center = np.array(
                [reference_marker_point["center"][0], reference_marker_point["center"][1], 1.0],
                dtype=np.float32,
            )
            transformed_center = affine_matrix @ reference_center
            target_centers.append(
                (
                    float(transformed_center[0]),
                    float(transformed_center[1]),
                )
            )

        if len(target_centers) != len(self.marker_points):
            return

        max_distance = 0.0

        for marker_point, target_center in zip(self.marker_points, target_centers):
            current_x, current_y = marker_point["center"]
            distance = float(np.hypot(target_center[0] - current_x, target_center[1] - current_y))
            max_distance = max(max_distance, distance)

        if max_distance > self.nut_anchor_max_step:
            blend = min(
                self.nut_anchor_smoothing,
                max(0.05, self.nut_anchor_max_step / max_distance),
            )
        else:
            blend = self.nut_anchor_smoothing

        for marker_point, target_center in zip(self.marker_points, target_centers):
            current_x, current_y = marker_point["center"]
            next_x = current_x + (target_center[0] - current_x) * blend
            next_y = current_y + (target_center[1] - current_y) * blend
            marker_point["center"] = (
                int(round(next_x)),
                int(round(next_y)),
            )

        self.nut_anchor_transform_active = True
        if frame_shape is not None:
            self.update_marker_auto_roi(frame_shape)

    def update_finger_markers(self, frame) -> None:
        self.raw_finger_markers = {}

        for finger_name, color_name in self.finger_marker_specs.items():
            if color_name not in COLOR_RANGES:
                continue

            detections = detect_markers_for_color(
                frame,
                color_name=color_name,
                min_area=self.finger_marker_min_area,
                roi=self.marker_roi,
                roi_polygon=self.marker_roi_polygon,
            )

            if detections:
                self.raw_finger_markers[finger_name] = detections[0]

    def estimate_fret_from_point(
        self,
        point: tuple[int, int],
        block_candidates: list[FretboardBlockDetection],
    ) -> dict | None:
        if not block_candidates:
            return None

        x, y = point
        containing_candidates = [
            candidate
            for candidate in block_candidates
            if candidate.block_rect[0] - 10 <= x <= candidate.block_rect[0] + candidate.block_rect[2] + 10
            and candidate.block_rect[1] - 20 <= y <= candidate.block_rect[1] + candidate.block_rect[3] + 20
        ]

        if containing_candidates:
            candidate = min(
                containing_candidates,
                key=lambda item: abs(
                    x - (item.block_rect[0] + item.block_rect[2] / 2.0)
                ),
            )
        else:
            candidate = min(
                block_candidates,
                key=lambda item: abs(
                    x - (item.block_rect[0] + item.block_rect[2] / 2.0)
                ),
            )

        left, top, width, height = candidate.block_rect

        if width <= 0:
            return None

        ratio = (x - left) / width
        ratio = max(0.0, min(0.999, ratio))
        fret_offset = int(ratio * candidate.visible_frets)
        fret_number = candidate.start_fret + fret_offset

        return {
            "block_name": candidate.block_name,
            "fret_number": fret_number,
            "marker_color": candidate.marker_color,
            "x_ratio": ratio,
        }

    def update_finger_fret_estimates(self) -> None:
        self.finger_fret_estimates = {}

        if not self.finger_detection_enabled or not self.raw_finger_markers:
            return

        block_candidates = self.make_block_candidates_from_marker_points()

        if not block_candidates:
            return

        for finger_name, marker in self.raw_finger_markers.items():
            estimate = self.estimate_fret_from_point(marker.center, block_candidates)

            if estimate is not None:
                self.finger_fret_estimates[finger_name] = estimate

        if not self.finger_fret_estimates:
            return

        anchor_estimate = self.finger_fret_estimates.get("index")

        if anchor_estimate is None:
            anchor_estimate = min(
                self.finger_fret_estimates.values(),
                key=lambda estimate: estimate["fret_number"],
            )

        self.config.start_fret = fretboard_display.get_current_block_start(
            anchor_estimate["fret_number"],
            block_size=self.display_block_size,
        )
        self.config.visible_frets = self.display_block_size

    def initialize_marker_setup_points(self, frame, min_area: float) -> None:
        marker_points = []
        marker_count_summary = []
        marker_count_signature = []

        for color_name, expected_count in EDGE_PAIR_MARKER_COUNTS.items():
            all_detections = detect_markers_for_color(
                frame,
                color_name=color_name,
                min_area=min_area,
                roi=self.marker_roi,
                roi_polygon=self.marker_roi_polygon,
            )
            marker_count_signature.append((color_name, len(all_detections)))
            marker_count_summary.append(
                f"{color_name}:{len(all_detections)}/{expected_count}"
            )
            detections = all_detections[:expected_count]

            detections = sorted(
                detections,
                key=lambda detection: (detection.center[1], detection.center[0]),
            )

            for index, detection in enumerate(detections, start=1):
                marker_points.append(
                    {
                        "color": color_name,
                        "index": index,
                        "center": detection.center,
                    }
                )

        self.marker_points = marker_points
        self.marker_setup_pending = True
        self.marker_points_locked = False
        self.marker_tracking_enabled = False
        self.previous_marker_gray_frame = None
        self.update_marker_auto_roi(frame.shape)
        self.reset_marker_stability()
        marker_signature = tuple(marker_count_signature)

        if marker_signature != self.marker_setup_last_signature:
            self.marker_setup_last_signature = marker_signature
            print(f"marker setup: detected {len(self.marker_points)}/12 edge-pair points")
            print(f"marker setup counts: {' '.join(marker_count_summary)}")

            if len(self.marker_points) > 0:
                print("Drag wrong points. Enter/Space: lock points / r: redetect")

    def update_marker_auto_roi(self, frame_shape) -> None:
        if (
            not self.marker_auto_roi_enabled
            or self.marker_roi_user_defined
            or not self.marker_points
        ):
            return

        roi = make_roi_from_points(
            [marker_point["center"] for marker_point in self.marker_points],
            frame_shape,
            self.marker_auto_roi_margin,
        )

        if roi is None:
            return

        self.marker_roi = roi
        self.marker_roi_polygon = None

    def find_nearest_marker_point(self, x: int, y: int, max_distance: float = 28.0):
        best_index = None
        best_distance = max_distance

        for index, marker_point in enumerate(self.marker_points):
            point_x, point_y = marker_point["center"]
            distance = float(np.hypot(point_x - x, point_y - y))

            if distance <= best_distance:
                best_index = index
                best_distance = distance

        return best_index

    def find_next_missing_marker_slot(self):
        for color_name, expected_count in EDGE_PAIR_MARKER_COUNTS.items():
            used_indices = {
                marker_point["index"]
                for marker_point in self.marker_points
                if marker_point["color"] == color_name
            }

            for index in range(1, expected_count + 1):
                if index not in used_indices:
                    return color_name, index

        return None

    def add_missing_marker_point(self, x: int, y: int) -> None:
        marker_slot = self.find_next_missing_marker_slot()

        if marker_slot is None:
            return

        color_name, marker_index = marker_slot
        self.marker_points.append(
            {
                "color": color_name,
                "index": marker_index,
                "center": (x, y),
            }
        )
        print(f"added marker point: {color_name}_{marker_index} ({x}, {y})")

    def begin_marker_tracking(self, gray_frame) -> None:
        required_count = sum(EDGE_PAIR_MARKER_COUNTS.values())

        if len(self.marker_points) < required_count:
            print(f"marker setup needs {required_count} points, current={len(self.marker_points)}")
            return

        self.marker_setup_pending = False
        self.marker_points_locked = True
        self.marker_roi_points = []
        self.marker_tracking_enabled = self.marker_point_tracking_requested
        self.previous_marker_gray_frame = (
            gray_frame.copy() if self.marker_tracking_enabled else None
        )
        self.marker_tracking_lost_frames = 0
        self.background_hand_reference = None
        self.background_hand_canonical_points = {}
        self.background_hand_homography = None
        self.background_hand_estimate = None
        self.background_hand_last_area = 0.0
        self.background_hand_largest_area = 0.0
        self.background_hand_changed_pixels = 0
        self.update_marker_auto_roi(gray_frame.shape)
        self.reset_marker_stability()
        self.capture_nut_anchor_reference()
        print(f"marker auto ROI: {self.marker_roi}")
        if self.marker_tracking_enabled:
            print("marker point optical tracking started")
        else:
            print("marker points locked. Optical tracking is OFF.")

    def update_marker_point_tracking(self, gray_frame) -> None:
        if not self.marker_tracking_enabled or not self.marker_points:
            self.previous_marker_gray_frame = gray_frame
            return

        if self.previous_marker_gray_frame is None:
            self.previous_marker_gray_frame = gray_frame
            return

        old_points = np.array(
            [[marker_point["center"]] for marker_point in self.marker_points],
            dtype=np.float32,
        )
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_marker_gray_frame,
            gray_frame,
            old_points,
            None,
            winSize=(25, 25),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                20,
                0.03,
            ),
        )

        if new_points is None or status is None:
            self.marker_tracking_lost_frames += 1

            if self.marker_tracking_lost_frames > 45:
                self.marker_tracking_enabled = False
                self.marker_setup_pending = True
                print("marker tracking lost. Please adjust points again.")

            return

        valid_count = int(status.reshape(-1).sum())

        if valid_count == 0:
            self.marker_tracking_lost_frames += 1

            if self.marker_tracking_lost_frames > 45:
                self.marker_tracking_enabled = False
                self.marker_setup_pending = True
                print("marker tracking lost. Please adjust points again.")

            return

        self.marker_tracking_lost_frames = 0

        for index, marker_point in enumerate(self.marker_points):
            if status[index][0] != 1:
                continue

            point = new_points[index][0]
            marker_point["center"] = (int(point[0]), int(point[1]))

        self.previous_marker_gray_frame = gray_frame

    def make_marker_detection_from_point(self, marker_point: dict) -> MarkerDetection:
        center_x, center_y = marker_point["center"]
        return MarkerDetection(
            color_name=marker_point["color"],
            center=(center_x, center_y),
            area=1.0,
            bounding_box=(center_x - 8, center_y - 8, 16, 16),
            confidence=0.95,
        )

    def make_block_candidates_from_marker_points(self) -> list[FretboardBlockDetection]:
        if len(self.marker_points) < 4:
            return []

        markers = tuple(
            self.make_marker_detection_from_point(marker_point)
            for marker_point in self.marker_points
        )
        return [
            FretboardBlockDetection(
                block_name="1-12F",
                start_fret=1,
                visible_frets=12,
                marker_color="mixed",
                confidence=0.95,
                marker=markers[0],
                markers=markers,
                block_rect=make_block_rect(markers),
            )
        ]

    def detect_block_from_tracked_markers(self, frame):
        block_candidates = self.make_block_candidates_from_marker_points()

        if not block_candidates:
            return None

        if self.hand_marker_color is not None and len(self.raw_hand_markers) >= self.hand_marker_count:
            return choose_block_by_hand_markers(
                block_candidates,
                hand_markers=tuple(self.raw_hand_markers[: self.hand_marker_count]),
                frame_shape=frame.shape,
                current_start_fret=self.last_marker_block_start or self.config.start_fret,
            )

        return max(block_candidates, key=lambda candidate: candidate.confidence)

    def required_marker_colors(self) -> set[str]:
        return set(EDGE_PAIR_MARKER_COUNTS.keys())

    def missing_marker_colors(self) -> list[str]:
        return sorted(self.required_marker_colors() - self.detected_marker_colors)

    def is_marker_scan_active(self) -> bool:
        return (
            self.marker_detection_enabled
            and (
                self.marker_roi_selection_pending
                or self.marker_setup_pending
                or self.marker_scan_pending
                or self.marker_scan_confirm_frames > 0
            )
        )

    def begin_marker_roi_selection(self) -> None:
        self.marker_roi = None
        self.marker_roi_polygon = None
        self.marker_roi_selection_pending = True
        self.reset_marker_tracking()
        self.marker_scan_pending = False
        self.marker_scan_completed = False
        self.marker_scan_confirm_frames = 0
        self.detected_marker_colors.clear()
        self.marker_roi_points = []
        self.reset_marker_stability()
        print("カメラ画面上で1-12F全体の四隅を順に4点クリックしてください")

    def begin_marker_auto_setup(self) -> None:
        if not self.marker_roi_user_defined:
            self.marker_roi = None
            self.marker_roi_polygon = None
        self.marker_roi_selection_pending = False
        self.marker_roi_points = []
        self.reset_marker_tracking()
        self.marker_scan_pending = False
        self.marker_scan_completed = False
        self.marker_scan_confirm_frames = 0
        self.detected_marker_colors.clear()
        self.marker_setup_pending = True
        self.reset_marker_stability()
        print("marker setup: auto-detecting 12 fretboard markers from the camera frame.")

    def add_marker_roi_point(self, point: tuple[int, int]) -> None:
        if not self.marker_roi_selection_pending:
            return

        if len(self.marker_roi_points) >= 4:
            return

        self.marker_roi_points.append(point)
        print(f"指板範囲点 {len(self.marker_roi_points)}/4: {point}")

        if len(self.marker_roi_points) == 4:
            self.confirm_marker_roi_selection()

    def confirm_marker_roi_selection(self) -> None:
        if not self.marker_roi_selection_pending:
            return

        if len(self.marker_roi_points) != 4:
            print("まだ4点そろっていません。1-12F全体の四隅を4点クリックしてください")
            return

        self.marker_roi_polygon = tuple(self.marker_roi_points)
        xs = [point[0] for point in self.marker_roi_polygon]
        ys = [point[1] for point in self.marker_roi_polygon]
        self.marker_roi = (
            min(xs),
            min(ys),
            max(xs) - min(xs),
            max(ys) - min(ys),
        )
        self.marker_roi_selection_pending = False
        self.marker_scan_pending = False
        self.marker_scan_completed = False
        self.marker_scan_confirm_frames = 0
        self.detected_marker_colors.clear()
        self.reset_marker_stability()
        self.marker_setup_pending = True
        print(f"指板範囲4点を設定しました: {self.marker_roi_polygon}")
        print("この四角形の内側だけで青・黄・白の四隅マーカーを検出します")

    def advance_marker_scan_confirmation(self) -> None:
        if self.marker_scan_confirm_frames > 0:
            self.marker_scan_confirm_frames -= 1

    def update_marker_scan(self, frame, min_area: float) -> None:
        if not self.marker_scan_pending:
            return

        marker_map = detect_all_markers(
            frame,
            color_names=tuple(sorted(self.required_marker_colors())),
            min_area=min_area,
            roi=self.marker_roi,
            roi_polygon=self.marker_roi_polygon,
        )

        for color_name, markers in marker_map.items():
            if len(markers) >= self.marker_corner_count:
                self.detected_marker_colors.add(color_name)

        if not self.missing_marker_colors():
            self.marker_scan_pending = False
            self.marker_scan_completed = True
            self.marker_scan_confirm_frames = 90
            print("各ブロックを検出しました。同期表示を開始します。")

    @property
    def internal_fret_line_count(self) -> int:
        return self.config.visible_frets - 1

    def load_existing_calibration(self) -> None:
        if not self.calibration_json_path.exists():
            print(f"キャリブレーションJSONが見つかりません: {self.calibration_json_path}")
            print("c キーでカメラ映像上のキャリブレーションを開始してください")
            return

        with self.calibration_json_path.open("r", encoding="utf-8") as file:
            raw_calibration_data = json.load(file)

        saved_flip_mode = raw_calibration_data.get("flip_mode", "none")

        if saved_flip_mode != self.flip_mode:
            print(
                "保存済みキャリブレーションの反転モードが現在の表示と違います: "
                f"saved={saved_flip_mode}, current={self.flip_mode}"
            )
            print("c キーでカメラ映像上のキャリブレーションをやり直してください")
            return

        calibration_data = load_calibration_data(self.calibration_json_path)
        self.calibration_points = calibration_data.points
        self.fret_boundaries = get_compatible_fret_boundaries(
            calibration_data.fret_boundaries,
            self.config,
        )
        self.perspective_matrix = build_perspective_matrix(
            self.calibration_points,
            self.config,
        )
        self.tracking_lost = False
        self.previous_gray_frame = None
        self.tracking_points = None
        print(f"キャリブレーションを読み込みました: {self.calibration_json_path}")

    def start_calibration(self) -> None:
        self.mode = "calibrating"
        self.corner_points = []
        self.internal_fret_points = []
        self.last_click = None
        self.last_result = None
        self.last_match_result = None
        self.tracking_lost = False
        self.previous_gray_frame = None
        self.tracking_points = None
        print("カメラ映像上で指板の4隅をクリックしてください")
        print("順番: 左上 -> 右上 -> 右下 -> 左下")

    def set_tracking_enabled(self, enabled: bool) -> None:
        self.tracking_enabled = enabled
        self.tracking_lost = False
        self.previous_gray_frame = None
        self.tracking_points = None

        if enabled:
            print("指板枠の追跡を有効にしました")
        else:
            print("指板枠の追跡を無効にしました")

    def initialize_tracking_points(self, gray_frame) -> None:
        if self.calibration_points is None:
            self.tracking_points = None
            return

        mask = np.zeros(gray_frame.shape, dtype=np.uint8)
        polygon_points = self.calibration_points.astype(np.int32)
        cv2.fillConvexPoly(mask, polygon_points, 255)

        center = self.calibration_points.mean(axis=0)
        inner_polygon_points = (
            center + (self.calibration_points - center) * 0.72
        ).astype(np.int32)
        cv2.fillConvexPoly(mask, inner_polygon_points, 0)

        self.tracking_points = cv2.goodFeaturesToTrack(
            gray_frame,
            maxCorners=90,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7,
            mask=mask,
        )

    def reject_tracking_update(self) -> None:
        self.tracking_lost = True

    def update_tracking(self, gray_frame) -> None:
        if self.previous_gray_frame is None:
            self.initialize_tracking_points(gray_frame)
            self.previous_gray_frame = gray_frame
            return

        if (
            self.mode != "mapping"
            or not self.tracking_enabled
            or self.calibration_points is None
        ):
            self.previous_gray_frame = gray_frame
            return

        if self.tracking_points is None or len(self.tracking_points) < 12:
            self.initialize_tracking_points(self.previous_gray_frame)

        if self.tracking_points is None or len(self.tracking_points) < 12:
            self.reject_tracking_update()
            return

        old_points = self.tracking_points.astype(np.float32)
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_gray_frame,
            gray_frame,
            old_points,
            None,
            winSize=(31, 31),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                20,
                0.03,
            ),
        )

        if new_points is None or status is None:
            self.reject_tracking_update()
            return

        valid_mask = status.reshape(-1) == 1
        old_valid_points = old_points.reshape(-1, 2)[valid_mask]
        new_valid_points = new_points.reshape(-1, 2)[valid_mask]

        if len(new_valid_points) < 12:
            self.reject_tracking_update()
            return

        homography_matrix, inlier_mask = cv2.findHomography(
            old_valid_points,
            new_valid_points,
            cv2.RANSAC,
            5.0,
        )

        if homography_matrix is None or inlier_mask is None:
            self.reject_tracking_update()
            return

        inlier_mask = inlier_mask.reshape(-1).astype(bool)
        inlier_count = int(inlier_mask.sum())
        inlier_ratio = inlier_count / len(inlier_mask)

        if inlier_count < 18 or inlier_ratio < 0.65:
            self.reject_tracking_update()
            return

        previous_area = abs(cv2.contourArea(self.calibration_points))
        previous_center = self.calibration_points.mean(axis=0)
        transformed_points = cv2.perspectiveTransform(
            self.calibration_points.reshape(-1, 1, 2),
            homography_matrix,
        ).reshape(4, 2)
        transformed_area = abs(cv2.contourArea(transformed_points))
        transformed_center = transformed_points.mean(axis=0)
        center_move = float(np.linalg.norm(transformed_center - previous_center))
        corner_moves = np.linalg.norm(
            transformed_points - self.calibration_points,
            axis=1,
        )
        max_corner_move = float(corner_moves.max())

        if (
            not np.isfinite(transformed_points).all()
            or transformed_area < 1000
            or transformed_area < previous_area * 0.85
            or transformed_area > previous_area * 1.18
            or center_move > max(gray_frame.shape) * 0.045
            or max_corner_move > max(gray_frame.shape) * 0.07
        ):
            self.reject_tracking_update()
            return

        self.calibration_points = transformed_points.astype(np.float32)
        self.perspective_matrix = build_perspective_matrix(
            self.calibration_points,
            self.config,
        )
        self.tracking_points = new_valid_points[inlier_mask].reshape(-1, 1, 2)

        self.tracking_lost = False
        self.previous_gray_frame = gray_frame

    def add_calibration_click(self, x: int, y: int) -> None:
        if len(self.corner_points) < 4:
            self.corner_points.append((x, y))
            print(f"Corner {len(self.corner_points)}: ({x}, {y})")

            if len(self.corner_points) == 4:
                print(
                    "必要なら、内部フレット線を左から順に "
                    f"{self.internal_fret_line_count} 本クリックしてください"
                )
                print("内部フレット線を指定しない場合は、このまま s で保存できます")
            return

        if len(self.internal_fret_points) < self.internal_fret_line_count:
            self.internal_fret_points.append((x, y))
            print(f"Internal fret line {len(self.internal_fret_points)}: ({x}, {y})")
            return

        print("必要な点はすべて選択済みです。s で保存、c でやり直せます")

    def build_virtual_fret_boundaries(self):
        if len(self.internal_fret_points) != self.internal_fret_line_count:
            return None

        image_points = np.array(self.corner_points, dtype=np.float32)
        perspective_matrix = build_perspective_matrix(image_points, self.config)
        internal_points = np.array(
            [[[x, y]] for x, y in self.internal_fret_points],
            dtype=np.float32,
        )
        virtual_points = cv2.perspectiveTransform(internal_points, perspective_matrix)
        internal_boundaries = sorted(float(point[0][0]) for point in virtual_points)

        return [0.0, *internal_boundaries, float(self.config.virtual_width)]

    def save_calibration(self) -> None:
        if self.mode != "calibrating":
            print("c キーでキャリブレーションを開始してから保存してください")
            return

        if len(self.corner_points) != 4:
            print("4隅がそろっていません")
            return

        if 0 < len(self.internal_fret_points) < self.internal_fret_line_count:
            print(
                "内部フレット線を使う場合は、"
                f"{self.internal_fret_line_count} 本すべてクリックしてください"
            )
            return

        data = {
            "order": ["left_top", "right_top", "right_bottom", "left_bottom"],
            "points": self.corner_points,
            "start_fret": self.config.start_fret,
            "visible_frets": self.config.visible_frets,
            "flip_mode": self.flip_mode,
            "virtual_size": {
                "width": self.config.virtual_width,
                "height": self.config.virtual_height,
            },
        }

        virtual_fret_boundaries = self.build_virtual_fret_boundaries()

        if virtual_fret_boundaries is not None:
            data["internal_fret_line_points"] = self.internal_fret_points
            data["virtual_fret_boundaries"] = virtual_fret_boundaries

        self.calibration_json_path.parent.mkdir(parents=True, exist_ok=True)

        with self.calibration_json_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)

        self.calibration_points = np.array(self.corner_points, dtype=np.float32)
        self.fret_boundaries = (
            tuple(virtual_fret_boundaries)
            if virtual_fret_boundaries is not None
            else None
        )
        self.perspective_matrix = build_perspective_matrix(
            self.calibration_points,
            self.config,
        )
        self.mode = "mapping"
        self.tracking_lost = False
        self.previous_gray_frame = None
        self.tracking_points = None
        print(f"保存しました: {self.calibration_json_path}")

    def handle_mapping_click(self, x: int, y: int) -> None:
        if self.perspective_matrix is None:
            print("キャリブレーションが未設定です。c キーで設定してください")
            return

        virtual_x, virtual_y = map_image_point_to_virtual_point(
            image_x=x,
            image_y=y,
            perspective_matrix=self.perspective_matrix,
        )
        result = estimate_string_and_fret(
            virtual_x=virtual_x,
            virtual_y=virtual_y,
            config=self.config,
            fret_boundaries=self.fret_boundaries,
        )

        self.last_click = (x, y)
        self.last_result = result
        self.refresh_pitch_match_result()

        if self.last_match_result is not None:
            print(format_pitch_match_result(self.last_match_result))

        if result["inside_fretboard"]:
            print(
                f'推定結果: {result["string_number"]}弦 {result["fret_number"]}フレット '
                f'(virtual_x={result["virtual_x"]:.1f}, virtual_y={result["virtual_y"]:.1f})'
            )
        else:
            print(
                f'指板範囲外: virtual_x={result["virtual_x"]:.1f}, '
                f'virtual_y={result["virtual_y"]:.1f}'
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use a live camera frame to estimate guitar string and fret positions.",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--start-fret", type=int, default=5)
    parser.add_argument("--visible-frets", type=int, default=4)
    parser.add_argument(
        "--midi-pitch",
        type=int,
        default=None,
        help="Optional MIDI pitch to match against the clicked camera estimate.",
    )
    parser.add_argument(
        "--audio-pitch",
        action="store_true",
        help="Use microphone input to update the MIDI pitch estimate.",
    )
    parser.add_argument(
        "--audio-device",
        default=None,
        help="Optional sounddevice input device index or name.",
    )
    parser.add_argument("--audio-sample-rate", type=int, default=44100)
    parser.add_argument("--audio-block-size", type=int, default=4096)
    parser.add_argument("--audio-rms-threshold", type=float, default=0.015)
    parser.add_argument(
        "--display-mode",
        choices=DISPLAY_MODES,
        default="fretboard",
        help="Choose whether mapping mode displays the virtual fretboard or camera feed.",
    )
    parser.add_argument(
        "--display-total-frets",
        type=int,
        default=fretboard_display.DEFAULT_TOTAL_FRETS,
        help="Number of frets to draw in the virtual fretboard display.",
    )
    parser.add_argument(
        "--display-block-size",
        type=int,
        default=fretboard_display.DEFAULT_BLOCK_SIZE,
        help="Number of frets per visual block in the virtual fretboard display.",
    )
    parser.add_argument(
        "--marker-detect",
        action="store_true",
        help="Detect colored fretboard block stickers from the camera frame.",
    )
    parser.add_argument(
        "--hand-marker-color",
        choices=tuple(sorted(COLOR_RANGES.keys())),
        default=None,
        help="Optional hand sticker color. The nearest detected block marker is selected.",
    )
    parser.add_argument(
        "--hand-marker-count",
        type=int,
        default=2,
        help="Number of hand stickers required when --hand-marker-color is used.",
    )
    parser.add_argument(
        "--hand-marker-min-area",
        type=float,
        default=30.0,
        help="Minimum contour area for hand stickers.",
    )
    parser.add_argument(
        "--finger-detect",
        action="store_true",
        help="Detect colored stickers on four fretting fingers and estimate fret numbers.",
    )
    parser.add_argument(
        "--finger-markers",
        type=parse_finger_marker_specs,
        default=parse_finger_marker_specs("index:pink,middle:green,ring:red,little:purple"),
        help=(
            "Comma-separated finger:color mapping, for example "
            "index:pink,middle:green,ring:red,little:purple."
        ),
    )
    parser.add_argument(
        "--finger-marker-min-area",
        type=float,
        default=24.0,
        help="Minimum contour area for finger stickers.",
    )
    parser.add_argument(
        "--pick-detect",
        action="store_true",
        help="Detect a colored pick marker, infer the played string, and combine it with audio pitch.",
    )
    parser.add_argument(
        "--pick-marker-color",
        choices=tuple(sorted(COLOR_RANGES.keys())),
        default="pink",
        help="Sticker color on the pick.",
    )
    parser.add_argument(
        "--pick-marker-min-area",
        type=float,
        default=20.0,
        help="Minimum contour area for the pick sticker.",
    )
    parser.add_argument(
        "--marker-min-area",
        type=float,
        default=80.0,
        help="Minimum contour area for colored sticker detection.",
    )
    parser.add_argument(
        "--marker-roi",
        default=None,
        help="Optional detection region as x,y,width,height.",
    )
    parser.add_argument(
        "--marker-corner-count",
        type=int,
        default=4,
        help="Number of same-color corner stickers required to accept a block.",
    )
    parser.add_argument(
        "--marker-stable-frames",
        type=int,
        default=8,
        help="Consecutive frames required before accepting a marker block change.",
    )
    parser.add_argument(
        "--marker-min-confidence",
        type=float,
        default=0.25,
        help="Minimum marker confidence before it can become stable.",
    )
    parser.add_argument(
        "--marker-hold-frames",
        type=int,
        default=30,
        help="Frames to keep the last stable marker after temporary loss.",
    )
    parser.add_argument(
        "--marker-track-points",
        action="store_true",
        help=(
            "Move the corrected 12 marker points with optical flow. "
            "Off by default because hands can drag the points when occluding stickers."
        ),
    )
    parser.add_argument(
        "--no-marker-homography-track",
        action="store_true",
        help="Disable homography tracking from all visible edge-pair markers after lock.",
    )
    parser.add_argument(
        "--marker-homography-smoothing",
        type=float,
        default=0.45,
        help="Low-pass factor for homography reprojection. Lower is smoother.",
    )
    parser.add_argument(
        "--marker-homography-min-inliers",
        type=int,
        default=6,
        help="Minimum RANSAC inliers required to accept homography tracking.",
    )
    parser.add_argument(
        "--marker-homography-max-median-jump",
        type=float,
        default=70.0,
        help="Reject homography updates whose median marker jump is larger than this many pixels.",
    )
    parser.add_argument(
        "--marker-homography-roi-margin",
        type=int,
        default=60,
        help="Pixels to expand the fretboard ROI while searching edge-pair markers after lock.",
    )
    parser.add_argument(
        "--no-occlusion-hand",
        action="store_true",
        help="Disable hand-position estimation from temporarily hidden edge-pair markers.",
    )
    parser.add_argument(
        "--occlusion-hand-min-hidden",
        type=int,
        default=2,
        help="Minimum hidden marker count required to estimate hand position.",
    )
    parser.add_argument(
        "--occlusion-hand-fret-radius",
        type=int,
        default=1,
        help="Number of frets around the estimated hand fret to highlight.",
    )
    parser.add_argument(
        "--no-audio-hand-match",
        action="store_true",
        help="Disable matching microphone pitch candidates against the estimated hand fret range.",
    )
    parser.add_argument(
        "--no-background-hand",
        action="store_true",
        help="Disable stickerless hand-position estimation by fretboard background subtraction.",
    )
    parser.add_argument(
        "--background-hand-threshold",
        type=int,
        default=10,
        help="Pixel difference threshold for background-subtraction hand detection.",
    )
    parser.add_argument(
        "--background-hand-min-area",
        type=float,
        default=80.0,
        help="Minimum changed area in the warped fretboard image for background hand detection.",
    )
    parser.add_argument(
        "--background-hand-fret-radius",
        type=int,
        default=1,
        help="Extra frets to include around the background-subtraction hand range.",
    )
    parser.add_argument(
        "--nut-anchor-detect",
        action="store_true",
        help=(
            "Track 2 or more always-visible nut/1F-left markers and reproject "
            "the locked 12 fretboard marker points from them."
        ),
    )
    parser.add_argument(
        "--nut-anchor-colors",
        type=parse_marker_color_list,
        default=None,
        help=(
            "Optional comma-separated unique nut anchor marker colors, for example "
            "pink,green. If omitted, two same-color nut anchors are used."
        ),
    )
    parser.add_argument(
        "--nut-anchor-block-color",
        choices=tuple(sorted(COLOR_RANGES.keys())),
        default="green",
        help="Same-color marker used for the two nut/1F-left anchor points.",
    )
    parser.add_argument(
        "--nut-anchor-min-area",
        type=float,
        default=30.0,
        help="Minimum contour area for nut/1F-left anchor stickers.",
    )
    parser.add_argument(
        "--nut-anchor-roi-margin",
        type=int,
        default=80,
        help="Pixels to expand the fretboard ROI while searching nut anchors.",
    )
    parser.add_argument(
        "--nut-anchor-smoothing",
        type=float,
        default=0.28,
        help="Low-pass factor for nut-anchor reprojection. Lower is smoother.",
    )
    parser.add_argument(
        "--nut-anchor-max-step",
        type=float,
        default=45.0,
        help="Maximum per-frame marker movement before extra damping is applied.",
    )
    parser.add_argument(
        "--sync-start-on",
        action="store_true",
        help="Start guitar sync recording immediately.",
    )
    parser.add_argument(
        "--sync-output-dir",
        type=Path,
        default=DEFAULT_SYNC_OUTPUT_DIR,
        help="Directory for sync session audio and display event logs.",
    )
    parser.add_argument(
        "--sync-no-audio",
        action="store_true",
        help="Record display events only, without saving audio.wav.",
    )
    parser.add_argument(
        "--sync-no-video",
        action="store_true",
        help="Do not save camera video.mp4 while sync recording is ON.",
    )
    parser.add_argument(
        "--sync-video-fps",
        type=float,
        default=30.0,
        help="FPS written to the sync session video file.",
    )
    parser.add_argument(
        "--sync-video-codec",
        default="mp4v",
        help="FourCC codec for sync session video, for example mp4v or MJPG.",
    )
    parser.add_argument(
        "--gpio-button-pin",
        type=int,
        default=None,
        help="Raspberry Pi GPIO pin for the sync ON/OFF tact switch.",
    )
    parser.add_argument(
        "--gpio-led-pin",
        type=int,
        default=None,
        help="Raspberry Pi GPIO pin for the sync status LED.",
    )
    parser.add_argument(
        "--no-gpio",
        action="store_true",
        help="Disable Raspberry Pi GPIO sync controls.",
    )
    parser.add_argument(
        "--flip",
        choices=FLIP_MODES,
        default="none",
        help="Flip camera preview before calibration and mapping.",
    )
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="Disable live tracking of the calibrated fretboard frame.",
    )
    parser.add_argument(
        "--calibration-json",
        type=Path,
        default=DEFAULT_CALIBRATION_JSON_PATH,
    )
    return parser.parse_args()


def draw_calibration_points(target_image, state: CameraMapperState) -> None:
    for index, (x, y) in enumerate(state.corner_points):
        cv2.circle(target_image, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(
            target_image,
            str(index + 1),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    if len(state.corner_points) == 4:
        closed_points = state.corner_points + [state.corner_points[0]]

        for index in range(4):
            cv2.line(
                target_image,
                closed_points[index],
                closed_points[index + 1],
                (0, 255, 0),
                2,
            )

    for index, (x, y) in enumerate(state.internal_fret_points):
        cv2.circle(target_image, (x, y), 6, (0, 255, 255), -1)
        cv2.putText(
            target_image,
            f"F{index + 1}",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )


def format_pitch_match_overlay(match_result: dict) -> str:
    selected_position = match_result["selected_position"]
    pitch_label = f'{match_result["note_name"]} MIDI {match_result["midi_pitch"]}'

    if selected_position is None:
        return f"{pitch_label}: no candidate"

    return (
        f"{pitch_label}: "
        f'{selected_position["string_number"]} string / '
        f'{selected_position["fret_number"]} fret '
        f'[{match_result["status"]}]'
    )


def format_manual_pitch_overlay(state: CameraMapperState) -> str:
    if state.pick_detection_enabled:
        print(
            "pick detection: ON "
            f"color={state.pick_marker_color} "
            f"min_area={state.pick_marker_min_area}"
        )
        print("pick ROI: select 4 corners around the picking-string area")
        if not args.audio_pitch:
            print("pick+audio: add --audio-pitch to use the USB microphone")
    if state.midi_pitch is None:
        return f"{state.pitch_source} pitch: none  ]:set D4"

    note_name = midi_pitch_to_note_name(state.midi_pitch)
    pitch_text = f"{state.pitch_source} pitch: {note_name} MIDI {state.midi_pitch}"

    if state.pitch_source == "audio" and state.last_audio_detection is not None:
        return (
            f"{pitch_text} "
            f"{state.last_audio_detection.frequency_hz:.1f}Hz "
            f"conf:{state.last_audio_detection.confidence:.2f}"
        )

    return f"{pitch_text}  [:down ]:up"


def format_sync_overlay(state: CameraMapperState) -> str:
    return "sync: ON" if state.sync_enabled else "sync: OFF"


def format_marker_overlay(state: CameraMapperState) -> str:
    if not state.marker_detection_enabled:
        return "marker: OFF"

    if state.marker_roi_selection_pending:
        return f"select fretboard ROI: click 4 corners ({len(state.marker_roi_points)}/4)"

    if state.marker_setup_pending:
        anchor_text = ""

        if state.nut_anchor_enabled:
            anchor_text = (
                f" anchors:{len(state.raw_nut_anchor_markers)}/"
                f"{state.nut_anchor_expected_count()}"
            )

        return (
            f"marker setup: {len(state.marker_points)}/12 points"
            f"{anchor_text}  drag / Enter:lock"
        )

    if state.marker_scan_pending:
        found = ",".join(sorted(state.detected_marker_colors)) or "-"
        missing = ",".join(state.missing_marker_colors()) or "-"
        return f"marker scan: found {found} / need {missing}"

    if state.marker_scan_confirm_frames > 0:
        return "marker scan: all edge-pair colors detected"

    counts = state.raw_block_marker_counts
    count_text = (
        f" g:{counts.get('green', 0)}"
        f" b:{counts.get('blue', 0)}"
        f" y:{counts.get('yellow', 0)}"
        f" p:{counts.get('pink', 0)}"
    )
    detection = state.last_marker_detection

    if detection is None:
        if state.raw_marker_detection is not None:
            return (
                f"marker raw: {state.raw_marker_detection.block_name} "
                f"{state.marker_candidate_frames}/{state.marker_stable_frames} "
                f"conf:{state.raw_marker_detection.confidence:.2f} "
                f"hand:{len(state.raw_hand_markers)}/{state.hand_marker_count}"
                f"{count_text}"
            )
        if state.hand_marker_color is not None:
            return (
                f"marker: searching  hand:{len(state.raw_hand_markers)}/{state.hand_marker_count}"
                f"{count_text}"
            )
        return f"marker: searching{count_text}"

    source = (
        f" hand:{detection.hand_markers[0].color_name}x{len(detection.hand_markers)}"
        if detection.hand_markers
        else f" hand:{detection.hand_marker.color_name}"
        if detection.hand_marker is not None
        else ""
    )
    point_mode = "setup"

    if state.marker_tracking_enabled:
        point_mode = "optical"
    elif state.marker_homography_active:
        point_mode = f"homography:{state.marker_homography_inliers}"
    elif state.nut_anchor_transform_active:
        point_mode = "nut-anchor"
    elif state.marker_points_locked:
        point_mode = "locked"

    anchor_text = ""

    if state.nut_anchor_enabled:
        anchor_delta = ""

        if state.nut_anchor_transform_active:
            anchor_delta = (
                f" d:{state.nut_anchor_transform_delta[0]:.0f},"
                f"{state.nut_anchor_transform_delta[1]:.0f}"
            )

        anchor_text = (
            f" anchors:{len(state.raw_nut_anchor_markers)}/"
            f"{state.nut_anchor_expected_count()}"
            f"{anchor_delta}"
        )

    return (
        f"marker: {detection.block_name} "
        f"{detection.marker_color} conf:{detection.confidence:.2f}{source}"
        f"{count_text}"
        f" points:{point_mode}{anchor_text}"
    )


def format_finger_overlay(state: CameraMapperState) -> str:
    if not state.finger_detection_enabled:
        return ""

    labels = []

    for finger_name in state.finger_marker_specs:
        estimate = state.finger_fret_estimates.get(finger_name)

        if estimate is None:
            labels.append(f"{finger_name}:-")
        else:
            labels.append(f"{finger_name}:{estimate['fret_number']}F")

    return "fingers: " + " ".join(labels)


def format_pick_overlay(state: CameraMapperState) -> str:
    if not state.pick_detection_enabled:
        return ""

    if state.pick_roi_selection_pending:
        return f"pick ROI: click 4 corners ({len(state.pick_roi_points)}/4)"

    if state.raw_pick_marker is None:
        return f"pick: searching {state.pick_marker_color}"

    string_text = (
        f"{state.pick_string_number} string"
        if state.pick_string_number is not None
        else "string:-"
    )

    if state.pick_audio_position is None:
        return f"pick: {string_text} conf:{state.pick_string_confidence:.2f} pitch:-"

    return (
        f"pick+audio: {state.pick_audio_position['string_number']} string / "
        f"{state.pick_audio_position['fret_number']} fret "
        f"conf:{state.pick_string_confidence:.2f}"
    )


def format_occlusion_hand_overlay(state: CameraMapperState) -> str:
    if not state.occlusion_hand_enabled:
        return ""

    if state.occlusion_hand_estimate is None:
        return f"hand estimate: - hidden:{len(state.hidden_marker_points)}"

    estimate = state.occlusion_hand_estimate
    return (
        f"hand estimate: around {estimate['fret_number']}F "
        f"hidden:{estimate['hidden_count']} "
        f"conf:{estimate['confidence']:.2f}"
    )


def format_background_hand_overlay(state: CameraMapperState) -> str:
    if not state.background_hand_enabled:
        return ""

    if state.background_hand_reference is None:
        return "bg hand: press b with no hand on fretboard"

    if state.background_hand_estimate is None:
        return (
            "bg hand: - "
            f"area:{state.background_hand_last_area:.0f} "
            f"largest:{state.background_hand_largest_area:.0f} "
            f"px:{state.background_hand_changed_pixels}"
        )

    estimate = state.background_hand_estimate
    return (
        f"bg hand: {estimate['detected_start_fret']}-{estimate['detected_end_fret']}F "
        f"center:{estimate['fret_number']}F "
        f"conf:{estimate['confidence']:.2f} "
        f"area:{estimate['area']:.0f} "
        f"px:{state.background_hand_changed_pixels}"
    )


def format_audio_hand_overlay(state: CameraMapperState) -> str:
    if not state.audio_hand_match_enabled:
        return ""

    if state.midi_pitch is None:
        return "audio+hand: pitch:-"

    if state.audio_hand_match_result is None:
        if state.occlusion_hand_estimate is None:
            return "audio+hand: hand:-"

        return "audio+hand: no match"

    result = state.audio_hand_match_result
    selected_position = result["selected_position"]
    range_count = len(result["in_range_candidates"])
    total_count = len(result["candidate_positions"])

    if selected_position is None:
        if result["hand_start_fret"] is None:
            return (
                f"audio+hand: {result['note_name']} "
                f"hand:- candidates:{range_count}/{total_count}"
            )

        return (
            f"audio+hand: {result['note_name']} "
            f"range:{result['hand_start_fret']}-{result['hand_end_fret']}F "
            f"candidates:{range_count}/{total_count}"
        )

    if result["hand_start_fret"] is None:
        return (
            f"audio+hand: {result['note_name']} -> "
            f"{selected_position['string_number']}s/{selected_position['fret_number']}F "
            f"hand:- candidates:{range_count}/{total_count} [{result['status']}]"
        )

    return (
        f"audio+hand: {result['note_name']} -> "
        f"{selected_position['string_number']}s/{selected_position['fret_number']}F "
        f"candidates:{range_count}/{total_count} [{result['status']}]"
    )


def get_highlight_position(state: CameraMapperState):
    if state.audio_hand_match_result is not None:
        selected_position = state.audio_hand_match_result["selected_position"]

        if selected_position is not None:
            return {
                "inside_fretboard": True,
                "string_number": selected_position["string_number"],
                "fret_number": selected_position["fret_number"],
                "source": "audio_hand",
            }

    if state.last_match_result is not None:
        selected_position = state.last_match_result["selected_position"]

        if selected_position is not None:
            return selected_position

    if state.last_result is not None and state.last_result["inside_fretboard"]:
        return state.last_result

    return None


def draw_status_overlay(frame, state: CameraMapperState, start_y: int = 32) -> None:
    status = "CALIBRATION" if state.mode == "calibrating" else "MAPPING"
    tracking_status = "off"

    if state.tracking_enabled:
        tracking_status = "lost" if state.tracking_lost else "on"

    controls = "b/f/v/o/q" if state.marker_only else "c/s/b/f/t/v/o/q"

    cv2.putText(
        frame,
        (
            f"{status}  display:{state.display_mode}  flip:{state.flip_mode}  "
            f"track:{tracking_status}  {controls}"
        ),
        (16, start_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        format_manual_pitch_overlay(state),
        (16, start_y + 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        format_sync_overlay(state),
        (16, start_y + 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 255) if state.sync_enabled else (185, 185, 185),
        2,
    )

    cv2.putText(
        frame,
        format_marker_overlay(state),
        (16, start_y + 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 255) if state.marker_detection_enabled else (185, 185, 185),
        2,
    )

    next_line_y = start_y + 144

    if state.pick_detection_enabled:
        cv2.putText(
            frame,
            format_pick_overlay(state),
            (16, next_line_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (120, 220, 255),
            2,
        )
        next_line_y += 36

    if state.occlusion_hand_enabled and state.marker_points_locked:
        cv2.putText(
            frame,
            format_occlusion_hand_overlay(state),
            (16, next_line_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 210, 120),
            2,
        )
        next_line_y += 36

    if state.background_hand_enabled and state.marker_points_locked:
        cv2.putText(
            frame,
            format_background_hand_overlay(state),
            (16, next_line_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (120, 255, 120),
            2,
        )
        next_line_y += 36

    if state.audio_hand_match_enabled and state.marker_points_locked:
        cv2.putText(
            frame,
            format_audio_hand_overlay(state),
            (16, next_line_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (140, 255, 210),
            2,
        )
        next_line_y += 36

    if state.finger_detection_enabled:
        cv2.putText(
            frame,
            format_finger_overlay(state),
            (16, next_line_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (120, 255, 180),
            2,
        )
        next_line_y += 36

    if state.last_match_result is not None:
        cv2.putText(
            frame,
            format_pitch_match_overlay(state.last_match_result),
            (16, next_line_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 255, 255),
            2,
        )


def draw_overlay(frame, state: CameraMapperState) -> None:
    if state.mode == "calibrating" and not state.marker_only:
        draw_calibration_points(frame, state)
    elif (
        not state.marker_only
        and state.calibration_points is not None
        and state.perspective_matrix is not None
    ):
        draw_calibration_frame(frame, state.calibration_points)
        draw_fret_boundaries(
            frame,
            state.perspective_matrix,
            state.fret_boundaries,
            state.config,
        )

    if state.last_click is not None and state.last_result is not None:
        draw_estimation_result(
            frame,
            state.last_click[0],
            state.last_click[1],
            state.last_result,
        )

    if state.marker_detection_enabled:
        if state.marker_roi_polygon is not None and not state.marker_points_locked:
            draw_marker_polygon(frame, state.marker_roi_polygon, label="FRETBOARD ROI")
        elif state.marker_roi is not None and not state.marker_points_locked:
            draw_marker_roi(frame, state.marker_roi)

        if state.marker_roi_points:
            draw_marker_polygon(frame, state.marker_roi_points, label="SELECT ROI")

        draw_marker_setup_points(frame, state)
        draw_occlusion_hand_estimate(frame, state)
        draw_background_hand_estimate(frame, state)
        draw_nut_anchor_markers(frame, state)
        draw_raw_hand_markers(frame, state)
        draw_finger_markers(frame, state)
        draw_marker_detection(frame, state.last_marker_detection)

        if state.marker_roi_selection_pending:
            cv2.putText(
                frame,
                "Click the 4 corners of the whole 1-12F fretboard area",
                (16, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        elif state.marker_setup_pending:
            cv2.putText(
                frame,
                "Drag wrong marker points. Enter/Space: start tracking / r: redetect",
                (16, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        elif state.marker_scan_pending:
            cv2.putText(
                frame,
                "Show green, blue, yellow, and pink edge-pair stickers",
                (16, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        elif state.marker_scan_confirm_frames > 0:
            cv2.putText(
                frame,
                "All edge-pair colors detected. Starting sync display...",
                (16, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    draw_pick_overlay(frame, state)
    draw_status_overlay(frame, state)


def draw_marker_setup_points(frame, state: CameraMapperState) -> None:
    if not state.marker_points:
        return

    for marker_point in state.marker_points:
        color_name = marker_point["color"]
        draw_color = COLOR_RANGES[color_name].draw_color_bgr
        center = marker_point["center"]
        label = f"{color_name[0].upper()}{marker_point['index']}"
        label_dx = 12 if marker_point["index"] % 2 == 1 else -56
        label_dy = -12 if marker_point["index"] <= 2 else 22
        cv2.circle(frame, center, 10, draw_color, 2, cv2.LINE_AA)
        cv2.circle(frame, center, 3, draw_color, -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            label,
            (center[0] + label_dx, center[1] + label_dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            draw_color,
            2,
            cv2.LINE_AA,
        )

    for color_name in EDGE_PAIR_MARKER_COUNTS:
        color_points = [
            marker_point
            for marker_point in state.marker_points
            if marker_point["color"] == color_name
        ]
        sorted_points = sorted(
            color_points,
            key=lambda marker_point: (
                marker_point["center"][1],
                marker_point["center"][0],
            ),
        )
        draw_color = COLOR_RANGES[color_name].draw_color_bgr

        for pair_index in range(0, len(sorted_points) - 1, 2):
            pair_points = sorted(
                sorted_points[pair_index : pair_index + 2],
                key=lambda marker_point: marker_point["center"][0],
            )
            cv2.line(
                frame,
                pair_points[0]["center"],
                pair_points[1]["center"],
                draw_color,
                2,
                cv2.LINE_AA,
            )


def draw_occlusion_hand_estimate(frame, state: CameraMapperState) -> None:
    if not state.occlusion_hand_enabled:
        return

    draw_occlusion_hand_fret_band(frame, state)

    for marker_point in state.hidden_marker_points:
        cv2.circle(
            frame,
            marker_point["center"],
            7,
            (80, 120, 255),
            2,
            cv2.LINE_AA,
        )

    if state.occlusion_hand_estimate is None:
        return

    center = state.occlusion_hand_estimate["center"]
    cv2.circle(frame, center, 18, (255, 210, 120), 2, cv2.LINE_AA)
    cv2.circle(frame, center, 4, (255, 210, 120), -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"hand~{state.occlusion_hand_estimate['fret_number']}F",
        (center[0] + 18, center[1] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (255, 210, 120),
        2,
        cv2.LINE_AA,
    )


def draw_occlusion_hand_fret_band(frame, state: CameraMapperState) -> None:
    if state.occlusion_hand_estimate is None or len(state.marker_points) < 4:
        return

    points = np.array(
        [marker_point["center"] for marker_point in state.marker_points],
        dtype=np.float32,
    )
    center = points.mean(axis=0)
    centered_points = points - center

    if len(centered_points) < 2:
        return

    _, _, vh = np.linalg.svd(centered_points, full_matrices=False)
    fret_axis = vh[0]
    string_axis = np.array([-fret_axis[1], fret_axis[0]], dtype=np.float32)
    fret_projections = points @ fret_axis
    string_projections = points @ string_axis
    fret_min = float(fret_projections.min())
    fret_max = float(fret_projections.max())
    string_min = float(string_projections.min())
    string_max = float(string_projections.max())
    fret_span = fret_max - fret_min

    if fret_span <= 1.0:
        return

    fret_number = state.occlusion_hand_estimate["fret_number"]
    start_fret = max(1, fret_number - state.occlusion_hand_fret_radius)
    end_fret = min(12, fret_number + state.occlusion_hand_fret_radius)
    start_ratio = (start_fret - 1) / 12.0
    end_ratio = end_fret / 12.0
    band_start = fret_min + fret_span * start_ratio
    band_end = fret_min + fret_span * end_ratio
    string_padding = max(16.0, (string_max - string_min) * 0.08)
    string_start = string_min - string_padding
    string_end = string_max + string_padding

    polygon_points = np.array(
        [
            fret_axis * band_start + string_axis * string_start,
            fret_axis * band_end + string_axis * string_start,
            fret_axis * band_end + string_axis * string_end,
            fret_axis * band_start + string_axis * string_end,
        ],
        dtype=np.int32,
    )
    overlay = frame.copy()
    cv2.fillConvexPoly(overlay, polygon_points, (255, 210, 80), cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)
    cv2.polylines(frame, [polygon_points], True, (255, 230, 120), 2, cv2.LINE_AA)


def draw_occlusion_hand_display_band(frame, state: CameraMapperState) -> None:
    if not state.occlusion_hand_enabled or state.occlusion_hand_estimate is None:
        return

    left, top, right, bottom = fretboard_display.get_fretboard_display_rect(frame)
    total_frets = state.display_total_frets
    fret_width = (right - left) / total_frets
    fret_number = state.occlusion_hand_estimate["fret_number"]
    start_fret = max(1, fret_number - state.occlusion_hand_fret_radius)
    end_fret = min(total_frets, fret_number + state.occlusion_hand_fret_radius)
    band_left = left + int((start_fret - 1) * fret_width)
    band_right = left + int(end_fret * fret_width)
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (band_left, top - 10),
        (band_right, bottom + 10),
        (0, 190, 255),
        -1,
        cv2.LINE_AA,
    )
    cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)
    cv2.rectangle(
        frame,
        (band_left, top - 16),
        (band_right, bottom + 16),
        (0, 230, 255),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        (
            f"HAND {start_fret}-{end_fret}F "
            f"hidden:{state.occlusion_hand_estimate['hidden_count']}"
        ),
        (band_left + 12, top - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 240, 255),
        2,
        cv2.LINE_AA,
    )


def draw_background_hand_estimate(frame, state: CameraMapperState) -> None:
    estimate = state.background_hand_estimate

    if (
        not state.background_hand_enabled
        or estimate is None
        or state.background_hand_homography is None
    ):
        return

    try:
        inverse_homography = np.linalg.inv(state.background_hand_homography)
    except np.linalg.LinAlgError:
        return
    left, top, right, bottom = estimate["canonical_rect"]
    polygon = np.array(
        [
            [[left, top]],
            [[right, top]],
            [[right, bottom]],
            [[left, bottom]],
        ],
        dtype=np.float32,
    )
    image_polygon = cv2.perspectiveTransform(polygon, inverse_homography).astype(np.int32)
    overlay = frame.copy()
    cv2.fillConvexPoly(overlay, image_polygon, (80, 255, 80), cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
    cv2.polylines(frame, [image_polygon], True, (80, 255, 80), 2, cv2.LINE_AA)
    center_polygon = np.array(
        [[[estimate["canonical_center"][0], estimate["canonical_center"][1]]]],
        dtype=np.float32,
    )
    center = cv2.perspectiveTransform(center_polygon, inverse_homography)[0][0]
    center_point = (int(round(float(center[0]))), int(round(float(center[1]))))
    cv2.circle(frame, center_point, 16, (120, 255, 120), 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"bg~{estimate['fret_number']}F",
        (center_point[0] + 18, center_point[1] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (120, 255, 120),
        2,
        cv2.LINE_AA,
    )


def draw_background_hand_display_band(frame, state: CameraMapperState) -> None:
    if not state.background_hand_enabled or state.background_hand_estimate is None:
        return

    left, top, right, bottom = fretboard_display.get_fretboard_display_rect(frame)
    total_frets = state.display_total_frets
    fret_width = (right - left) / total_frets
    estimate = state.background_hand_estimate
    start_fret = max(1, min(total_frets, estimate["start_fret"]))
    end_fret = max(1, min(total_frets, estimate["end_fret"]))
    band_left = left + int((start_fret - 1) * fret_width)
    band_right = left + int(end_fret * fret_width)
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (band_left, top - 4),
        (band_right, bottom + 4),
        (40, 220, 90),
        -1,
        cv2.LINE_AA,
    )
    cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
    cv2.rectangle(
        frame,
        (band_left, top - 10),
        (band_right, bottom + 10),
        (80, 255, 120),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"BG HAND {estimate['detected_start_fret']}-{estimate['detected_end_fret']}F",
        (band_left + 12, bottom + 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (120, 255, 120),
        2,
        cv2.LINE_AA,
    )


def draw_nut_anchor_markers(frame, state: CameraMapperState) -> None:
    if not state.nut_anchor_enabled:
        return

    for anchor_name, marker in state.raw_nut_anchor_markers.items():
        draw_color = COLOR_RANGES[marker.color_name].draw_color_bgr
        x, y, width, height = marker.bounding_box
        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            draw_color,
            2,
            cv2.LINE_AA,
        )
        cv2.circle(frame, marker.center, 7, draw_color, -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"A:{anchor_name}",
            (marker.center[0] + 8, marker.center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            draw_color,
            2,
            cv2.LINE_AA,
        )

    if len(state.raw_nut_anchor_markers) >= 2:
        points = [
            state.raw_nut_anchor_markers[anchor_name].center
            for anchor_name in (
                state.nut_anchor_colors
                if state.nut_anchor_colors
                else state.reference_nut_anchor_points.keys()
            )
            if anchor_name in state.raw_nut_anchor_markers
        ]

        for start_point, end_point in zip(points, points[1:]):
            cv2.line(
                frame,
                start_point,
                end_point,
                (120, 255, 180) if state.nut_anchor_transform_active else (180, 180, 180),
                2,
                cv2.LINE_AA,
            )


def draw_raw_hand_markers(frame, state: CameraMapperState) -> None:
    if not state.raw_hand_markers:
        return

    hand_color = (0, 0, 255)

    if state.hand_marker_color in COLOR_RANGES:
        hand_color = COLOR_RANGES[state.hand_marker_color].draw_color_bgr

    for index, marker in enumerate(state.raw_hand_markers):
        x, y, width, height = marker.bounding_box
        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            hand_color,
            2,
            cv2.LINE_AA,
        )
        cv2.circle(frame, marker.center, 6, hand_color, -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"H{index + 1}",
            (marker.center[0] + 8, marker.center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            hand_color,
            2,
            cv2.LINE_AA,
        )

    if len(state.raw_hand_markers) >= 2:
        cv2.line(
            frame,
            state.raw_hand_markers[0].center,
            state.raw_hand_markers[1].center,
            hand_color,
            2,
            cv2.LINE_AA,
        )


def draw_finger_markers(frame, state: CameraMapperState) -> None:
    if not state.finger_detection_enabled:
        return

    for finger_name, marker in state.raw_finger_markers.items():
        color_name = state.finger_marker_specs.get(finger_name)
        draw_color = (120, 255, 180)

        if color_name in COLOR_RANGES:
            draw_color = COLOR_RANGES[color_name].draw_color_bgr

        x, y, width, height = marker.bounding_box
        estimate = state.finger_fret_estimates.get(finger_name)
        label = finger_name

        if estimate is not None:
            label = f"{finger_name}:{estimate['fret_number']}F"

        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            draw_color,
            2,
            cv2.LINE_AA,
        )
        cv2.circle(frame, marker.center, 5, draw_color, -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            label,
            (marker.center[0] + 8, marker.center[1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            draw_color,
            2,
            cv2.LINE_AA,
        )


def draw_pick_overlay(frame, state: CameraMapperState) -> None:
    if not state.pick_detection_enabled:
        return

    if state.pick_roi_polygon is not None:
        draw_marker_polygon(frame, state.pick_roi_polygon, label="PICK ROI")
        points = order_quad_points(state.pick_roi_polygon).astype(int)
        left_top, right_top, right_bottom, left_bottom = points

        for index, string_number in enumerate(state.config.string_order_top_to_bottom):
            ratio = (index + 0.5) / 6.0
            left_point = (
                int(left_top[0] + (left_bottom[0] - left_top[0]) * ratio),
                int(left_top[1] + (left_bottom[1] - left_top[1]) * ratio),
            )
            right_point = (
                int(right_top[0] + (right_bottom[0] - right_top[0]) * ratio),
                int(right_top[1] + (right_bottom[1] - right_top[1]) * ratio),
            )
            color = (
                (120, 220, 255)
                if string_number == state.pick_string_number
                else (125, 145, 150)
            )
            thickness = 3 if string_number == state.pick_string_number else 1
            cv2.line(frame, left_point, right_point, color, thickness, cv2.LINE_AA)
            cv2.putText(
                frame,
                str(string_number),
                (left_point[0] + 6, left_point[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    if state.pick_roi_points:
        draw_marker_polygon(frame, state.pick_roi_points, label="SELECT PICK ROI")

    if state.raw_pick_marker is None:
        return

    draw_color = COLOR_RANGES[state.pick_marker_color].draw_color_bgr
    x, y, width, height = state.raw_pick_marker.bounding_box
    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        draw_color,
        2,
        cv2.LINE_AA,
    )
    cv2.circle(frame, state.raw_pick_marker.center, 6, draw_color, -1, cv2.LINE_AA)
    label = "pick"

    if state.pick_audio_position is not None:
        label = (
            f"P:{state.pick_audio_position['string_number']}s/"
            f"{state.pick_audio_position['fret_number']}F"
        )
    elif state.pick_string_number is not None:
        label = f"P:{state.pick_string_number}s"

    cv2.putText(
        frame,
        label,
        (state.raw_pick_marker.center[0] + 8, state.raw_pick_marker.center[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        draw_color,
        2,
        cv2.LINE_AA,
    )


def draw_camera_preview_inset(
    target_frame,
    camera_frame,
    state: CameraMapperState,
) -> None:
    if not state.marker_detection_enabled and not state.pick_detection_enabled:
        return

    preview_frame = camera_frame.copy()

    if state.marker_roi_polygon is not None and not state.marker_points_locked:
        draw_marker_polygon(preview_frame, state.marker_roi_polygon, label="FRETBOARD ROI")
    elif state.marker_roi is not None and not state.marker_points_locked:
        draw_marker_roi(preview_frame, state.marker_roi)

    draw_marker_setup_points(preview_frame, state)
    draw_occlusion_hand_estimate(preview_frame, state)
    draw_background_hand_estimate(preview_frame, state)
    draw_nut_anchor_markers(preview_frame, state)
    draw_raw_hand_markers(preview_frame, state)
    draw_finger_markers(preview_frame, state)
    draw_marker_detection(preview_frame, state.last_marker_detection)
    draw_pick_overlay(preview_frame, state)

    target_height, target_width = target_frame.shape[:2]
    preview_width = min(420, max(280, int(target_width * 0.34)))
    preview_height = int(preview_width * preview_frame.shape[0] / preview_frame.shape[1])
    preview_height = min(preview_height, int(target_height * 0.38))
    preview_width = int(preview_height * preview_frame.shape[1] / preview_frame.shape[0])

    preview_frame = cv2.resize(
        preview_frame,
        (preview_width, preview_height),
        interpolation=cv2.INTER_AREA,
    )

    margin = 18
    left = target_width - preview_width - margin
    top = margin
    right = left + preview_width
    bottom = top + preview_height

    cv2.rectangle(
        target_frame,
        (left - 4, top - 28),
        (right + 4, bottom + 4),
        (18, 22, 26),
        -1,
        cv2.LINE_AA,
    )
    target_frame[top:bottom, left:right] = preview_frame
    cv2.rectangle(
        target_frame,
        (left, top),
        (right, bottom),
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        target_frame,
        "camera / ROI",
        (left, top - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def run_camera_mapper() -> None:
    args = parse_args()

    if args.finger_detect and not args.marker_detect:
        raise ValueError("--finger-detect requires --marker-detect")

    if args.nut_anchor_detect and not args.marker_detect:
        raise ValueError("--nut-anchor-detect requires --marker-detect")

    config = FretboardMappingConfig(
        virtual_width=800,
        virtual_height=240,
        start_fret=args.start_fret,
        visible_frets=args.visible_frets,
        string_order_top_to_bottom=(6, 5, 4, 3, 2, 1),
    )
    state = CameraMapperState(config, args.calibration_json)
    state.flip_mode = args.flip
    state.marker_only = args.marker_detect
    state.tracking_enabled = not args.no_track and not state.marker_only
    state.midi_pitch = args.midi_pitch
    state.pitch_source = "audio" if args.audio_pitch else "manual"
    state.display_mode = args.display_mode
    state.display_total_frets = args.display_total_frets
    state.display_block_size = args.display_block_size
    state.marker_detection_enabled = args.marker_detect
    state.hand_marker_color = args.hand_marker_color
    state.hand_marker_count = max(1, args.hand_marker_count)
    state.hand_marker_min_area = args.hand_marker_min_area
    state.finger_detection_enabled = args.finger_detect
    state.finger_marker_specs = dict(args.finger_markers)
    state.finger_marker_min_area = args.finger_marker_min_area
    state.pick_detection_enabled = args.pick_detect
    state.pick_marker_color = args.pick_marker_color
    state.pick_marker_min_area = args.pick_marker_min_area
    state.marker_roi = parse_roi(args.marker_roi)
    state.marker_roi_user_defined = state.marker_roi is not None
    state.marker_corner_count = max(1, args.marker_corner_count)
    state.marker_stable_frames = max(1, args.marker_stable_frames)
    state.marker_min_confidence = args.marker_min_confidence
    state.marker_hold_frames = max(0, args.marker_hold_frames)
    state.marker_point_tracking_requested = args.marker_track_points
    state.marker_homography_enabled = not args.no_marker_homography_track
    state.marker_homography_smoothing = min(1.0, max(0.01, args.marker_homography_smoothing))
    state.marker_homography_min_inliers = max(
        state.marker_homography_min_points,
        args.marker_homography_min_inliers,
    )
    state.marker_homography_max_median_jump = max(
        1.0,
        args.marker_homography_max_median_jump,
    )
    state.marker_homography_roi_margin = max(0, args.marker_homography_roi_margin)
    state.occlusion_hand_enabled = not args.no_occlusion_hand
    state.occlusion_hand_min_hidden = max(1, args.occlusion_hand_min_hidden)
    state.occlusion_hand_fret_radius = max(0, args.occlusion_hand_fret_radius)
    state.audio_hand_match_enabled = not args.no_audio_hand_match
    state.background_hand_enabled = not args.no_background_hand
    state.background_hand_threshold = max(1, args.background_hand_threshold)
    state.background_hand_min_area = max(1.0, args.background_hand_min_area)
    state.background_hand_fret_radius = max(0, args.background_hand_fret_radius)
    state.nut_anchor_enabled = args.nut_anchor_detect
    state.nut_anchor_colors = tuple(args.nut_anchor_colors or ())
    state.nut_anchor_block_color = args.nut_anchor_block_color
    state.nut_anchor_min_area = args.nut_anchor_min_area
    state.nut_anchor_roi_margin = max(0, args.nut_anchor_roi_margin)
    state.nut_anchor_smoothing = min(1.0, max(0.01, args.nut_anchor_smoothing))
    state.nut_anchor_max_step = max(1.0, args.nut_anchor_max_step)

    if args.marker_detect:
        state.begin_marker_auto_setup()

    if args.pick_detect:
        state.begin_pick_roi_selection()

    if state.marker_only:
        print("marker-only mode: manual click calibration is disabled")
    else:
        state.load_existing_calibration()

    sync_recorder = SyncSessionRecorder(
        SyncSessionConfig(
            output_dir=args.sync_output_dir,
            sample_rate=args.audio_sample_rate,
            record_audio=not args.sync_no_audio,
            record_video=not args.sync_no_video,
            video_fps=args.sync_video_fps,
            video_codec=args.sync_video_codec,
        )
    )
    gpio_controller = None
    audio_tracker = None

    def start_audio_tracker_if_needed() -> None:
        nonlocal audio_tracker

        if audio_tracker is not None:
            return

        audio_device = (
            int(args.audio_device)
            if args.audio_device and args.audio_device.isdigit()
            else args.audio_device
        )
        audio_tracker = AudioPitchTracker(
            device=audio_device,
            sample_rate=args.audio_sample_rate,
            block_size=args.audio_block_size,
            rms_threshold=args.audio_rms_threshold,
            audio_block_callback=sync_recorder.add_audio_block,
        )

        try:
            audio_tracker.start()
            print("audio input started")
        except Exception as error:
            audio_tracker = None
            print(f"audio input disabled: {error}")

    def make_sync_metadata() -> dict:
        return {
            "recording_mode": "camera_pi",
            "camera": {
                "index": args.camera_index,
                "width": args.width,
                "height": args.height,
                "flip": state.flip_mode,
            },
            "fretboard": {
                "display_total_frets": state.display_total_frets,
                "display_block_size": state.display_block_size,
                "start_fret": state.config.start_fret,
                "visible_frets": state.config.visible_frets,
                "string_order_top_to_bottom": list(state.config.string_order_top_to_bottom),
            },
            "markers": {
                "enabled": state.marker_detection_enabled,
                "edge_pair_counts": dict(EDGE_PAIR_MARKER_COUNTS),
                "roi": state.marker_roi,
                "homography_enabled": state.marker_homography_enabled,
                "nut_anchor_enabled": state.nut_anchor_enabled,
            },
            "audio_pitch": {
                "enabled": args.audio_pitch,
                "device": args.audio_device,
                "sample_rate": args.audio_sample_rate,
                "block_size": args.audio_block_size,
            },
        }

    def set_sync_enabled(enabled: bool) -> None:
        if enabled:
            if not args.sync_no_audio:
                start_audio_tracker_if_needed()
            sync_recorder.start(metadata=make_sync_metadata())
        else:
            sync_recorder.stop()

        state.sync_enabled = sync_recorder.enabled

        if gpio_controller is not None:
            gpio_controller.set_led(state.sync_enabled)

    def toggle_sync() -> None:
        set_sync_enabled(not sync_recorder.enabled)

    gpio_controller = SyncGPIOController(
        button_pin=args.gpio_button_pin,
        led_pin=args.gpio_led_pin,
        on_toggle=toggle_sync,
        enabled=not args.no_gpio,
    )

    pending_sync_start = args.sync_start_on

    if args.sync_start_on and not state.marker_scan_pending:
        set_sync_enabled(True)

    if args.audio_pitch:
        start_audio_tracker_if_needed()

    capture = cv2.VideoCapture(args.camera_index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not capture.isOpened():
        raise RuntimeError(f"カメラを開けませんでした: index={args.camera_index}")

    def handle_mouse_click(event, x, y, flags, param):
        if state.pick_roi_selection_pending:
            if event == cv2.EVENT_LBUTTONDOWN:
                state.add_pick_roi_point((x, y))
            return

        if state.marker_only:
            if state.marker_roi_selection_pending:
                if event == cv2.EVENT_LBUTTONDOWN:
                    state.add_marker_roi_point((x, y))
                return

            if state.marker_setup_pending:
                if event == cv2.EVENT_LBUTTONDOWN:
                    state.drag_marker_index = state.find_nearest_marker_point(x, y)

                    if state.drag_marker_index is None:
                        state.add_missing_marker_point(x, y)
                        state.drag_marker_index = state.find_nearest_marker_point(x, y)
                elif event == cv2.EVENT_MOUSEMOVE and state.drag_marker_index is not None:
                    state.marker_points[state.drag_marker_index]["center"] = (x, y)
                elif event == cv2.EVENT_LBUTTONUP:
                    if state.drag_marker_index is not None:
                        state.marker_points[state.drag_marker_index]["center"] = (x, y)
                    state.drag_marker_index = None
                return

            return

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if state.mode == "calibrating":
            state.add_calibration_click(x, y)
        elif state.display_mode == "fretboard":
            display_frame = np.zeros(
                (FRETBOARD_DISPLAY_HEIGHT, FRETBOARD_DISPLAY_WIDTH, 3),
                dtype=np.uint8,
            )
            display_position = fretboard_display.map_display_point_to_fret_position(
                x,
                y,
                display_frame,
                state.config,
                total_frets=state.display_total_frets,
            )

            if display_position is not None:
                state.set_display_position_result(display_position)
        else:
            state.handle_mapping_click(x, y)

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, handle_mouse_click)

    print("ライブカメラを開始しました")
    if state.marker_detection_enabled:
        print("controls: b: bg hand / f: flip / v: display / o: sync / [: MIDI-1 / ]: MIDI+1 / q: quit")
        print(
            "marker detection: ON "
            "edge-pairs=green:2/blue:2/yellow:4/pink:4 "
            f"hand={state.hand_marker_color or 'none'}"
        )
        if state.hand_marker_color is not None:
            print(f"required hand markers: {state.hand_marker_count}")
            print(f"hand marker min area: {state.hand_marker_min_area}")
        if state.finger_detection_enabled:
            marker_text = ", ".join(
                f"{finger}:{color}"
                for finger, color in state.finger_marker_specs.items()
            )
            print(f"finger markers: {marker_text}")
            print(f"finger marker min area: {state.finger_marker_min_area}")
        if state.marker_roi is not None:
            print(f"marker ROI: {state.marker_roi}")
        else:
            print("marker ROI: none. Auto-scanning the full camera frame for 12 markers.")
        print("required edge-pair markers: green=2 blue=2 yellow=4 pink=4")
        print(
            "marker homography: "
            f"{'ON' if state.marker_homography_enabled else 'OFF'} "
            f"smoothing={state.marker_homography_smoothing:.2f} "
            f"min_inliers={state.marker_homography_min_inliers} "
            f"max_jump={state.marker_homography_max_median_jump:.0f} "
            f"roi_margin={state.marker_homography_roi_margin}"
        )
        if state.nut_anchor_enabled:
            print(
                "nut anchor: ON "
                f"anchors={state.nut_anchor_description()} "
                f"min_area={state.nut_anchor_min_area} "
                f"roi_margin={state.nut_anchor_roi_margin} "
                f"smoothing={state.nut_anchor_smoothing:.2f} "
                f"max_step={state.nut_anchor_max_step:.0f}"
            )
        print(
            "marker stability: "
            f"{state.marker_stable_frames} frames, "
            f"min_confidence={state.marker_min_confidence:.2f}, "
            f"hold={state.marker_hold_frames} frames"
        )
        if state.background_hand_enabled:
            print(
                "background hand: ON "
                f"threshold={state.background_hand_threshold} "
                f"min_area={state.background_hand_min_area:.0f}. "
                "After marker lock, remove your hand and press b."
            )
        print("First, the program auto-detects the 12 fretboard markers. Drag only if a point is wrong.")
    else:
        print("c: calibration / s: save / t: tracking / v: display / o: sync / [: MIDI-1 / ]: MIDI+1 / q: quit")
        print("キャリブレーション後、指板上をクリックすると推定結果を表示します")
    if state.midi_pitch is None:
        print("仮MIDIは未設定です。] キーで D4 (MIDI 62) から開始できます")
    else:
        print(
            "manual MIDI pitch: "
            f"{midi_pitch_to_note_name(state.midi_pitch)} (MIDI {state.midi_pitch})"
        )

    try:
        while True:
            ok, frame = capture.read()

            if not ok:
                print("カメラフレームを取得できませんでした")
                break

            frame = apply_frame_flip(frame, state.flip_mode)
            sync_recorder.record_video_frame(frame)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if not state.marker_only:
                state.update_tracking(gray_frame)
            elif state.marker_tracking_enabled:
                state.update_marker_point_tracking(gray_frame)

            if (
                state.marker_setup_pending
                and not state.marker_points
            ):
                state.initialize_marker_setup_points(frame, min_area=args.marker_min_area)

            was_marker_scan_pending = state.marker_scan_pending
            state.update_nut_anchor_detection(frame)
            state.update_marker_detection(frame, min_area=args.marker_min_area)
            state.update_background_hand_estimate(frame)
            state.update_pick_detection(frame)

            if (
                was_marker_scan_pending
                and not state.marker_scan_pending
                and pending_sync_start
                and not sync_recorder.enabled
            ):
                set_sync_enabled(True)

            if audio_tracker is not None:
                detection = audio_tracker.poll()

                if args.audio_pitch:
                    state.update_audio_pitch(detection)

            sync_recorder.drain_audio()

            if (
                (state.mode == "calibrating" and not state.marker_only)
                or state.display_mode == "camera"
                or state.is_marker_scan_active()
                or state.is_pick_setup_active()
            ):
                display_frame = frame.copy()
                draw_overlay(display_frame, state)
            else:
                display_frame = np.zeros(
                    (FRETBOARD_DISPLAY_HEIGHT, FRETBOARD_DISPLAY_WIDTH, 3),
                    dtype=np.uint8,
                )
                fretboard_display.draw_fretboard_diagram(
                    display_frame,
                    state.config,
                    fret_boundaries=state.fret_boundaries,
                    highlight_position=get_highlight_position(state),
                    total_frets=state.display_total_frets,
                    block_size=state.display_block_size,
                    current_block_start=fretboard_display.get_current_block_start(
                        (
                            get_highlight_position(state)["fret_number"]
                            if get_highlight_position(state) is not None
                            else state.config.start_fret
                        ),
                        block_size=state.display_block_size,
                    ),
                    title="FretTone Camera Sync",
                    subtitle="12 frets / edge-pair markers",
                )
                draw_occlusion_hand_display_band(display_frame, state)
                draw_background_hand_display_band(display_frame, state)
                draw_status_overlay(
                    display_frame,
                    state,
                    start_y=display_frame.shape[0]
                    - (
                        156
                        + (36 if state.pick_detection_enabled else 0)
                        + (
                            36
                            if state.occlusion_hand_enabled and state.marker_points_locked
                            else 0
                        )
                        + (
                            36
                            if state.background_hand_enabled and state.marker_points_locked
                            else 0
                        )
                        + (
                            36
                            if state.audio_hand_match_enabled and state.marker_points_locked
                            else 0
                        )
                        + (36 if state.finger_detection_enabled else 0)
                    ),
                )
                draw_camera_preview_inset(display_frame, frame, state)

            highlight_position = get_highlight_position(state)
            current_block_start = fretboard_display.get_current_block_start(
                (
                    highlight_position["fret_number"]
                    if highlight_position is not None
                    else state.config.start_fret
                ),
                block_size=state.display_block_size,
            )
            sync_recorder.record_highlight(
                highlight_position=highlight_position,
                midi_pitch=state.midi_pitch,
                note_name=(
                    midi_pitch_to_note_name(state.midi_pitch)
                    if state.midi_pitch is not None
                    else None
                ),
                pitch_source=state.pitch_source,
                current_block_start=current_block_start,
                match_result=state.last_match_result,
            )

            cv2.imshow(WINDOW_NAME, display_frame)
            state.advance_marker_scan_confirmation()

            key = cv2.waitKey(1) & 0xFF

            if key in (13, 10, 32) and state.pick_roi_selection_pending:
                state.confirm_pick_roi_selection()
                continue

            if key in (13, 10, 32) and state.marker_roi_selection_pending:
                state.confirm_marker_roi_selection()
                continue

            if key in (13, 10, 32) and state.marker_setup_pending:
                state.begin_marker_tracking(gray_frame)
                continue

            if key == ord("c"):
                if state.marker_only:
                    print("marker-only mode: manual calibration is disabled")
                elif state.pick_detection_enabled:
                    state.begin_pick_roi_selection()
                    print("display flipped. Please select the pick ROI again.")
                else:
                    state.start_calibration()
            elif key == ord("s"):
                if state.marker_only:
                    print("marker-only mode: calibration save is disabled")
                else:
                    state.save_calibration()
            elif key == ord("f"):
                state.flip_mode = next_flip_mode(state.flip_mode)
                print(f"反転モードを変更しました: {state.flip_mode}")

                if state.marker_only:
                    state.begin_marker_auto_setup()
                    print("表示が変わったため、指板範囲の指定からやり直します")
                else:
                    state.start_calibration()
                    print("表示が変わったため、カメラ映像上で再キャリブレーションしてください")
            elif key == ord("t"):
                if state.marker_only:
                    print("marker-only mode: tracking is disabled")
                else:
                    state.set_tracking_enabled(not state.tracking_enabled)
            elif key == ord("r") and state.marker_only:
                state.marker_points = []
                state.marker_setup_pending = True
                state.marker_points_locked = False
                state.reference_marker_points = []
                state.marker_homography_active = False
                state.marker_homography_inliers = 0
                state.background_hand_reference = None
                state.background_hand_canonical_points = {}
                state.background_hand_homography = None
                state.background_hand_estimate = None
                state.background_hand_last_area = 0.0
                state.background_hand_largest_area = 0.0
                state.background_hand_changed_pixels = 0
                state.marker_setup_last_signature = None
                state.reference_nut_anchor_points = {}
                state.nut_anchor_transform_active = False
                state.marker_tracking_enabled = False
                state.reset_marker_stability()
                print("marker setup redetect requested")
            elif key == ord("b"):
                state.capture_background_hand_reference(frame)
            elif key == ord("v"):
                state.toggle_display_mode()
            elif key == ord("o"):
                toggle_sync()
            elif key == ord("["):
                state.adjust_midi_pitch(-1)
            elif key == ord("]"):
                state.adjust_midi_pitch(1)
            elif key == ord("q"):
                break
    finally:
        set_sync_enabled(False)

        if audio_tracker is not None:
            audio_tracker.stop()

        if gpio_controller is not None:
            gpio_controller.close()

        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera_mapper()
