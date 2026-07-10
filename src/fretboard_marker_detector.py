from __future__ import annotations

from dataclasses import dataclass
from math import hypot

import cv2
import numpy as np


@dataclass(frozen=True)
class ColorRange:
    name: str
    hsv_ranges: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]
    draw_color_bgr: tuple[int, int, int]


@dataclass(frozen=True)
class BlockMarkerSpec:
    color_name: str
    block_name: str
    start_fret: int
    visible_frets: int = 4


@dataclass(frozen=True)
class MarkerDetection:
    color_name: str
    center: tuple[int, int]
    area: float
    bounding_box: tuple[int, int, int, int]
    confidence: float


@dataclass(frozen=True)
class FretboardBlockDetection:
    block_name: str
    start_fret: int
    visible_frets: int
    marker_color: str
    confidence: float
    marker: MarkerDetection
    markers: tuple[MarkerDetection, ...]
    block_rect: tuple[int, int, int, int]
    hand_marker: MarkerDetection | None = None
    hand_markers: tuple[MarkerDetection, ...] = ()


COLOR_RANGES: dict[str, ColorRange] = {
    "blue": ColorRange(
        name="blue",
        hsv_ranges=(((85, 45, 35), (140, 255, 255)),),
        draw_color_bgr=(255, 120, 20),
    ),
    "yellow": ColorRange(
        name="yellow",
        hsv_ranges=(((18, 70, 70), (42, 255, 255)),),
        draw_color_bgr=(0, 220, 255),
    ),
    "white": ColorRange(
        name="white",
        hsv_ranges=(((0, 0, 150), (179, 70, 255)),),
        draw_color_bgr=(235, 235, 235),
    ),
    "red": ColorRange(
        name="red",
        hsv_ranges=(
            ((0, 80, 50), (10, 255, 255)),
            ((170, 80, 50), (179, 255, 255)),
        ),
        draw_color_bgr=(0, 0, 255),
    ),
    "green": ColorRange(
        name="green",
        hsv_ranges=(((45, 60, 45), (85, 255, 255)),),
        draw_color_bgr=(0, 210, 80),
    ),
    "pink": ColorRange(
        name="pink",
        hsv_ranges=(((128, 40, 55), (179, 255, 255)),),
        draw_color_bgr=(210, 60, 255),
    ),
    "purple": ColorRange(
        name="purple",
        hsv_ranges=(((125, 45, 45), (152, 255, 255)),),
        draw_color_bgr=(170, 60, 210),
    ),
    "cyan": ColorRange(
        name="cyan",
        hsv_ranges=(((78, 45, 55), (98, 255, 255)),),
        draw_color_bgr=(255, 220, 40),
    ),
    "orange": ColorRange(
        name="orange",
        hsv_ranges=(((8, 80, 70), (22, 255, 255)),),
        draw_color_bgr=(0, 150, 255),
    ),
}

DEFAULT_BLOCK_MARKERS: tuple[BlockMarkerSpec, ...] = (
    BlockMarkerSpec(color_name="blue", block_name="1-4F", start_fret=1),
    BlockMarkerSpec(color_name="yellow", block_name="5-8F", start_fret=5),
    BlockMarkerSpec(color_name="pink", block_name="9-12F", start_fret=9),
)


def parse_roi(roi_text: str | None) -> tuple[int, int, int, int] | None:
    if roi_text is None or roi_text.strip() == "":
        return None

    values = [int(part.strip()) for part in roi_text.split(",")]

    if len(values) != 4:
        raise ValueError("--marker-roi must be x,y,width,height")

    x, y, width, height = values

    if width <= 0 or height <= 0:
        raise ValueError("--marker-roi width and height must be greater than 0")

    return x, y, width, height


def clamp_roi(frame, roi: tuple[int, int, int, int] | None):
    if roi is None:
        return None

    frame_height, frame_width = frame.shape[:2]
    x, y, width, height = roi
    left = max(0, min(frame_width - 1, x))
    top = max(0, min(frame_height - 1, y))
    right = max(left + 1, min(frame_width, x + width))
    bottom = max(top + 1, min(frame_height, y + height))
    return left, top, right - left, bottom - top


def polygon_to_roi(
    polygon: tuple[tuple[int, int], ...] | list[tuple[int, int]] | None,
) -> tuple[int, int, int, int] | None:
    if not polygon:
        return None

    left = min(point[0] for point in polygon)
    top = min(point[1] for point in polygon)
    right = max(point[0] for point in polygon)
    bottom = max(point[1] for point in polygon)
    return left, top, right - left, bottom - top


def crop_frame_to_roi(
    frame,
    roi: tuple[int, int, int, int] | None,
    roi_polygon: tuple[tuple[int, int], ...] | list[tuple[int, int]] | None = None,
):
    clamped_roi = clamp_roi(frame, polygon_to_roi(roi_polygon) or roi)

    if clamped_roi is None:
        return frame, (0, 0), None, None

    x, y, width, height = clamped_roi
    mask = None

    if roi_polygon:
        shifted_points = np.array(
            [[point[0] - x, point[1] - y] for point in roi_polygon],
            dtype=np.int32,
        )
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [shifted_points], 255)

    return frame[y : y + height, x : x + width], (x, y), clamped_roi, mask


def offset_marker_detection(
    detection: MarkerDetection,
    offset: tuple[int, int],
) -> MarkerDetection:
    offset_x, offset_y = offset
    x, y, width, height = detection.bounding_box
    return MarkerDetection(
        color_name=detection.color_name,
        center=(detection.center[0] + offset_x, detection.center[1] + offset_y),
        area=detection.area,
        bounding_box=(x + offset_x, y + offset_y, width, height),
        confidence=detection.confidence,
    )


def make_color_mask(
    hsv_frame,
    color_range: ColorRange,
    kernel_size: int = 5,
):
    mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)

    for lower, upper in color_range.hsv_ranges:
        mask = cv2.bitwise_or(
            mask,
            cv2.inRange(
                hsv_frame,
                np.array(lower, dtype=np.uint8),
                np.array(upper, dtype=np.uint8),
            ),
        )

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def contour_confidence(contour, frame_area: int) -> float:
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))

    if perimeter <= 0 or frame_area <= 0:
        return 0.0

    circularity = min(1.0, (4.0 * np.pi * area) / (perimeter * perimeter))
    area_score = min(1.0, area / max(1.0, frame_area * 0.0025))
    return float(0.65 * area_score + 0.35 * circularity)


def detect_markers_for_color(
    frame,
    color_name: str,
    min_area: float = 80.0,
    roi: tuple[int, int, int, int] | None = None,
    roi_polygon: tuple[tuple[int, int], ...] | list[tuple[int, int]] | None = None,
) -> list[MarkerDetection]:
    if color_name not in COLOR_RANGES:
        raise ValueError(f"Unknown marker color: {color_name}")

    detection_frame, offset, _, roi_mask = crop_frame_to_roi(
        frame,
        roi=roi,
        roi_polygon=roi_polygon,
    )
    hsv_frame = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2HSV)
    mask = make_color_mask(hsv_frame, COLOR_RANGES[color_name])

    if roi_mask is not None:
        mask = cv2.bitwise_and(mask, roi_mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = detection_frame.shape[0] * detection_frame.shape[1]
    detections: list[MarkerDetection] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))

        if area < min_area:
            continue

        moments = cv2.moments(contour)

        if moments["m00"] == 0:
            continue

        center = (
            int(moments["m10"] / moments["m00"]),
            int(moments["m01"] / moments["m00"]),
        )
        x, y, width, height = cv2.boundingRect(contour)
        detection = MarkerDetection(
            color_name=color_name,
            center=center,
            area=area,
            bounding_box=(x, y, width, height),
            confidence=contour_confidence(contour, frame_area),
        )
        detections.append(
            offset_marker_detection(detection, offset)
            if offset != (0, 0)
            else detection
        )

    return sorted(detections, key=lambda item: item.confidence, reverse=True)


def detect_all_markers(
    frame,
    color_names: tuple[str, ...],
    min_area: float = 80.0,
    roi: tuple[int, int, int, int] | None = None,
    roi_polygon: tuple[tuple[int, int], ...] | list[tuple[int, int]] | None = None,
) -> dict[str, list[MarkerDetection]]:
    return {
        color_name: detect_markers_for_color(
            frame,
            color_name=color_name,
            min_area=min_area,
            roi=roi,
            roi_polygon=roi_polygon,
        )
        for color_name in color_names
    }


def make_block_rect(
    markers: tuple[MarkerDetection, ...],
    padding: int = 10,
) -> tuple[int, int, int, int]:
    left = min(marker.bounding_box[0] for marker in markers) - padding
    top = min(marker.bounding_box[1] for marker in markers) - padding
    right = max(
        marker.bounding_box[0] + marker.bounding_box[2]
        for marker in markers
    ) + padding
    bottom = max(
        marker.bounding_box[1] + marker.bounding_box[3]
        for marker in markers
    ) + padding
    return left, top, right - left, bottom - top


def choose_block_corner_markers(
    markers: list[MarkerDetection],
    required_corner_count: int = 4,
) -> tuple[MarkerDetection, ...] | None:
    if len(markers) < required_corner_count:
        return None

    ranked_markers = sorted(
        markers,
        key=lambda marker: (marker.confidence, marker.area),
        reverse=True,
    )
    return tuple(ranked_markers[:required_corner_count])


def get_marker_group_center(markers: tuple[MarkerDetection, ...]) -> tuple[float, float]:
    if not markers:
        return 0.0, 0.0

    return (
        sum(marker.center[0] for marker in markers) / len(markers),
        sum(marker.center[1] for marker in markers) / len(markers),
    )


def point_inside_rect(
    point: tuple[int, int],
    rect: tuple[int, int, int, int],
    margin: int = 0,
) -> bool:
    x, y = point
    left, top, width, height = rect
    return (
        left - margin <= x <= left + width + margin
        and top - margin <= y <= top + height + margin
    )


def choose_block_by_hand_markers(
    block_candidates: list[FretboardBlockDetection],
    hand_markers: tuple[MarkerDetection, ...],
    frame_shape,
    current_start_fret: int | None = None,
) -> FretboardBlockDetection | None:
    if not block_candidates or not hand_markers:
        return None

    containing_candidates = []

    for candidate in block_candidates:
        contained_markers = tuple(
            hand_marker
            for hand_marker in hand_markers
            if point_inside_rect(hand_marker.center, candidate.block_rect, margin=12)
        )

        if contained_markers:
            containing_candidates.append((candidate, contained_markers))

    if containing_candidates:
        if current_start_fret is not None:
            moved_candidates = [
                (candidate, contained_markers)
                for candidate, contained_markers in containing_candidates
                if candidate.start_fret != current_start_fret
            ]

            if moved_candidates:
                containing_candidates = moved_candidates

        best_candidate, contained_markers = max(
            containing_candidates,
            key=lambda item: (
                len(item[1]),
                item[0].confidence,
                -abs(
                    item[0].start_fret
                    - (current_start_fret if current_start_fret is not None else item[0].start_fret)
                ),
            ),
        )
        return FretboardBlockDetection(
            block_name=best_candidate.block_name,
            start_fret=best_candidate.start_fret,
            visible_frets=best_candidate.visible_frets,
            marker_color=best_candidate.marker_color,
            confidence=float(min(1.0, best_candidate.confidence + 0.18)),
            marker=best_candidate.marker,
            markers=best_candidate.markers,
            block_rect=best_candidate.block_rect,
            hand_marker=contained_markers[0],
            hand_markers=hand_markers,
        )

    return None


def detect_fretboard_block(
    frame,
    block_specs: tuple[BlockMarkerSpec, ...] = DEFAULT_BLOCK_MARKERS,
    hand_marker_color: str | None = None,
    required_hand_marker_count: int = 1,
    hand_marker_min_area: float | None = None,
    min_area: float = 80.0,
    roi: tuple[int, int, int, int] | None = None,
    roi_polygon: tuple[tuple[int, int], ...] | list[tuple[int, int]] | None = None,
    required_corner_count: int = 4,
    current_start_fret: int | None = None,
) -> FretboardBlockDetection | None:
    block_color_names = tuple(sorted({spec.color_name for spec in block_specs}))
    marker_map = detect_all_markers(
        frame,
        color_names=block_color_names,
        min_area=min_area,
        roi=roi,
        roi_polygon=roi_polygon,
    )

    if hand_marker_color is not None:
        marker_map[hand_marker_color] = detect_markers_for_color(
            frame,
            color_name=hand_marker_color,
            min_area=hand_marker_min_area if hand_marker_min_area is not None else min_area,
            roi=roi,
            roi_polygon=roi_polygon,
        )
    block_candidates: list[FretboardBlockDetection] = []

    for spec in block_specs:
        markers = marker_map.get(spec.color_name, [])

        corner_markers = choose_block_corner_markers(
            markers,
            required_corner_count=required_corner_count,
        )

        if corner_markers is None:
            continue

        primary_marker = corner_markers[0]
        confidence = float(
            sum(marker.confidence for marker in corner_markers)
            / len(corner_markers)
        )
        block_candidates.append(
            FretboardBlockDetection(
                block_name=spec.block_name,
                start_fret=spec.start_fret,
                visible_frets=spec.visible_frets,
                marker_color=spec.color_name,
                confidence=confidence,
                marker=primary_marker,
                markers=corner_markers,
                block_rect=make_block_rect(corner_markers),
            )
        )

    if not block_candidates:
        return None

    if hand_marker_color is not None:
        hand_markers = tuple(marker_map.get(hand_marker_color, [])[:required_hand_marker_count])

        if len(hand_markers) >= required_hand_marker_count:
            return choose_block_by_hand_markers(
                block_candidates,
                hand_markers=hand_markers,
                frame_shape=frame.shape,
                current_start_fret=current_start_fret,
            )

    return max(block_candidates, key=lambda item: item.confidence)


def draw_marker_detection(frame, detection: FretboardBlockDetection | None) -> None:
    if detection is None:
        cv2.putText(
            frame,
            "marker: not found",
            (16, 142),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (120, 120, 255),
            2,
            cv2.LINE_AA,
        )
        return

    color = COLOR_RANGES.get(detection.marker_color, COLOR_RANGES["yellow"]).draw_color_bgr
    block_x, block_y, block_width, block_height = detection.block_rect
    cv2.rectangle(
        frame,
        (block_x, block_y),
        (block_x + block_width, block_y + block_height),
        color,
        3,
        cv2.LINE_AA,
    )

    for marker in detection.markers:
        marker_color = COLOR_RANGES.get(
            marker.color_name,
            COLOR_RANGES["yellow"],
        ).draw_color_bgr
        x, y, width, height = marker.bounding_box
        cv2.rectangle(frame, (x, y), (x + width, y + height), marker_color, 2, cv2.LINE_AA)
        cv2.circle(frame, marker.center, 6, marker_color, -1, cv2.LINE_AA)

    if detection.hand_markers:
        hand_color = COLOR_RANGES[detection.hand_markers[0].color_name].draw_color_bgr

        for hand_marker in detection.hand_markers:
            hx, hy, hwidth, hheight = hand_marker.bounding_box
            cv2.rectangle(
                frame,
                (hx, hy),
                (hx + hwidth, hy + hheight),
                hand_color,
                2,
                cv2.LINE_AA,
            )
            cv2.circle(frame, hand_marker.center, 6, hand_color, -1, cv2.LINE_AA)

        if len(detection.hand_markers) >= 2:
            first_hand_marker = detection.hand_markers[0]
            second_hand_marker = detection.hand_markers[1]
            cv2.line(
                frame,
                first_hand_marker.center,
                second_hand_marker.center,
                hand_color,
                2,
                cv2.LINE_AA,
            )
            hand_center = get_marker_group_center(detection.hand_markers)
            cv2.circle(
                frame,
                (int(hand_center[0]), int(hand_center[1])),
                7,
                hand_color,
                -1,
                cv2.LINE_AA,
            )

        cv2.line(frame, detection.marker.center, detection.hand_markers[0].center, color, 2, cv2.LINE_AA)
    elif detection.hand_marker is not None:
        hand_color = COLOR_RANGES[detection.hand_marker.color_name].draw_color_bgr
        hx, hy, hwidth, hheight = detection.hand_marker.bounding_box
        cv2.rectangle(
            frame,
            (hx, hy),
            (hx + hwidth, hy + hheight),
            hand_color,
            2,
            cv2.LINE_AA,
        )
        cv2.circle(frame, detection.hand_marker.center, 6, hand_color, -1, cv2.LINE_AA)
        cv2.line(frame, detection.marker.center, detection.hand_marker.center, color, 2, cv2.LINE_AA)

    cv2.putText(
        frame,
        (
            f"marker: {detection.block_name} "
            f"{detection.marker_color} conf:{detection.confidence:.2f}"
        ),
        (16, 142),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_marker_roi(frame, roi: tuple[int, int, int, int] | None) -> None:
    clamped_roi = clamp_roi(frame, roi)

    if clamped_roi is None:
        return

    x, y, width, height = clamped_roi
    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_marker_polygon(
    frame,
    polygon: tuple[tuple[int, int], ...] | list[tuple[int, int]] | None,
    label: str = "ROI",
) -> None:
    if not polygon:
        return

    points = np.array(polygon, dtype=np.int32)
    closed = len(points) >= 3
    cv2.polylines(frame, [points], closed, (0, 255, 255), 2, cv2.LINE_AA)

    for index, point in enumerate(points):
        center = (int(point[0]), int(point[1]))
        cv2.circle(frame, center, 6, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            str(index + 1),
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    first_point = tuple(points[0].tolist())
    cv2.putText(
        frame,
        label,
        (first_point[0] + 8, first_point[1] + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
