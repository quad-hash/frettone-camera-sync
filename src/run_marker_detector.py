import argparse

import cv2

from fretboard_marker_detector import (
    COLOR_RANGES,
    detect_fretboard_block,
    draw_marker_detection,
    draw_marker_roi,
    parse_roi,
)
from run_with_camera_pi import FLIP_MODES, apply_frame_flip


WINDOW_NAME = "FretTone - Marker Detector"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect colored fretboard block stickers from a live camera.",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--flip", choices=FLIP_MODES, default="none")
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
    parser.add_argument("--marker-min-area", type=float, default=80.0)
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
    return parser.parse_args()


def run_marker_detector() -> None:
    args = parse_args()
    capture = cv2.VideoCapture(args.camera_index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera: index={args.camera_index}")

    last_block_name = None
    marker_roi = parse_roi(args.marker_roi)
    marker_corner_count = max(1, args.marker_corner_count)
    cv2.namedWindow(WINDOW_NAME)
    print("marker detector started")
    print("blue:1-4F / yellow:5-8F / pink:9-12F")
    print(f"hand marker: {args.hand_marker_color or 'none'}")
    print(f"required hand markers: {max(1, args.hand_marker_count)}")
    print(f"hand marker min area: {args.hand_marker_min_area}")
    print(f"marker ROI: {marker_roi or 'full frame'}")
    print(f"required corner markers per block: {marker_corner_count}")
    print("q: quit")

    try:
        while True:
            ok, frame = capture.read()

            if not ok:
                print("Could not read camera frame")
                break

            frame = apply_frame_flip(frame, args.flip)
            detection = detect_fretboard_block(
                frame,
                hand_marker_color=args.hand_marker_color,
                required_hand_marker_count=max(1, args.hand_marker_count),
                hand_marker_min_area=args.hand_marker_min_area,
                min_area=args.marker_min_area,
                roi=marker_roi,
                roi_polygon=None,
                required_corner_count=marker_corner_count,
            )

            if detection is not None and detection.block_name != last_block_name:
                last_block_name = detection.block_name
                print(
                    f"{detection.block_name} "
                    f"start_fret={detection.start_fret} "
                    f"color={detection.marker_color} "
                    f"confidence={detection.confidence:.2f}"
                )

            draw_marker_roi(frame, marker_roi)
            draw_marker_detection(frame, detection)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_marker_detector()
