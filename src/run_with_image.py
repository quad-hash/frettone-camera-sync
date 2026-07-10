# filename: src/run_with_image.py
# role: サンプル画像上をクリックして、何弦・何フレットかを表示する実行スクリプト

import argparse
from pathlib import Path

import cv2
import numpy as np

from fretboard_position_mapper import (
    FretboardMappingConfig,
    build_perspective_matrix,
    estimate_string_and_fret,
    load_calibration_data,
    map_image_point_to_virtual_point,
)
from fretboard_note_mapper import match_camera_estimate_with_midi_pitch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_JSON_PATH = PROJECT_ROOT / "data" / "calibration_points.json"
WINDOW_NAME = "FretTone Camera Sync - Image Test"


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Click a sample fretboard image and estimate string/fret position.",
    )
    parser.add_argument("--start-fret", type=int, default=5)
    parser.add_argument("--visible-frets", type=int, default=4)
    parser.add_argument(
        "--midi-pitch",
        type=int,
        default=None,
        help="Optional MIDI pitch to match against the clicked camera estimate.",
    )

    return parser.parse_args()


def find_sample_image_path() -> Path:
    """sample_images内のサンプル画像を探す"""

    candidate_paths = [
        PROJECT_ROOT / "sample_images" / "fretboard_sample.jpg",
        PROJECT_ROOT / "sample_images" / "fretboard_sample.jpeg",
        PROJECT_ROOT / "sample_images" / "fretboard_sample.png",
    ]

    for image_path in candidate_paths:
        if image_path.exists():
            return image_path

    raise FileNotFoundError(
        "sample_images/fretboard_sample.jpg, .jpeg, .png のどれも見つかりません"
    )


def draw_calibration_frame(
    target_image,
    calibration_points,
) -> None:
    """キャリブレーション済みの指板枠を画像上に描画する"""

    points = calibration_points.astype(int).tolist()
    points = [tuple(point) for point in points]
    closed_points = points + [points[0]]

    for index in range(4):
        cv2.line(
            target_image,
            closed_points[index],
            closed_points[index + 1],
            (0, 255, 0),
            2,
        )

    for index, point in enumerate(points):
        cv2.circle(target_image, point, 6, (0, 0, 255), -1)
        cv2.putText(
            target_image,
            str(index + 1),
            (point[0] + 8, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )


def draw_fret_boundaries(
    target_image,
    perspective_matrix,
    fret_boundaries,
    config: FretboardMappingConfig,
) -> None:
    """キャリブレーション済みのフレット境界線を画像上に描画する"""

    if fret_boundaries is None:
        return

    inverse_matrix = cv2.invert(perspective_matrix)[1]

    for boundary_x in fret_boundaries[1:-1]:
        virtual_line_points = np.array(
            [
                [[boundary_x, 0]],
                [[boundary_x, config.virtual_height]],
            ],
            dtype=np.float32,
        )
        image_line_points = cv2.perspectiveTransform(
            virtual_line_points,
            inverse_matrix,
        )

        top_point = tuple(image_line_points[0][0].astype(int))
        bottom_point = tuple(image_line_points[1][0].astype(int))

        cv2.line(target_image, top_point, bottom_point, (0, 255, 255), 2)


def draw_estimation_result(
    target_image,
    click_x: int,
    click_y: int,
    result: dict,
) -> None:
    """クリック位置と推定結果を画像上に描画する"""

    cv2.circle(target_image, (click_x, click_y), 8, (255, 0, 0), -1)

    if result["inside_fretboard"]:
        message = f'{result["string_number"]} string / {result["fret_number"]} fret'
    else:
        message = "outside fretboard"

    cv2.putText(
        target_image,
        message,
        (click_x + 10, click_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
    )


def format_pitch_match_result(match_result: dict) -> str:
    selected_position = match_result["selected_position"]

    if selected_position is None:
        return (
            f'pitch {match_result["note_name"]} '
            f'(MIDI {match_result["midi_pitch"]}): no candidate in window'
        )

    return (
        f'pitch {match_result["note_name"]} '
        f'(MIDI {match_result["midi_pitch"]}) -> '
        f'{selected_position["string_number"]} string '
        f'{selected_position["fret_number"]} fret '
        f'[{match_result["status"]}]'
    )


def run_position_mapper_with_image() -> None:
    """サンプル画像を使って、クリック位置から弦・フレットを推定する"""

    args = parse_args()
    image_path = find_sample_image_path()
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"画像を読み込めませんでした: {image_path}")

    config = FretboardMappingConfig(
        virtual_width=800,
        virtual_height=240,
        start_fret=args.start_fret,
        visible_frets=args.visible_frets,
        string_order_top_to_bottom=(6, 5, 4, 3, 2, 1),
    )

    calibration_data = load_calibration_data(CALIBRATION_JSON_PATH)
    calibration_points = calibration_data.points
    fret_boundaries = get_compatible_fret_boundaries(
        calibration_data.fret_boundaries,
        config,
    )
    perspective_matrix = build_perspective_matrix(calibration_points, config)

    display_image = image.copy()
    draw_calibration_frame(display_image, calibration_points)
    draw_fret_boundaries(display_image, perspective_matrix, fret_boundaries, config)

    def handle_mouse_click(event, x, y, flags, param):
        """クリックされた位置を弦・フレットに変換する"""

        nonlocal display_image

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        virtual_x, virtual_y = map_image_point_to_virtual_point(
            image_x=x,
            image_y=y,
            perspective_matrix=perspective_matrix,
        )

        result = estimate_string_and_fret(
            virtual_x=virtual_x,
            virtual_y=virtual_y,
            config=config,
            fret_boundaries=fret_boundaries,
        )

        display_image = image.copy()
        draw_calibration_frame(display_image, calibration_points)
        draw_fret_boundaries(display_image, perspective_matrix, fret_boundaries, config)
        draw_estimation_result(display_image, x, y, result)

        if result["inside_fretboard"]:
            print(
                f'推定結果: {result["string_number"]}弦 {result["fret_number"]}フレット '
                f'(virtual_x={result["virtual_x"]:.1f}, virtual_y={result["virtual_y"]:.1f})'
            )
            if args.midi_pitch is not None:
                match_result = match_camera_estimate_with_midi_pitch(
                    estimated_position=result,
                    midi_pitch=args.midi_pitch,
                    start_fret=config.start_fret,
                    visible_frets=config.visible_frets,
                )
                print(format_pitch_match_result(match_result))
        else:
            print(
                f'指板範囲外: virtual_x={result["virtual_x"]:.1f}, '
                f'virtual_y={result["virtual_y"]:.1f}'
            )

        cv2.imshow(WINDOW_NAME, display_image)

    cv2.imshow(WINDOW_NAME, display_image)
    cv2.setMouseCallback(WINDOW_NAME, handle_mouse_click)

    print(f"画像を読み込みました: {image_path}")
    print("指板上をクリックしてください")
    if args.midi_pitch is not None:
        print(f"midi_pitch: {args.midi_pitch}")
    print("q: 終了")

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_position_mapper_with_image()
