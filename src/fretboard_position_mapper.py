# filename: src/fretboard_position_mapper.py
# role: 画像上の座標を、ギターの弦番号・フレット番号に変換する処理

from dataclasses import dataclass
from pathlib import Path
import json

import cv2
import numpy as np


@dataclass
class FretboardMappingConfig:
    """指板位置推定の設定"""

    virtual_width: int = 800
    virtual_height: int = 240

    # カメラに映している最初のフレット番号
    # 例：5〜8フレットを映しているなら start_fret=5, visible_frets=4
    start_fret: int = 5
    visible_frets: int = 4

    # 画像上で上から見える弦の順番
    # 普通にギターを構えた状態を上から見るなら、多くの場合は上から 6,5,4,3,2,1
    string_order_top_to_bottom: tuple[int, int, int, int, int, int] = (6, 5, 4, 3, 2, 1)


@dataclass
class CalibrationData:
    """キャリブレーションJSONから読み込んだ情報"""

    points: np.ndarray
    fret_boundaries: tuple[float, ...] | None = None


def load_calibration_data(calibration_json_path: str | Path) -> CalibrationData:
    """キャリブレーションJSONから4点座標を読み込む"""

    calibration_json_path = Path(calibration_json_path)

    with calibration_json_path.open("r", encoding="utf-8") as file:
        calibration_data = json.load(file)

    points = np.array(calibration_data["points"], dtype=np.float32)

    if points.shape != (4, 2):
        raise ValueError("キャリブレーション点は4点である必要があります")

    raw_fret_boundaries = calibration_data.get("virtual_fret_boundaries")
    fret_boundaries = None

    if raw_fret_boundaries is not None:
        fret_boundaries = tuple(float(value) for value in raw_fret_boundaries)

        if len(fret_boundaries) < 2:
            raise ValueError("フレット境界は2点以上である必要があります")

        for left, right in zip(fret_boundaries, fret_boundaries[1:]):
            if left >= right:
                raise ValueError("フレット境界は左から右へ昇順である必要があります")

    return CalibrationData(points=points, fret_boundaries=fret_boundaries)


def load_calibration_points(calibration_json_path: str | Path) -> np.ndarray:
    """キャリブレーションJSONから4点座標だけを読み込む"""

    return load_calibration_data(calibration_json_path).points


def build_perspective_matrix(
    image_points: np.ndarray,
    config: FretboardMappingConfig,
) -> np.ndarray:
    """画像上の斜め指板を、仮想的な長方形指板に変換する行列を作る"""

    virtual_points = np.array(
        [
            [0, 0],
            [config.virtual_width, 0],
            [config.virtual_width, config.virtual_height],
            [0, config.virtual_height],
        ],
        dtype=np.float32,
    )

    return cv2.getPerspectiveTransform(image_points, virtual_points)


def map_image_point_to_virtual_point(
    image_x: int,
    image_y: int,
    perspective_matrix: np.ndarray,
) -> tuple[float, float]:
    """画像上のクリック座標を、仮想指板上の座標に変換する"""

    image_point = np.array([[[image_x, image_y]]], dtype=np.float32)
    virtual_point = cv2.perspectiveTransform(image_point, perspective_matrix)

    virtual_x = float(virtual_point[0][0][0])
    virtual_y = float(virtual_point[0][0][1])

    return virtual_x, virtual_y


def estimate_string_and_fret(
    virtual_x: float,
    virtual_y: float,
    config: FretboardMappingConfig,
    fret_boundaries: tuple[float, ...] | list[float] | None = None,
) -> dict:
    """仮想指板座標から、弦番号とフレット番号を推定する"""

    is_inside = (
        0 <= virtual_x <= config.virtual_width
        and 0 <= virtual_y <= config.virtual_height
    )

    if not is_inside:
        return {
            "inside_fretboard": False,
            "string_number": None,
            "fret_number": None,
            "virtual_x": virtual_x,
            "virtual_y": virtual_y,
        }

    string_area_height = config.virtual_height / 6

    string_index = int(virtual_y // string_area_height)

    string_index = min(max(string_index, 0), 5)

    if fret_boundaries is None:
        fret_area_width = config.virtual_width / config.visible_frets
        fret_index = int(virtual_x // fret_area_width)
        fret_index = min(max(fret_index, 0), config.visible_frets - 1)
        fret_estimation_method = "equal_split"
    else:
        if len(fret_boundaries) != config.visible_frets + 1:
            raise ValueError(
                "フレット境界の数は visible_frets + 1 と一致している必要があります"
            )

        fret_index = config.visible_frets - 1

        for index in range(config.visible_frets):
            left_boundary = fret_boundaries[index]
            right_boundary = fret_boundaries[index + 1]

            if left_boundary <= virtual_x < right_boundary:
                fret_index = index
                break

        fret_index = min(max(fret_index, 0), config.visible_frets - 1)
        fret_estimation_method = "calibrated_boundaries"

    string_number = config.string_order_top_to_bottom[string_index]
    fret_number = config.start_fret + fret_index

    return {
        "inside_fretboard": True,
        "string_number": string_number,
        "fret_number": fret_number,
        "virtual_x": virtual_x,
        "virtual_y": virtual_y,
        "fret_estimation_method": fret_estimation_method,
    }
