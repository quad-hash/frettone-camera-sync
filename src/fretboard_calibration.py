import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ここを自分の画像ファイル名に変える
IMAGE_PATH = PROJECT_ROOT / "sample_images" / "fretboard_sample.png"
SAVE_PATH = PROJECT_ROOT / "data" / "calibration_points.json"
WINDOW_NAME = "select fretboard calibration points"

VIRTUAL_WIDTH = 800
VIRTUAL_HEIGHT = 240
START_FRET = 5
VISIBLE_FRETS = 4
INTERNAL_FRET_LINE_COUNT = VISIBLE_FRETS - 1

corner_points = []
internal_fret_points = []


def build_perspective_matrix():
    image_points = np.array(corner_points, dtype=np.float32)
    virtual_points = np.array(
        [
            [0, 0],
            [VIRTUAL_WIDTH, 0],
            [VIRTUAL_WIDTH, VIRTUAL_HEIGHT],
            [0, VIRTUAL_HEIGHT],
        ],
        dtype=np.float32,
    )

    return cv2.getPerspectiveTransform(image_points, virtual_points)


def build_virtual_fret_boundaries():
    if len(internal_fret_points) != INTERNAL_FRET_LINE_COUNT:
        return None

    perspective_matrix = build_perspective_matrix()
    image_points = np.array(
        [[[x, y]] for x, y in internal_fret_points],
        dtype=np.float32,
    )
    virtual_points = cv2.perspectiveTransform(image_points, perspective_matrix)
    internal_boundaries = sorted(float(point[0][0]) for point in virtual_points)

    return [0.0, *internal_boundaries, float(VIRTUAL_WIDTH)]


def mouse_callback(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if len(corner_points) < 4:
        corner_points.append((x, y))
        print(f"Corner {len(corner_points)}: ({x}, {y})")

        if len(corner_points) == 4:
            print(
                "必要なら、内部フレット線を左から順に "
                f"{INTERNAL_FRET_LINE_COUNT} 本クリックしてください"
            )
            print("内部フレット線を指定しない場合は、このまま s で保存できます")
    elif len(internal_fret_points) < INTERNAL_FRET_LINE_COUNT:
        internal_fret_points.append((x, y))
        print(f"Internal fret line {len(internal_fret_points)}: ({x}, {y})")
    else:
        print("必要な点はすべて選択済みです。s で保存、r でリセットできます")

    redraw()


def redraw():
    img_display = img.copy()

    for index, (x, y) in enumerate(corner_points):
        cv2.circle(img_display, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(
            img_display,
            str(index + 1),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    if len(corner_points) == 4:
        closed_points = corner_points + [corner_points[0]]

        for index in range(4):
            cv2.line(
                img_display,
                closed_points[index],
                closed_points[index + 1],
                (0, 255, 0),
                2,
            )

    for index, (x, y) in enumerate(internal_fret_points):
        cv2.circle(img_display, (x, y), 6, (0, 255, 255), -1)
        cv2.putText(
            img_display,
            f"F{index + 1}",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    cv2.imshow(WINDOW_NAME, img_display)


def save_calibration():
    if len(corner_points) != 4:
        print("4隅がそろっていません")
        return

    if 0 < len(internal_fret_points) < INTERNAL_FRET_LINE_COUNT:
        print(
            "内部フレット線を使う場合は、"
            f"{INTERNAL_FRET_LINE_COUNT} 本すべてクリックしてください"
        )
        return

    data = {
        "order": ["left_top", "right_top", "right_bottom", "left_bottom"],
        "points": corner_points,
        "start_fret": START_FRET,
        "visible_frets": VISIBLE_FRETS,
        "virtual_size": {
            "width": VIRTUAL_WIDTH,
            "height": VIRTUAL_HEIGHT,
        },
    }

    virtual_fret_boundaries = build_virtual_fret_boundaries()

    if virtual_fret_boundaries is not None:
        data["internal_fret_line_points"] = internal_fret_points
        data["virtual_fret_boundaries"] = virtual_fret_boundaries

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with SAVE_PATH.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

    print(f"保存しました: {SAVE_PATH}")

    if virtual_fret_boundaries is None:
        print("フレット境界は未指定です。推定時は等分割を使います")
    else:
        print("フレット境界も保存しました。推定時は個別境界を使います")


def reset_calibration():
    corner_points.clear()
    internal_fret_points.clear()
    redraw()
    print("リセットしました")


img = cv2.imread(str(IMAGE_PATH))

if img is None:
    raise FileNotFoundError(f"画像が見つかりません: {IMAGE_PATH}")

cv2.imshow(WINDOW_NAME, img)
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

print("指板の4隅をクリックしてください")
print("順番: 左上 -> 右上 -> 右下 -> 左下")
print("4隅のあと、必要なら内部フレット線を左から順にクリックできます")
print("s: 保存 / r: リセット / q: 終了")

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        save_calibration()
    elif key == ord("r"):
        reset_calibration()
    elif key == ord("q"):
        break

cv2.destroyAllWindows()
