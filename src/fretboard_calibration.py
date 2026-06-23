import cv2
import json

# ここを自分の画像ファイル名に変える
IMAGE_PATH = "sample_images/fretboard.jpg"
SAVE_PATH = "calibration_points.json"

points = []

def mouse_callback(event, x, y, flags, param):
    global points, img_display

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            print(f"Point {len(points)}: ({x}, {y})")

        redraw()

def redraw():
    global img_display

    img_display = img.copy()

    # 点を描画
    for i, (x, y) in enumerate(points):
        cv2.circle(img_display, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(
            img_display,
            str(i + 1),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    # 4点そろったら枠を描画
    if len(points) == 4:
        pts = points + [points[0]]
        for i in range(4):
            cv2.line(img_display, pts[i], pts[i + 1], (0, 255, 0), 2)

    cv2.imshow("select fretboard corners", img_display)

img = cv2.imread(IMAGE_PATH)

if img is None:
    raise FileNotFoundError(f"画像が見つかりません: {IMAGE_PATH}")

img_display = img.copy()

cv2.imshow("select fretboard corners", img_display)
cv2.setMouseCallback("select fretboard corners", mouse_callback)

print("指板の4隅をクリックしてください")
print("順番: 左上 → 右上 → 右下 → 左下")
print("s: 保存 / r: リセット / q: 終了")

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        if len(points) == 4:
            data = {
                "order": ["left_top", "right_top", "right_bottom", "left_bottom"],
                "points": points
            }
            with open(SAVE_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"保存しました: {SAVE_PATH}")
        else:
            print("4点そろっていません")

    elif key == ord("r"):
        points = []
        redraw()
        print("リセットしました")

    elif key == ord("q"):
        break

cv2.destroyAllWindows()