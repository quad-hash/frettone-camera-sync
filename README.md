# FretTone Camera Sync

Raspberry PiカメラやPCカメラを用いてギター指板を撮影し、画像上の座標を弦番号・フレット番号へ変換するためのプロトタイプです。

現段階では、指先の自動検出までは行わず、画像上でクリックした位置を「仮の押弦位置」として扱います。
クリック座標をもとに、指板上の何弦・何フレット付近かを推定します。

## 目的

ギター学習では、タブ譜や画面上の指板図を見ても、それが実際のギター上のどこに対応するのかを直感的に理解しにくい場合があります。

このプロジェクトでは、実際のギター指板をカメラで撮影し、画像座標を弦・フレットという音楽的な座標に変換することで、実物の指板と画面上の指板表示を同期するための基礎システムを作成します。

将来的には、以下のような機能への発展を想定しています。

* カメラ画像から指先位置を検出する
* USBマイクで鳴った音名を推定する
* カメラによる位置候補と音名推定を照合する
* FretToneの画面上の指板と実ギターの押弦位置を同期する
* 画面に表示された音・度数・コードトーンを実際に弾けているか判定する

## 現在できること

* サンプル指板画像を読み込む
* 指板の4隅をクリックしてキャリブレーションする
* キャリブレーション結果をJSONとして保存する
* 画像上のクリック位置を仮想指板座標へ変換する
* クリック位置から弦番号・フレット番号を推定する

## 現在の制限

現段階では、以下の処理はまだ実装途中または未実装です。

* 指先の自動検出
* ライブカメラ映像での安定動作
* USBマイクによる音名推定
* 音名と押弦位置の照合
* フレット幅の厳密な補正
* 複数音やコード演奏への対応

現在の実装は、指先検出の前段階として、画像上の座標を弦・フレット番号に変換するための土台です。

## ディレクトリ構成

```text
frettone-camera-sync/
├── README.md
├── requirements.txt
├── data/
│   └── calibration_points.json
├── docs/
│   └── report_notes.md
├── sample_images/
│   └── fretboard_sample.png
└── src/
    ├── fretboard_calibration.py
    ├── fretboard_position_mapper.py
    ├── run_with_image.py
    └── run_with_camera_pi.py
```

## 使用技術

* Python
* OpenCV
* NumPy
* Raspberry Pi
* Raspberry Pi Camera / PC Webcam

## セットアップ

Pythonの仮想環境を使う場合は、以下のように作成します。

```bash
# command name: create_virtual_environment
python -m venv .venv
```

Windows PowerShellの場合:

```powershell
# command name: activate_virtual_environment_windows
.\.venv\Scripts\Activate.ps1
```

macOS / Linux / Raspberry Piの場合:

```bash
# command name: activate_virtual_environment_unix
source .venv/bin/activate
```

必要なライブラリをインストールします。

```bash
# command name: install_dependencies
python -m pip install -r requirements.txt
```

最小構成では、`requirements.txt` は以下で十分です。

```text
opencv-python
numpy
```

## 使い方

### 1. サンプル画像を配置する

指板画像を以下の場所に置きます。

```text
sample_images/fretboard_sample.png
```

JPEGを使う場合は、以下のような名前でも読み込めます。

```text
sample_images/fretboard_sample.jpg
sample_images/fretboard_sample.jpeg
```

### 2. 指板の4点キャリブレーションを行う

以下を実行します。

```bash
# command name: run_fretboard_calibration
python src/fretboard_calibration.py
```

画像が表示されたら、認識したい指板範囲の4隅を以下の順番でクリックします。

```text
1. 左上
2. 右上
3. 右下
4. 左下
```

クリックするのは、フレットの内側ではなく、認識したいフレット範囲全体の外枠です。

例えば、5〜9フレットを認識したい場合は、5〜9フレットの領域全体を囲むように4点を指定します。

操作:

```text
s: 保存
r: リセット
q: 終了
```

保存後、以下のファイルが作成されます。

```text
data/calibration_points.json
```

### 3. クリック位置から弦・フレットを推定する

以下を実行します。

```bash
# command name: run_position_mapper_with_image
python src/run_with_image.py
```

画像上の指板をクリックすると、ターミナルに推定結果が表示されます。

例:

```text
推定結果: 3弦 7フレット (virtual_x=392.4, virtual_y=132.8)
```

## 設定項目

`src/run_with_image.py` 内の以下の設定で、画像に映っているフレット範囲を指定します。

```python
config = FretboardMappingConfig(
    virtual_width=800,
    virtual_height=240,
    start_fret=5,
    visible_frets=5,
    string_order_top_to_bottom=(6, 5, 4, 3, 2, 1),
)
```

### start_fret

画像に映っている最初のフレット番号です。

例: 5〜9フレットが映っている場合

```python
start_fret=5
```

### visible_frets

画像に映っているフレット数です。

例: 5〜9フレットが映っている場合、5個分なので

```python
visible_frets=5
```

### string_order_top_to_bottom

画像上で上から見える弦の順番です。

通常、構えたギターを上から撮影した場合は以下のようになります。

```python
string_order_top_to_bottom=(6, 5, 4, 3, 2, 1)
```

上下が逆に表示される場合は、以下のように変更します。

```python
string_order_top_to_bottom=(1, 2, 3, 4, 5, 6)
```

## 仕組み

このプロジェクトでは、以下の流れで座標変換を行います。

```text
1. 指板の4隅をクリックする
2. 斜めに写った指板領域を、仮想的な長方形の指板に変換する
3. 仮想指板を縦に6分割する
4. 仮想指板を横に表示フレット数ぶん分割する
5. クリック位置がどの区画に入るかを調べる
6. 弦番号・フレット番号として表示する
```

現在の実装では、クリック位置を仮の指先位置として扱っています。
将来的には、このクリック座標を画像処理で検出した指先座標に置き換えることで、実際の押弦位置推定へ発展させます。

## 今後の実装予定

* PC内蔵カメラでのライブ映像対応
* Raspberry Piカメラでのライブ映像対応
* 指先位置の自動検出
* フレット線の個別指定または自動検出
* USBマイクによる音名推定
* カメラ候補と音名候補の照合
* FretTone本体との連携
* 画面上の指板へのリアルタイムハイライト表示
* LEDや7セグメント表示器によるGPIO出力

## 授業課題としての位置づけ

本作品は、Raspberry Piカメラを用いたギター練習支援ツールのプロトタイプです。

単に画像を表示するだけでなく、カメラ画像上の指板領域をキャリブレーションし、画像座標を弦番号・フレット番号へ変換することで、実物の楽器と画面上の指板表示を対応づけることを目的としています。

タブ譜は弦番号とフレット番号を記号として示す点では分かりやすい一方で、実際の指板上の位置感覚や音名・度数との対応を直感的に理解するには限界があります。

このシステムは、実物の指板、画面上の指板図、音楽理論情報を接続するための基礎技術として設計しています。
