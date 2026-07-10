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
* 必要に応じて、内部フレット線をクリックして個別のフレット境界を保存する
* キャリブレーション結果をJSONとして保存する
* 画像上のクリック位置を仮想指板座標へ変換する
* クリック位置から弦番号・フレット番号を推定する

## 現在の制限

現段階では、以下の処理はまだ実装途中または未実装です。

* 指先の自動検出
* ライブカメラ映像での安定動作
* USBマイクによる音高推定の精度向上
* 音名と押弦位置の照合
* フレット線の自動検出
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
    ├── audio_pitch_detector.py
    ├── fretboard_calibration.py
    ├── fretboard_display.py
    ├── fretboard_note_mapper.py
    ├── fretboard_position_mapper.py
    ├── run_audio_pitch.py
    ├── run_fretboard_display.py
    ├── run_sync_gpio_test.py
    ├── run_note_candidates.py
    ├── run_sync_playback.py
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

4隅をクリックしたあと、必要に応じて内部フレット線を左から順にクリックできます。
例えば5〜9フレットの5フレット分を認識する場合、内部の境界線は4本です。

内部フレット線を指定すると、推定時にフレット幅の等分割ではなく、クリックした境界位置を使います。
指定しない場合は、従来どおり表示フレット数で等分割します。

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

内部フレット線まで指定した場合は、JSON内に `virtual_fret_boundaries` が保存されます。

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

### 4. ライブカメラ映像で試す

PCカメラやUSBカメラを使う場合は、以下を実行します。

```bash
# command name: run_position_mapper_with_camera
python src/run_with_camera_pi.py
```

カメラ番号を変える場合:

```bash
python src/run_with_camera_pi.py --camera-index 1
```

操作:

```text
c: カメラ映像上でキャリブレーション開始
s: キャリブレーション保存
f: 反転モードの切り替え
t: 指板枠追跡の切り替え
q: 終了
```

キャリブレーションでは、まず指板の4隅を左上、右上、右下、左下の順にクリックします。
必要に応じて、そのあと内部フレット線を左から順にクリックします。

保存後は、ライブ映像上の指板をクリックすると、何弦・何フレット付近かを推定します。

カメラ映像の左右が逆に見える場合は、起動時に反転を指定できます。

```bash
python src/run_with_camera_pi.py --flip horizontal
```

反転モードは以下から選べます。

```text
none
horizontal
vertical
both
```

実行中に `f` キーを押すと、反転モードを切り替えられます。
反転モードを変えた場合、表示上の座標が変わるため、もう一度キャリブレーションしてください。

カメラ版では、キャリブレーションした指板範囲内の特徴点をライブ映像内で追跡します。
画面左上の `track:on` は追跡中、`track:lost` は追跡に失敗した状態です。
追跡を切り替える場合は `t` キーを押します。

この追跡は小さな位置ずれを補正するための簡易機能です。
手に引っ張られて枠が大きくずれることを避けるため、怪しい動きは追従せず `track:lost` にします。
ギターを大きく動かした場合や、手で指板の特徴点を大きく隠した場合は、`c` キーで再キャリブレーションしてください。

追跡を使わず固定枠で試す場合は、以下のように起動します。

```bash
python src/run_with_camera_pi.py --no-track
```

カメラ版のキャリブレーションは、サンプル画像版とは別に以下へ保存されます。

```text
data/camera_calibration_points.json
```

解像度や表示フレット範囲はオプションで指定できます。

```bash
python src/run_with_camera_pi.py --width 1280 --height 720 --start-fret 5 --visible-frets 5
```

## 設定項目

`src/run_with_image.py` 内の以下の設定で、画像に映っているフレット範囲を指定します。

```python
config = FretboardMappingConfig(
    virtual_width=800,
    virtual_height=240,
    start_fret=5,
    visible_frets=4,
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

例: 5〜8フレットが映っている場合、4個分なので

```python
visible_frets=4
```

## 音高候補の確認

音高から、現在のフレット窓内で候補になる弦・フレットを列挙できます。

```bash
python src/run_note_candidates.py --midi-pitch 62 --start-fret 5 --visible-frets 4
```

例では、5〜8フレット窓の中から MIDI 62、つまり D4 の候補を探します。

現在の窓内の全音を表示する場合:

```bash
python src/run_note_candidates.py --start-fret 5 --visible-frets 4 --show-window
```

カメラ推定位置と音高候補を照合する場合:

```bash
python src/run_note_candidates.py --midi-pitch 64 --start-fret 5 --visible-frets 5 --estimated-string 2 --estimated-fret 6
```

サンプル画像上のクリック位置と仮のMIDI番号を照合する場合:

```bash
python src/run_with_image.py --midi-pitch 62 --start-fret 5 --visible-frets 4
```

ライブカメラ上のクリック位置と仮のMIDI番号を照合する場合:

```bash
python src/run_with_camera_pi.py --flip horizontal --midi-pitch 62 --start-fret 5 --visible-frets 4
```

クリック後、カメラ画面左上とターミナルに音高照合結果が表示されます。
`--midi-pitch` を使う場合はUSBマイクや外部MIDI入力を使わず、指定した仮MIDI番号を固定値として使います。
実行中は `[` / `]` キーで、仮MIDI番号を半音ずつ下げたり上げたりできます。

## 音声入力から音高を推定する

## 指板図だけを表示する

カメラやマイクを使わず、画面に映す指板図だけを確認できます。
指板図は12フレット全体を表示し、4フレットずつの3ブロックに分けて表示します。
現在対象になっているブロックは黄色い枠で強調されます。

```bash
python src/run_fretboard_display.py --start-fret 5 --visible-frets 4 --total-frets 12 --block-size 4
```

表示された指板をクリックすると、その弦・フレットが黄色く光ります。
`a` / `d` キーで現在ブロックを4フレットずつ左右に動かせます。

## 音声入力から音高を推定する

Windowsデモ環境でもRaspberry Pi本番環境でも、音声入力は `sounddevice` 経由で扱います。
まず入力デバイスを確認します。

```bash
python src/run_audio_pitch.py --list-devices
```

マイク単体で音高推定を確認する場合:

```bash
python src/run_audio_pitch.py
```

特定のUSBマイクを指定する場合:

```bash
python src/run_audio_pitch.py --device 1
```

カメラ推定とマイク音高推定を同時に使う場合:

```bash
python src/run_with_camera_pi.py --flip horizontal --audio-pitch --start-fret 5 --visible-frets 4
```

通常のマッピング中は、カメラ映像ではなく作成した仮想指板を表示します。
仮想指板は12フレット全体を表示し、4フレットずつの3ブロック境界と現在ブロックを示します。
押弦位置として推定された弦・フレット、または音高照合で選ばれた弦・フレットが黄色く光ります。
キャリブレーション中は4隅をクリックする必要があるため、カメラ映像を表示します。

カメラ映像を見ながら確認したい場合:

```bash
python src/run_with_camera_pi.py --display-mode camera
```

実行中は `v` キーで、仮想指板表示とカメラ表示を切り替えられます。
仮想指板表示中に指板をクリックすると、押弦位置の表示テストもできます。

## ギター同期のON/OFFと保存

本番のRaspberry Piでは、タクトスイッチでギター同期をON/OFFし、LEDで状態を確認する想定です。
Windowsデモ環境では `o` キーで同期をON/OFFできます。

同期ON中は、音声入力と画面上で光った指板位置を保存します。

保存先:

```text
data/sync_sessions/sync_YYYYMMDD_HHMMSS/
├── audio.wav
├── video.mp4
├── events.json
├── frame_times.json
└── meta.json
```

実行例:

```bash
python src/run_with_camera_pi.py --gpio-button-pin 17 --gpio-led-pin 27
```

GPIO制御にはRaspberry Pi側で `gpiozero` が必要です。未導入の場合はGPIOだけ無効化され、Windowsデモと同じく `o` キーで操作できます。

カメラやマイクなしで、タクトスイッチとLEDだけを先に確認する場合:

```bash
python src/run_sync_gpio_test.py --gpio-button-pin 17 --gpio-led-pin 27
```

タクトスイッチを押すたびに記録ON/OFFが切り替わり、記録中はLEDが点灯します。

指板ハイライト履歴の保存まで確認したい場合:

```bash
python src/run_sync_gpio_test.py --gpio-button-pin 17 --gpio-led-pin 27 --demo-events
```

`--demo-events` では、記録ON中にダミーの弦・フレット履歴が `events.json` に保存されます。

起動直後から同期ONにする場合:

```bash
python src/run_with_camera_pi.py --sync-start-on
```

画面イベントだけ保存して、音声録音をしない場合:

```bash
python src/run_with_camera_pi.py --sync-no-audio
```

GPIOを使わずに試す場合:

```bash
python src/run_with_camera_pi.py --no-gpio
```

保存した同期セッションを後から確認する場合:

```bash
python src/run_sync_playback.py --latest
```

特定のセッションを開く場合:

```bash
python src/run_sync_playback.py data/sync_sessions/sync_YYYYMMDD_HHMMSS
```

音声を鳴らさず、指板の光り方だけ確認する場合:

```bash
python src/run_sync_playback.py --latest --no-audio
```

再生画面では `space` で再生/一時停止、`r` で最初から再生、`q` で終了できます。

USBマイクの入力レベルが小さい場合は、しきい値を下げます。

```bash
python src/run_with_camera_pi.py --audio-pitch --audio-rms-threshold 0.008
```

現段階の音高推定は単音向けです。
コードや複数音、強いノイズがある環境では誤検出する可能性があります。

## 次のCodexへの引き継ぎ

現在の設計方針は、指先検出を使わず、色付きシールで現在ブロックを判定し、USBマイクの音高と理論表で弦・フレットを一意に決める方式です。

対象範囲は20フレットではなく、12フレットに変更しました。

```text
1-4F   : 青シール
5-8F   : 黄シール
9-12F  : 白シール
```

各ブロックの4隅に同じ色のシールを貼っています。
カメラ側では、この4隅の色付きシールを検出して、現在弾いているブロックを以下の3つから決めます。

```text
blue   -> 1-4F
yellow -> 5-8F
pink   -> 9-12F
```

未実装の主要タスク:

1. 色付きシール検出モジュールを追加する
   - 推奨ファイル名: `src/fretboard_marker_detector.py`
   - OpenCVでHSV変換し、青・黄・白の色範囲を検出する
   - 検出結果からブロック名、開始フレット、信頼度を返す

2. カメラ映像から現在ブロックを推定する
   - 返り値例:
     ```python
     {
         "block_name": "5-8F",
         "start_fret": 5,
         "visible_frets": 4,
         "marker_color": "yellow",
         "confidence": 0.82,
     }
     ```
   - 4隅すべてが見えなくても、同じ色のシールが一定数見えたらブロック候補として扱う

3. ブロック推定結果を `run_with_camera_pi.py` に接続する
   - `state.config.start_fret` を検出した `start_fret` に更新する
   - 画面の12F指板上で現在ブロックを黄色枠で強調する
   - `events.json` には検出ブロックも保存する

4. 音高との照合は既存の `fretboard_note_mapper.py` を使う
   - ブロックが `5-8F`、音高が `D4` なら、その4F範囲内の候補を列挙する
   - 候補が1つなら確定
   - 候補が複数なら、現段階では候補一覧を保存/表示する

5. 開放弦は別扱いにする
   - 12Fブロックには含めない
   - 将来、ナット付近にOPEN用シールを貼る場合は `start_fret=0, visible_frets=1` の特別ブロックとして扱う

後回しにするもの:

```text
指先検出
音高推定精度の本格改善
統合ルールの高度化
管理画面
エラー表示の磨き込み
```

次に実装するなら、まずは `src/fretboard_marker_detector.py` を作り、保存済みカメラ画像またはライブカメラ1フレームから青・黄・白シールを検出するところから始めてください。

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
4. フレット境界が保存されていればそれを使い、なければ仮想指板を横に表示フレット数ぶん等分割する
5. クリック位置がどの区画に入るかを調べる
6. 弦番号・フレット番号として表示する
```

現在の実装では、クリック位置を仮の指先位置として扱っています。
将来的には、このクリック座標を画像処理で検出した指先座標に置き換えることで、実際の押弦位置推定へ発展させます。

## 今後の実装予定

* PC内蔵カメラでのライブ映像対応
* Raspberry Piカメラでのライブ映像対応
* 指先位置の自動検出
* フレット線の自動検出
* USBマイクによる音高推定の精度向上
* カメラ候補と音名候補の照合
* FretTone本体との連携
* 画面上の指板へのリアルタイムハイライト表示
* LEDや7セグメント表示器によるGPIO出力

## 授業課題としての位置づけ

本作品は、Raspberry Piカメラを用いたギター練習支援ツールのプロトタイプです。

単に画像を表示するだけでなく、カメラ画像上の指板領域をキャリブレーションし、画像座標を弦番号・フレット番号へ変換することで、実物の楽器と画面上の指板表示を対応づけることを目的としています。

タブ譜は弦番号とフレット番号を記号として示す点では分かりやすい一方で、実際の指板上の位置感覚や音名・度数との対応を直感的に理解するには限界があります。

このシステムは、実物の指板、画面上の指板図、音楽理論情報を接続するための基礎技術として設計しています。
