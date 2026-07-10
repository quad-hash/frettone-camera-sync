# Raspberry Pi Capture Setup

This package turns the Raspberry Pi into a capture device.

The Pi records:

- `audio.wav`
- `video.mp4`
- `frame_times.json`
- `events.json`
- `meta.json`

Then the session folder can be transferred to the Windows PC and analyzed there.

## Recommended Hardware

- Raspberry Pi 5 or Pi 4, 64-bit Raspberry Pi OS
- USB camera
- USB microphone
- HDMI display for the recorder preview
- GPIO button on GPIO17 for record/stop
- LED on GPIO27 for recording state

Optional:

- second button for future cancel wiring
- completion/error LEDs

## Install On Raspberry Pi

Use Raspberry Pi OS 64-bit.

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-opencv portaudio19-dev libportaudio2 ffmpeg openssh-client

python3 -m venv .venv --system-site-packages
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-pi.txt
```

`python3-opencv` is installed through apt so that OpenCV matches Raspberry Pi OS.

## Wiring

Record button:

```text
GPIO17 --- button --- GND
```

Recording LED:

```text
GPIO27 --- 330 ohm resistor --- LED --- GND
```

The button uses gpiozero's internal pull-up.

## Check Devices

Audio devices:

```bash
python src/run_audio_pitch.py --list-devices
```

Camera preview/recording:

```bash
python src/run_fretlog_recorder.py --flip horizontal
```

Controls:

```text
o: record/stop
c: cancel
q: finish
```

## Capture And Transfer With SCP

Replace the destination with your Windows PC SSH destination.

```bash
python src/run_fretlog_pi_capture.py \
  --flip horizontal \
  --transfer-to quad1@WINDOWS-PC:/c/Users/quad1/frettone-camera-sync/data/sync_sessions/
```

If you only want to capture and copy manually:

```bash
python src/run_fretlog_pi_capture.py --flip horizontal
```

Transfer the latest session later:

```bash
python src/transfer_latest_session.py quad1@WINDOWS-PC:/c/Users/quad1/frettone-camera-sync/data/sync_sessions/
```

## Analyze On Windows PC

After transfer, run this on the Windows PC:

```powershell
python src/run_fretlog_pipeline.py data\sync_sessions\sync_YYYYMMDD_HHMMSS --mediapipe-hands
python src/run_sync_playback.py data\sync_sessions\sync_YYYYMMDD_HHMMSS
```
