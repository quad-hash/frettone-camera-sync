# FretLog Quickstart

FretLog is currently organized around an offline workflow:

1. Record a session.
2. Estimate audio/video sync.
3. Analyze audio/video into `notes.json`.
4. Inspect low-confidence notes.
5. Replay the fretboard with audio.

## One-Command Take

Record, analyze, write the report, and open playback:

```powershell
python src/run_fretlog_take.py
```

The recorder opens first. Press `o` to start/stop recording and `q` to close
the recorder. Press `c` to cancel a bad take and skip analysis. After a normal
finish, analysis and playback run automatically.

Useful variants:

```powershell
python src/run_fretlog_take.py --hand-viewer
python src/run_fretlog_take.py --skip-record --no-playback
python src/run_fretlog_take.py --no-mediapipe-hands
```

## Record

For clean offline capture, use the lightweight recorder. Toggle recording with `o`
or the GPIO button.

```powershell
python src/run_fretlog_recorder.py --flip horizontal
```

The older live camera app can still record too:

```powershell
python src/run_with_camera_pi.py --marker-detect --nut-anchor-detect --flip horizontal --audio-pitch
```

A session directory is written under:

```text
data/sync_sessions/sync_YYYYMMDD_HHMMSS/
```

Expected files:

```text
audio.wav
video.mp4
events.json
frame_times.json
meta.json
```

At the start of a session, clap once in view of the camera. This is used to estimate AV offset.

## Run The Pipeline

For the latest recorded session:

```powershell
python src/run_fretlog_pipeline.py --latest
```

Optional hand landmark extraction requires MediaPipe:

```powershell
pip install mediapipe
python src/run_mediapipe_hands.py --latest
python src/run_fretlog_pipeline.py --latest --mediapipe-hands
```

This writes `hand_tracks.json` next to `notes.json`. When `hand_tracks.json`
exists, `run_offline_analyze.py` uses the hand position as extra evidence for
ambiguous pitch candidates.

Preview the hand tracking result over the camera video:

```powershell
python src/run_hand_tracks_viewer.py --latest
```

To write a review video without opening a preview window:

```powershell
python src/run_hand_tracks_viewer.py --latest --no-preview --output-video hand_tracks_overlay.mp4
```

This runs:

```powershell
python src/estimate_av_sync.py --latest --update-meta
python src/run_offline_analyze.py --latest
python src/inspect_notes.py --latest
python src/write_analysis_report.py --latest
```

Outputs:

```text
av_sync.json
notes.json
analysis_report.md
```

## Replay

```powershell
python src/run_sync_playback.py --latest
```

If `notes.json` exists, playback uses the offline note log instead of live `events.json`.

## UI Preview Only

Render static PNG previews without opening the camera, microphone, or a GUI window:

```powershell
python src/make_ui_previews.py
```

The images are written to `data/ui_previews/`.

## Inspect A Note

List low-confidence notes:

```powershell
python src/inspect_notes.py --latest
```

Show all candidates for one note:

```powershell
python src/inspect_notes.py --latest --note-index 2
```

## Demo Without Recording Hardware

Create a synthetic audio session and run the full pipeline:

```powershell
python src/run_fretlog_pipeline.py --make-demo --no-video-analysis
```

This is useful for checking that the offline analysis tools still work when no camera or microphone is available.

## Analysis Config

Copy and edit:

```powershell
copy analysis_config.example.json analysis_config.local.json
python src/run_fretlog_pipeline.py --latest --config analysis_config.local.json
```

Useful fields:

- `onset_threshold`: higher means fewer detected notes.
- `rms_threshold`: lower means quieter notes are accepted.
- `video_marker_min_area`: lower means marker detection is more sensitive.
- `visual_weight`: higher means video evidence affects candidate choice more.
