# ThinkCam

Real-time visualization and capture tool for the LUCID TRT009S-E event camera (EVS). Built with PySide6 and OpenCV. Streams the raw asynchronous event stream (XYTPFrame) and renders it at a fixed display rate, so raw recording stays lossless regardless of what the GUI can paint.

## Features

- Live event view (events rendered as polarity on mid-gray)
- **Save PNG** snapshot and **Record Video** to MP4
- **Record RAW** — the lossless event stream to `recordings/<take>/events.h5`
- **Capture dashboard** — live events-per-lit-pixel, lit coverage and camera
  speed at 2 Hz, plus the last full probe's gate table
- **Bias controls** — sensor thresholds and refractory period, adjustable while
  streaming; the values in effect are written into each take's `metadata.json`
- Status bar with event rate, GVSP FPS, link bandwidth, render time, and frame counter

## The capture loop

Shoot, probe, adjust, shoot again — `plans/01_CaptureDashboard.md`.

Record a take with **R**, then press **D** (or *Run Probe*) to score it:
`capture_probe.py` accumulates motion-compensated frames, LK-tracks them into
COLMAP correspondences, runs the mapper and prints a gate table, writing
`work/probe/<take>/probe.json` for the panel to render. It runs out of process
on a separate analysis interpreter (`THINKCAM_PROBE_PYTHON`, default
`~/envs/phase1/bin/python`) because the analysis tools need numpy/h5py/cv2 and
the `colmap` binary rather than `arena_api` + PySide6. A 200-frame probe is
~3 minutes; `/usr/bin/colmap` is built without CUDA, so the mapper is CPU-only.

The same thing from a terminal, with no camera attached:

```bash
~/envs/phase1/bin/python capture_probe.py --input recordings/<take>
~/envs/phase1/bin/python frame_metrics.py fragments --scene work/probe/<take>
PYTHONPATH=. ~/envs/phase1/bin/python test_probe_stats.py
```

The headline gate is **fragments**: the LK path clears the mean-track-length
bar and the mapper still returns a dozen disconnected models, so what the table
foregrounds is how far one reconstruction reaches before it breaks.

## Requirements

- Python 3.10+
- LUCID ArenaSDK for Linux x64 ([download](https://thinklucid.com/downloads-hub/))
- `arena_api` Python wheel (separate download from the same downloads hub)

### Python packages

```bash
pip install -r requirements.txt
pip install arena_api-*.whl
```

## Setup

1. **ArenaSDK**: Download and extract ArenaSDK for Linux x64. Set the `ARENA_SDK` environment variable to the extracted `ArenaSDK_Linux_x64` directory, or edit the default in `run_evs.sh`.

2. **arena_api config**: Point the Python wrapper at your SDK's native libraries by editing `arena_api_config.py` in your site-packages:

   ```python
   ARENAC_CUSTOM_PATHS = {
       ...
       'python64_lin': '/path/to/ArenaSDK_Linux_x64/lib64/libarenac.so'
   }
   SAVEC_CUSTOM_PATHS = {
       ...
       'python64_lin': '/path/to/ArenaSDK_Linux_x64/lib64/libsavec.so'
   }
   ```

3. **Network**: The camera uses link-local addressing. Assign an IP on the same subnet to your Ethernet interface:

   ```bash
   sudo ip addr add 169.254.80.1/16 dev <interface>
   ```

   Default camera IP is `169.254.80.199` (configurable in `thinkcam/constants.py`).

## Usage

```bash
./run_evs.sh
```

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| S | Save PNG snapshot |
| P | Show derivative plots |
| R | Start/stop RAW event recording |
| D | Run the capture probe on the last finished take |
| Q / Esc | Quit |

## Project structure

```
ThinkCam/
  thinkcam/
    main.py            # Application entry point
    main_window.py     # Main window layout and signal wiring
    camera_worker.py   # QThread for camera acquisition + bias mailbox
    visualizer.py      # Event batch -> BGR
    controls.py        # Save / Record / Biases sidebar
    dashboard.py       # Live readouts + last probe's gate table
    probe_worker.py    # QThread for the live statistics (ev/lit-px, speed)
    status_bar.py      # Live statistics status bar
    recorder.py        # MP4 video recording
    raw_recorder.py    # Lossless HDF5 event recording
    derivative_plot.py # Rolling polarity plots
    constants.py       # Camera and probe defaults
  capture_probe.py     # Score a finished take: frames -> tracks -> COLMAP -> gates
  accumulate_frames.py # Events -> deterministic intensity frames
  track_frames.py      # LK tracks -> COLMAP correspondences
  frame_metrics.py     # Acceptance gates, incl. fragment analysis
  view_sparse.py       # Render a COLMAP sparse model to a PNG
  test_probe_stats.py  # Offline check of the live probe tier
  run_evs.sh           # Launcher script
  requirements.txt
```

## License

MIT
