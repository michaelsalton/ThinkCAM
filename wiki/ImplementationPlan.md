# ThinkCam Raw-Event Recording — Revised Implementation Plan

This plan supersedes the integration assumptions in [`DataCollect.txt`](DataCollect.txt)
after reading the live ThinkCam source and the LUCID ArenaSDK examples. The
**deliverables, HDF5 schema, metadata sidecar, converters, and operator protocol**
from that spec stand. What changes is the **acquisition contract** — the spec's central
assumption (tap a `dv.EventStore` before rendering) does not hold for this codebase.

---

## 0. What the recon changed

| `DataCollect.txt` assumption | Verified reality | Source |
|---|---|---|
| Events available as `dv.EventStore` in the acquisition loop | No `dv.EventStore` exists. dv-processing is not imported and not in `requirements.txt`. | `camera_worker.py`, `requirements.txt` |
| Camera streams raw events; ThinkCam renders them | Camera runs in **CDFrame** mode — events are rasterized to grayscale frames **on the camera**. ThinkCam only ever receives pictures. | `camera_worker.py:65` |
| `pos_count`/`neg_count` are event counts | They are **pixel** counts (`mono>200`, `mono<50`) from the rendered frame — an estimate, not real events. | `visualizer.py:34-35` |
| AEDAT4 (dv-processing) is the primary format | dv-processing is absent. HDF5 from the raw arrays is the path of least resistance. | `requirements.txt` |
| Biases pulled from live UI state | Biases are **hardcoded constants**; no live bias UI exists. | `constants.py:8-11`, `controls.py` |
| Visualizer has 5 modes | One mode (CDFrame → BGR). | `visualizer.py` |
| Confirm resolution | Already read live from nodemap and plumbed via `connected` signal. | `camera_worker.py:111-114` |

**The real raw-event tap point** (from SDK example
[`py_evs_xytp_frame_heatmap.py`](../ArenaSDK/ArenaSDK_Linux_x64/Examples/Python/py_evs_xytp_frame_heatmap.py)):
switch the EVS output format from `CDFrame` to **`XYTPFrame`**. Each `get_buffer()` then
returns pixel format `LUCID_LucidXYTP128f` — 4× `float32` per event = `(x, y, t, p)`,
containing only pixels that fired:

```python
nm["EventFormat"].value = "EVT3_0"                 # already set
tl["StreamEvsOutputFormat"].value = "XYTPFrame"    # instead of "CDFrame"
...
n   = int(buffer.size_filled / (buffer.bits_per_pixel / 8))   # valid event count
src = ctypes.cast(buffer.pdata, ctypes.POINTER(ctypes.c_float))
# stride 4: src[i*4+0]=x, +1=y, +2=t, +3=p
```

This is an `arena_api` float32 buffer, **not** a dv object.

---

## 1. The architectural decision (must settle before coding)

The camera exposes **one** `StreamEvsOutputFormat` node — output is either `CDFrame`
(today's preview) **or** `XYTPFrame` (raw events), not both simultaneously. So we cannot
"add a recorder beside the MP4 writer" without deciding how preview and recording share
the single stream.

**Recommended: Option A — single XYTP stream, render preview from events.**
Switch acquisition permanently to `XYTPFrame`; rebuild `render_cdframe` as a
from-events accumulator (the preview is reconstructed from the same `(x,y,t,p)` arrays
the recorder consumes). One stream feeds both. This makes preview, stats, and recording
all derive from real events — `pos_count`/`neg_count` become true counts, and the flash
filter and derivative plot get accurate data for free.

- *Cost:* touches `visualizer.py` and the stats path.
- *Benefit:* no mode-switching glitches; everything downstream is real-event-based.

Alternatives, for the record:
- **Option B — mode toggle:** keep CDFrame for preview, flip to XYTPFrame only while
  recording. Preview changes/degrades mid-take and stats diverge between modes. Less
  code now, worse operator experience and inconsistent metadata.
- **Option C — interleave:** unlikely supported on this sensor; verify on hardware, do
  not assume.

This plan assumes **Option A** unless hardware testing forces otherwise.

---

## 2. Hardware unknowns to pin down first (one short capture)

These are unanswered by the SDK example (it never reads `t` or `p`) and gate the HDF5 /
converter conventions. Resolve with a single throwaway XYTPFrame capture before locking
formats:

1. **`t` unit & origin** — float seconds vs microseconds; per-buffer-relative vs
   monotonic sensor clock. (Print `min/max t` across several buffers.)
2. **`p` encoding** — `{0,1}` vs `{-1,1}` vs `{0,255}`.
3. **`x,y` ranges** — confirm `0..width-1` / `0..height-1` and integer-valued floats.
4. **Buffer cadence/size** — events per buffer at representative scene rates (sizes the
   queue and HDF5 chunking).

Capture these into the metadata `polarity_encoding` / `timestamp_unit` fields rather
than hardcoding.

---

## 3. Build order

Priority unchanged from the spec: **1 → 2 → 3 → 4**. Items 1–3 must work before the next
capture session; item 4 (Event-3DGS / COLMAP) follows.

### Phase 1 — Acquisition switch + preview from events  *(Option A)*

- `constants.py`: add `EVS_OUTPUT_FORMAT = "XYTPFrame"` (and raw-record defaults:
  output root `recordings/`, HDF5 chunk size, queue capacity, default format).
- `camera_worker.py`:
  - Set `StreamEvsOutputFormat` from the new constant.
  - In the acquisition loop, decode each buffer into a contiguous
    `np.ndarray` of shape `(n, 4)` float32 (zero-copy view via `np.ctypeslib`/`frombuffer`
    on `buffer.pdata`, then `.copy()` before `requeue_buffer`).
  - Emit the event batch to (a) the renderer and (b) the recorder. Add a new
    `events_ready = Signal(np.ndarray, dict)` **or** hand batches to the recorder directly
    inside the worker thread (preferred — keeps high-rate data off the Qt signal/slot path;
    see Phase 1 threading note).
- `visualizer.py`: new `render_events(batch, width, height) -> (bgr, pos_count, neg_count)`
  that accumulates polarity into an image (white=pos, black=neg, gray=none), replacing the
  CDFrame pixel-threshold path. Real counts come from `p`.

**Threading note:** acquisition is a `QThread`. Per the spec, file I/O must not stall it.
The recorder owns a **bounded queue + dedicated writer thread**; the camera worker only
`put_nowait`s batches and increments a **dropped-events** counter on `queue.Full` (must stay
0). Do not write files from the camera worker or the GUI thread.

### Phase 1 (cont.) — Raw recorder

New `thinkcam/raw_recorder.py`, modeled on `recorder.py`'s start/stop/is_recording shape
but with the queue+thread:

- `start(width, height, label="")`: create session dir
  `recordings/<YYYYmmdd_HHMMSS>_<label>/`, open the HDF5 writer, start writer thread,
  record `start_utc`.
- `submit(batch)`: `put_nowait` onto bounded queue; count drops.
- writer thread: drains queue, appends blocks to chunked datasets; never holds the whole
  take in RAM.
- `stop()`: flush, close HDF5, join thread, then write `metadata.json` (final counts known
  only at stop), return session dir.

**HDF5 layout** (from spec §3.3 — four parallel 1-D arrays, not a compound dtype):

```
events.h5
  /events/x  uint16     /events/y  uint16
  /events/t  int64  (µs, monotonic — confirm unit in Phase 2 capture)
  /events/p  int8   (store as {0,1}; record convention in attrs)
  attrs: width, height, polarity_encoding, t_unit, sensor="IMX636"
```

Chunk ~1e6 along the event axis; gzip or lz4. Cast camera floats to these dtypes at write
time (after confirming `t` unit / `p` encoding).

**AEDAT4:** deferred. dv-processing is not installed and the camera does not hand us an
`EventStore`. If AEDAT4 is later required, add `dv-processing` to requirements and build an
`EventStore` from the XYTP arrays in the writer thread — track as a follow-up, not a
blocker for the capture session.

### Phase 2 — Metadata sidecar (Deliverable 2)

`metadata.json` written at stop (spec §3.5 schema). Sources, corrected for reality:
- `resolution`: `_cam_width/_cam_height` from the `connected` signal.
- `biases` / `noise_filter`: read from `constants.py`
  (`BIAS_THRESHOLD_POS/NEG`, `BIAS_REFRACTORY`, `BURST_FILTER_ENABLE`) — **not** UI state.
  If these become UI-adjustable later, switch the source then.
- `polarity_encoding` / `timestamp_unit`: the values confirmed in §2.
- `event_count` / `duration_s`: final tallies from the writer thread.
- `intrinsics`: nulls present, filled post-calibration.
- `take_label` / `operator_notes`: from the UI take-label field.

### Phase 1 (cont.) — GUI wiring

- `controls.py`: add a **"Record RAW"** checkable button (distinct from "Record Video")
  + a take-label `QLineEdit`. New signals `raw_record_toggled(bool)` and a label getter.
- `status_bar.py`: while raw-recording, show **elapsed, events written, event rate,
  file size, dropped-events** (0). Add labels alongside the existing ones.
- `main_window.py`: instantiate `RawEventRecorder`; wire `raw_record_toggled` →
  start/stop; route event batches to it; ensure `closeEvent` flushes/closes the recorder
  and writes the sidecar. Independent of MP4 — operator can run raw-only, MP4-only, both,
  neither.
- Shortcut: bind **`R`** to toggle raw recording (`S`/`P`/`Q`/`Esc` already exist;
  note: no `R` currently bound).

### Phase 3 — IncEventGS converter (Deliverable 3)

Standalone `convert_to_inceventgs.py` (spec §4). Reads `events.h5`, writes
`<scene>/event_threshold_<C>/gray_events_data.npy`. **VERIFY-FIRST against the real
`IncEventGS/datasets/` loader** (clone or read it) before locking: column order
(`[t,x,y,p]` vs `[x,y,t,p]`), `t` unit (s vs µs), polarity (`{0,1}` vs `{-1,1}`), dtype.
All four are CLI flags with defaults set from the verified loader; print the chosen
convention at runtime. Pose-free — emit `images/`/`traj.txt`/`poses_ts.txt` only if poses
supplied. Emit a config-YAML stub with `width,height,fx,fy,cx,cy` from `metadata.json`.

### Phase 4 — Event-3DGS converter (Deliverable 4, last)

Heavy external deps (event→intensity reconstruction + COLMAP). Per spec §5: **prefer the
repo's own preprocessing** for `images/` vs `images_event/`; run COLMAP for `sparse/`.
Out of scope for the first capture session — keep as a separate, later workstream.

---

## 4. Acceptance tests (carry over from spec)

- **Recorder:** load HDF5 → `len(x)==len(y)==len(t)==len(p)`, `t` non-decreasing,
  `x<width`, `y<height`, `p∈{0,1}`; `metadata.json` count == array length; multi-minute
  high-rate take keeps dropped-events at 0 and preview FPS steady; hard-stop mid-take
  leaves valid/readable files.
- **Preview parity (new, Option A):** events→frame preview is visually comparable to the
  old CDFrame preview on the same scene; `pos/neg_count` now reflect real events.
- **IncEventGS converter:** produced `.npy` matches verified shape/dtype/ranges;
  tiny synthetic round-trip lossless; dry-run the real IncEventGS dataloader against output.

---

## 5. Open decisions for the operator

1. **Option A vs B** for the preview/stream tradeoff (this plan assumes A).
2. Run the §2 hardware capture to confirm `t` unit and `p` encoding before formats lock.
3. AEDAT4: skip for now (no dv-processing) or add the dependency? Default: skip,
   HDF5-only, revisit if a downstream consumer needs replay in ThinkCam.
