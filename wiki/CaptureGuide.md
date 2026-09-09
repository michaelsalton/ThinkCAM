# ThinkCam Capture Guide

How to collect usable raw event-camera data for later 3D reconstruction
(IncEventGS first, Event-3DGS/EventSplat later).

The recorder and HDF5 format are finalized and validated; this guide covers the
**capture procedure**, which is what actually determines whether a take is usable.
Each take is written to `recordings/<timestamp>_<label>/{events.h5, metadata.json}`.

---

## 1. Launch

```bash
cd /home/ubuntu/Documents/Development/ThinkCam
./run_evs.sh
```

This wires up the ArenaSDK libraries + GenTL producer, activates the `~/envs/default`
venv, and starts the GUI (`run_evs.sh` → `thinkcam.main`). You should see the live
event preview once the camera at `169.254.80.199` connects.

## 2. Set the lens once, then don't touch it

Focus the lens on a subject at your working distance and **lock focus and aperture**.
Intrinsics are tied to this setting — if you change focus or zoom after calibrating,
the calibration (and any take that relies on it) is invalid.

## 3. Shoot a calibration take (first, once per lens setting)

Required for any reconstruction later, even though the recorder stores
`intrinsics: null` (they're solved offline afterward).

1. Type a label like `calib` in the **Take label** field.
2. Click **Record RAW** (or press **`R`**) to start — the button turns red ("Stop RAW").
3. Hold a **checkerboard / AprilGrid** in front of the camera and move it slowly so it
   appears across the **whole frame** at several angles and distances (~20–30 s). Events
   come from motion, so keep it gently moving.
4. Press **`R`** again to stop.

## 4. Shoot the subject — orbit *with translation*

This is the part that determines whether the data is usable.

1. New take label (e.g. `phantom_orbit_01`), press **`R`**.
2. **Arc the camera around the subject** with real travel — physically move around it,
   keeping it roughly centered. Do **not** just pan/rotate in place; the reconstruction
   needs parallax.
3. Keep **continuous, smooth motion** the whole time (events only fire on change — no
   motion = dead footage). Aim for a full orbit or sweep over ~20–60 s.
4. Good texture/contrast and adequate lighting help event density.
5. Press **`R`** to stop. Repeat for more views/objects as needed.

## 5. Watch the status bar while recording

It shows elapsed / events written / event rate / file size / **dropped-events**.
**Dropped must stay 0.** The recorder is configured to out-run the camera's max
output rate (uncompressed HDF5, `RAW_HDF5_COMPRESSION = None`), so drops should
not occur. If they ever do, the scene rate is extreme — lower the camera's
`ERC_RATE_LIMIT_MEV` (on-sensor rate cap) or raise the contrast-threshold biases
in `thinkcam/constants.py`. A healthy take shows a steadily climbing event count
with no dead stretches. After stop, `metadata.json` records `dropped_events` and
a `lossless` flag — **only keep takes where `lossless: true`**.

> Trade-off: uncompressed capture is larger (~13 bytes/event; at the 10-MEV cap
> that's ~130 MB/s). Keep takes to tens of seconds, or set `RAW_HDF5_COMPRESSION`
> to `"lzf"`/`"gzip"` for long takes where disk space matters more than peak rate.

## 6. Sanity-check each take

Confirm a take is dense and continuous (not like an early test take that was only active
~8 s out of 28 s):

```bash
~/envs/default/bin/python -c "
import h5py, numpy as np
g = h5py.File('recordings/<your_dir>/events.h5','r')['events']
t = g['t'][:]/1e6; n = len(t)
print(f'{n:,} events, span {t.max()-t.min():.1f}s, ~{n/(t.max()-t.min()):.0f} ev/s')
"
```

Want activity spanning essentially the whole take at a healthy rate.

---

## Checklist per session

- [ ] Lens focused and **locked** (focus/aperture unchanged for the whole session).
- [ ] Calibration take shot (checkerboard/AprilGrid filling the frame, multiple angles).
- [ ] Subject takes shot as **orbit with translation**, continuous motion, well lit.
- [ ] Dropped-events stayed **0** on every take.
- [ ] Each take sanity-checked for density + continuous time span.

## Notes for later processing (not needed at capture time)

- Convert a take to IncEventGS layout with `convert_to_inceventgs.py`.
- Intrinsics are solved offline from the calibration take and written into the
  IncEventGS run config YAML (`cam.fx/fy/cx/cy`), not the dataset dir.
- See [ImplementationPlan.md](ImplementationPlan.md) and [DataCollect.txt](DataCollect.txt)
  for the reconstruction-side pipeline and remaining gates.
