# ThinkCam Phase 1 — Validation Baseline

This plan covers **Phase 1** of the Event-Camera 3D Reconstruction Pipeline: events →
intensity frames (E2VID) → COLMAP structure-from-motion → standard 3D Gaussian Splatting.
It picks up where [`ImplementationPlan.md`](ImplementationPlan.md) stops — that plan ends
at the IncEventGS converter (§3, Phase 3) and defers all COLMAP/3DGS work as "a separate,
later workstream". Everything upstream of intensity frames is **built and validated**;
everything from intensity frames onward is **absent**. This document is the plan to close
that gap.


---

## 0. What Phase 1 is, and what it is not

Phase 1 exists to answer one question: **is the captured event data geometrically
consistent enough to reconstruct from?** It produces a PSNR/SSIM/LPIPS **number to beat**,
not a deliverable reconstruction. If COLMAP cannot register the frames, the problem is in
the capture, not in the reconstruction method — and that is worth knowing before Phase 2
invests in a custom pipeline.

The report's Phase 1 description assumes more is off-the-shelf than actually is. Verified
against the live source:

| Phase 1 assumption | Verified reality | Source |
|---|---|---|
| E2VID is available in the Event-3DGS repo | **Absent.** No learned event-to-video code exists anywhere in it. Its Readme points at external `uzh-rpg/rpg_e2vid` and `ziweiWWANG/AKF`. Its own `--e2vid` flag only *scores* externally-produced frames. | `external/Event-3DGS/eval.py:235`, `external/Event-3DGS/Readme.md` |
| Event-3DGS can run a "standard 3DGS" baseline | **It cannot.** `readColmapSceneInfo` unconditionally reads a `renders/` folder, `--eval` is commented out so all cameras become training cameras, and `render()` forces `convert_SHs_python = True`. Not a vanilla floor. | `external/Event-3DGS/scene/dataset_readers.py`, `gaussian_renderer/__init__.py` |
| Its `convert.py` is the stock COLMAP wrapper | It is — except `--colmap_executable` defaults to a hard-coded Windows path. Non-empty, so on Linux every `os.system` call silently no-ops. | `external/Event-3DGS/convert.py:18-26` |
| Intrinsics come from the calibration takes | **No usable calibration take exists.** All three `calibration_test_*` recordings run 2-8 Kev/s with 25-61% duty cycle, and no offline solver was ever written. `fx`/`fy` are still `FIXME_REQUIRES_CALIBRATION`. | `datasets/thinkcam_capture/run_config_stub.yaml:20-21`, `recordings/*/metadata.json` |
| Frames can be exported from a take | No frame export of any kind exists. `_save_frame()` is one PNG per keypress; `VideoRecorder` writes the lossy preview. Neither is deterministic or windowed. | `thinkcam/main_window.py:183-191`, `thinkcam/recorder.py:8-38` |

**Two decisions follow from the table**, and this plan assumes both:

- **Baseline 3DGS is upstream `graphdeco-inria/gaussian-splatting`**, added as a second
  submodule. The whole point of a validation baseline is that it is unmodified.
  Event-3DGS stays untouched, reserved for Phase 2.
- **Intrinsics come from COLMAP self-calibration**, not from a calibration take. COLMAP
  solves and refines intrinsics during mapping, and `image_undistorter` emits the PINHOLE
  cameras the 3DGS loader requires. This unblocks Phase 1 with no reshoot and no solver to
  write. Real calibration is a **Phase 3 prerequisite**, where `e2calib` is needed for
  cross-sensor extrinsics anyway.

---

## 1. Pipeline contract

```
recordings/<session>/events.h5          # EXISTS — lossless (x,y,t,p), see ImplementationPlan §3
  │
  │  export_e2vid_input.py              # BUILD — streaming, windowed
  ▼
<work>/events.txt (or .zip)             # E2VID input format  <-- VERIFY-FIRST, see §3
  │
  │  rpg_e2vid/run_reconstruction.py    # ADD — external clone + pretrained weights
  ▼
<scene>/input/frame%06d.png             # 3-channel grayscale intensity frames
  │
  │  gaussian-splatting/convert.py      # ADD — wraps the colmap binary
  ▼
<scene>/images/  <scene>/sparse/0/      # undistorted PINHOLE cams + sparse point cloud
  │                                     # intrinsics solved HERE, written back to metadata.json
  │  gaussian-splatting/train.py
  ▼
output/<scene>/                         # trained model + results.json  <-- THE FLOOR
```

Phase 1 deliberately sidesteps two things that block the other routes:

- **IncEventGS's loader assert.** `ReplicaEventDataset`/`SimuEventDataset` require
  `len(poses_ts) == len(images) == len(traj)`, which a pure event-only capture cannot
  satisfy — documented at `convert_to_inceventgs.py:264-267`. Going through COLMAP
  produces real poses and real frames, so the assert is moot.
- **Offline calibration.** See §0.

---

## 2. Environment reality

**This machine cannot run Phase 1 today.** Four separate blockers, all fixable, none
optional:

**GPU driver.** An RTX 5080 is present (`GB203`, Blackwell, compute capability **sm_120**)
but the `nvidia-driver-595-open` kernel module is **not loaded** — `nvidia-smi` fails and
`lsmod` shows nothing. The package is installed, so this is most likely a pending reboot
or a Secure Boot / DKMS signing issue.

**The stock 3DGS environment is unusable on this GPU — this is the important one.**
Upstream `environment.yml` pins PyTorch 1.12.1 / cudatoolkit 11.6, which has **no sm_120
kernels at all**. It will not work on Blackwell no matter what the driver does. Locally
installed CUDA is **13.1**. So:

```
Do NOT use the shipped environment.yml.
Build a fresh env on PyTorch >= 2.7 with CUDA >= 12.8 (sm_120 support),
then compile the rasterizer explicitly for this card:

    TORCH_CUDA_ARCH_LIST="12.0" pip install submodules/diff-gaussian-rasterization
    TORCH_CUDA_ARCH_LIST="12.0" pip install submodules/simple-knn
```

**Two mutually incompatible torch pins.** E2VID's README suggests `cudatoolkit=10.0`;
3DGS pins 11.6. Neither runs on Blackwell, and they cannot share one environment. Use
**two separate venvs**. E2VID is a small recurrent UNet — it will run happily on modern
PyTorch, and on CPU if needed, so it is the easier of the two to modernise.

**Missing binaries.** `colmap` is not on PATH (required), and neither is `magick`
(only needed for `convert.py --resize`). Neither is `conda` — use `venv` + `pip`.

> Note on interpreters: `run_evs.sh:25` and the sanity-check snippet at
> [`CaptureGuide.md`](CaptureGuide.md) §6 both source `~/envs/default`, which **does not
> exist on this machine**. Every command in this document assumes an explicit venv path
> instead. State the interpreter when you run anything.

---

## 3. `VERIFY-FIRST` items

Three downstream conventions are **not yet confirmed**. Per
[`DataCollect.txt`](DataCollect.txt) §0, these must be settled by reading the downstream
loader, not by trusting a README.

**3.1 `VERIFY-FIRST` — the E2VID input format.** rpg_e2vid's README says only that it
reads "events from a text file or ZIP file" and documents **no** column order, timestamp
unit, or polarity encoding. Read the loader before writing the exporter:

```
rpg_e2vid/utils/inference_utils.py
    FixedDurationEventReader ....... window-by-time path
    FixedSizeEventReader ........... window-by-count path (pandas read_csv)
```

Confirm all four: header line contents (width/height?), column order (`t x y p`?),
timestamp unit (**seconds vs µs** — ThinkCam records µs), and polarity encoding
(`{0,1}` vs `{-1,+1}` — ThinkCam records `{0,1}`, see `raw_recorder.py:96`).

**3.2 `VERIFY-FIRST` — the COLMAP camera model 3DGS accepts.** Upstream
`scene/dataset_readers.py` rejects anything that is not PINHOLE or SIMPLE_PINHOLE. This
means `image_undistorter` **must** run — you cannot hand it raw OPENCV intrinsics.
`convert.py` does this for you; confirm the assert text before relying on it.

**3.3 `VERIFY-FIRST` — frame channel count.** `utils/general_utils.py::PILtoTorch`
returns a **1-channel** tensor for a single-channel PNG, and the camera loader slices
`[:3, ...]`. A 1-channel ground truth against a 3-channel render **broadcasts silently**
in `l1_loss` — no error, just a quietly wrong loss. Save E2VID output as **3-channel**
PNGs. Verify the tensor shape once, at the first camera load, rather than trusting this.

Record the resolutions in the exporter docstring using the ledger idiom already
established at `convert_to_inceventgs.py:8-15`.

---

## 4. What exists vs what must be built

**Exists and is directly reusable:**

- `_resolve_input(path)` — `convert_to_inceventgs.py:47-54`. Accepts either an
  `events.h5` or a session dir. Reuse verbatim so every converter's CLI feels identical.
- `remap_polarity(p, mode)` — `:72-81`. `mode="pm1"` gives signed `{-1,+1}` if E2VID
  wants it; `"01"` passes through.
- `write_config_stub(...)` — `:104-141`. The `calibrated = None not in (fx,fy,cx,cy)`
  idiom, the `FIXME_REQUIRES_CALIBRATION` sentinel, and the `(W/2, H/2)` principal-point
  default are exactly the shape needed to write **solved** intrinsics back.
- Timestamp re-zero + unit conversion — `:206-210`, and the `np.linspace` knot grid at
  `:241-242`. The latter is precisely the frame-timestamp grid an exporter needs, with
  `N = duration * fps`.
- `validate(...)` — `:91-101`. The bounds and monotonicity asserts generalise.

**Must be built or added:**

- `export_e2vid_input.py` — new, repo root, beside `convert_to_inceventgs.py`.
- `external/gaussian-splatting` — upstream submodule, plus its two CUDA extensions.
- `external/rpg_e2vid` — external clone plus pretrained weights.
- `colmap` binary, and a modern venv per §2.

**Do not reuse:** `load_events()` (`:57-69`) slurps all four arrays into RAM — 676 MB /
51.2 M events for the target take. `render_events()` (`thinkcam/visualizer.py:9-39`) is a
live-preview renderer keyed to `DISPLAY_FPS`, not a deterministic windowed accumulator.

---

## 5. Build order

Priority is strictly sequential — each step's output is the next step's input.

### Step 1 — Submodule and environment

```bash
git submodule add https://github.com/graphdeco-inria/gaussian-splatting external/gaussian-splatting
git -C external/gaussian-splatting submodule update --init --recursive
```

Then fix the driver (§2), install COLMAP, and build a venv on PyTorch ≥ 2.7 / CUDA ≥ 12.8,
compiling both extensions with `TORCH_CUDA_ARCH_LIST="12.0"`. **Gate:** `nvidia-smi`
reports the 5080 and `torch.cuda.is_available()` is `True` before continuing.

### Step 2 — `export_e2vid_input.py`

Reads a recording, emits E2VID's input format (§3.1). Requirements:

- **Must stream.** Read `/events/{x,y,t,p}` in chunks — the datasets are chunked at
  1,000,000 (`thinkcam/constants.py:36`), so iterate on that boundary. Do not call
  `load_events()`.
- **Must not assume a compression filter.** Takes before `20260601_203611` are LZF;
  later ones are uncompressed (`RAW_HDF5_COMPRESSION`, `constants.py:43`). h5py handles
  this transparently — just never hardcode it.
- Re-zero `t` to the first event and convert µs → whatever §3.1 confirms.
- Reuse `_resolve_input` and `remap_polarity`.
- Print every chosen convention at runtime, as `convert_to_inceventgs.py:252-267` does.

### Step 3 — E2VID reconstruction

```bash
git clone https://github.com/uzh-rpg/rpg_e2vid external/rpg_e2vid
wget "http://rpg.ifi.uzh.ch/data/E2VID/models/E2VID_lightweight.pth.tar" \
     -O external/rpg_e2vid/pretrained/E2VID_lightweight.pth.tar

python run_reconstruction.py \
    -c pretrained/E2VID_lightweight.pth.tar \
    -i <work>/events.txt \
    --auto_hdr \
    --output_folder <scene>/input
```

**The window size is the load-bearing parameter.** Too few frames starves COLMAP of
overlap and it fails to register; too many inflates matching cost quadratically under
exhaustive matching. Start at **~150-300 frames** across the 64 s orbit (roughly 2.5-5
fps) and adjust on COLMAP's registration rate. Save 3-channel PNGs (§3.3).

### Step 4 — COLMAP structure-from-motion

Frames go in `<scene>/input/`; upstream `convert.py` does the rest.

```bash
python external/gaussian-splatting/convert.py -s <scene>
```

This runs `feature_extractor` → `exhaustive_matcher` → `mapper` → `image_undistorter`,
producing `<scene>/images/` and `<scene>/sparse/0/`. Add `--no_gpu` for a CPU-only run.

> Using **upstream's** `convert.py` is deliberate: Event-3DGS's copy defaults
> `--colmap_executable` to a Windows path, so on Linux it fails silently rather than
> loudly (§0). If you ever do use that copy, pass `--colmap_executable colmap`.

Intrinsics are solved here. Write them back into the take's `metadata.json` `intrinsics`
block — currently hardcoded to nulls at `thinkcam/raw_recorder.py:251-255` — reusing the
`write_config_stub` idiom from §4 so `calibrated` flips to `true`.

### Step 5 — Train, render, score

```bash
python train.py   -s <scene> -m output/<scene> --eval
python render.py  -m output/<scene>
python metrics.py -m output/<scene>
```

`results.json` is the Phase 1 floor. Record it alongside the take label — that number is
what Phase 2 must beat.

---

## 6. Input data selection

Only the three June-2 takes are serious candidates. All are `lossless: true` with a 100%
duty cycle:

| Take | Events | Span | Duty | Rate |
|---|---|---|---|---|
| `20260602_101754_demo_scene_test_orbit_1` | 1,428,890 | 5.3 s | 100% | 271 Kev/s |
| `20260601_203611_object_orbit_4` | 3,992,716 | 22.9 s | 100% | 175 Kev/s |
| `20260602_102004_demo_scene_orbit_1` | 51,177,900 | 64.1 s | 100% | **798 Kev/s** |

Use `demo_scene_test_orbit_1` (5.3 s) as the **smoke-test loop** — it is small enough to
iterate the whole pipeline in minutes. Use `demo_scene_orbit_1` (64.1 s, full orbit) for
the **real run**.

**Excluded:** the six June-1 takes all run 25-70% duty cycle at 2-13 Kev/s — large dead
stretches where nothing fired, exactly the failure [`CaptureGuide.md`](CaptureGuide.md) §6
warns about. `20260601_175434` is worse still: its metadata predates hardware
verification (`timestamp_unit: "raw_sensor_float_unit_TBD"`), so its `t` column has
**undocumented units**.

> `datasets/thinkcam_capture/` was generated from that same `20260601_175434` take — the
> oldest schema and the lowest quality in the corpus, 8.3 s of events inside a 27.7 s
> recording. Its `run_config_stub.yaml:11` also points at a path that predates the repo
> move. Regenerate it from `demo_scene_orbit_1` or delete it; do not treat it as a
> reference.

**One note for whoever converts a long take to IncEventGS later:** the default
`--dtype float32 --t-unit us` holds integer precision only to 2²⁴ ≈ 16.7 s. Five of nine
takes exceed that, including `demo_scene_orbit_1` at 64.1 s. `convert_to_inceventgs.py:213-216`
only prints a warning. Pass `--t-unit s` or `--dtype float64`.

---

## 7. Acceptance tests

- **Exporter:** emitted event count equals the H5 length; `t` non-decreasing after
  export; peak RSS stays flat across a 51.2 M-event take (proves streaming); re-reading
  the emitted file round-trips to the same `(x,y,t,p)`.
- **E2VID:** frame count matches the requested window grid; output PNGs are 1280×720 and
  **3-channel**; no all-black or all-saturated frames.
- **COLMAP:** ≥90% of input frames registered; solved `fx`/`fy` physically plausible for
  the lens at its locked focus; `sparse/0/cameras.bin` reports PINHOLE after undistortion.
- **3DGS:** trains 30 k iterations without NaN loss; `metrics.py` emits `results.json`
  with PSNR/SSIM/LPIPS.
- **Smoke gate:** the 5.3 s take must complete end-to-end **before** the 64 s take is
  attempted. Do not debug a 51 M-event run.

---

## 8. Open decisions for the operator

1. **Fix the driver locally, or move to a GPU box?** Phase 1 is blocked on CUDA either
   way. The 5080 is capable, but sm_120 forces a modern-PyTorch rebuild of the rasterizer
   (§2) that the upstream repo does not document.
2. **E2VID, AKF, or FireNet?** The report specifies E2VID and it is the best-documented,
   but Event-3DGS's Readme recommends AKF alongside it. Reconstruction quality here caps
   everything downstream — it may be worth running two and comparing COLMAP registration
   rates before committing.
3. **Where do solved intrinsics live?** Writing them back into the recorded
   `metadata.json` mutates a capture artifact; a sidecar keeps recordings immutable.
   Recommend a sidecar, with `calibrated` in the metadata left as the source of truth for
   *whether* a solution exists.
4. **Regenerate or delete `datasets/thinkcam_capture/`?** It is stale on every axis (§6).
5. **Frame rate for export** — settle after the first COLMAP registration rate comes back;
   it is the parameter most likely to need two or three attempts.

---

## Checklist per Phase 1 attempt

- [ ] GPU driver loaded; `torch.cuda.is_available()` is **True** on PyTorch ≥ 2.7.
- [ ] Both CUDA extensions compiled with `TORCH_CUDA_ARCH_LIST="12.0"`.
- [ ] `colmap` on PATH.
- [ ] All three **`VERIFY-FIRST`** items (§3) resolved by reading the loader, and the
      resolutions recorded in the exporter docstring.
- [ ] Exporter **streams** — peak RSS flat on the 51.2 M-event take.
- [ ] E2VID frames are **3-channel** PNGs.
- [ ] COLMAP registered **≥90%** of frames; cameras are PINHOLE.
- [ ] Solved intrinsics recorded, and `calibrated` no longer `false`.
- [ ] Smoke take (5.3 s) passed end-to-end **before** the 64 s run.
- [ ] `results.json` archived against the take label — that is the floor.
