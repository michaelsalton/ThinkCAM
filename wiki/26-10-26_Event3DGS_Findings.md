# Event-3DGS on EventNeRF Lego

First runs of Event-3DGS (`external/Event-3DGS`, NeurIPS 2024) on the EventNeRF synthetic
lego truck, against the upstream 3DGS baseline from
[01_Lego_Baseline_Results.md](01_Lego_Baseline_Results.md). All runs from 2026-10-26.

**Headline:** the event loss in Event-3DGS is worth about +1 dB against the true scene,
and only once it is fed event information E2VID doesn't already carry. Where the
intensity frames come from matters far more. Plain 3DGS on frames integrated directly
from the events reaches **31.6 dB** against the true renders. Everything trained on
E2VID frames, Event-3DGS included, sits at **~21 dB**. Event-3DGS trained from scratch
collapses every time; it needs a warm start from an intensity-only model.

**Follow-up (§7):** under realistic sensor conditions the intensity advantage
disappears, but the event-loss gain survives and grows. With no starting image, 5%
threshold mismatch and 1 Hz noise, integrated frames are *worse* than E2VID as an
intensity source (18.7 vs 20.7 dB). Used only as Event-3DGS's event frames, on top of
E2VID intensity, they give **22.1 dB, +1.4 dB over the E2VID baseline**. That
configuration (E2VID intensity, integrated events, warm start) is the one to carry to
real captures.

EventSplat has no public code. IncEventGS is running on lego (§7.5). Its pose-free
tracking drifts to 43 cm ATE over the first 141 frames (COLMAP on E2VID: 1.64 cm),
though it stays at 4–7 cm within any third of the orbit. Final global BA and render
scores are pending.

---

## 1. Setup

| | |
|---|---|
| Scene | EventNeRF synthetic `lego`, 346×260, 2,350,773 events over 1.0 s |
| Poses | COLMAP `simple` (SIMPLE_RADIAL) on E2VID frames, undistorted = `scene_e2vid/gs` (191 views, ATE 1.64 cm). The same poses are used for every run. |
| Holdout | sorted views with idx % 8 == 0 → 24 views (= upstream `--eval` llffhold) |
| Iterations | 30 k from scratch; 8 k for warm-started event runs (README example) |
| Env | `~/envs/phase1` (torch 2.13 / cu130, RTX 5080); upstream `diff_gaussian_rasterization` |

### How Event-3DGS uses data

It never reads raw events. A scene folder holds:

- `images/`: intensity frames (the L1 target)
- `images_event/`: frames whose consecutive log differences stand in for events
- `renders/`: GT for its test hook (loaded, unused in training)
- `sparse/0`: COLMAP poses

With `--event` each step renders view *i* and *i+1*. It compares
`(log I_{i+1} − log I_i) / c` (learnable *c*) against the same quantity from
`images_event`, with *c* fixed at 0.17. Event weight is hard-coded at 0.9, applied on
pixels where the event target is non-zero; the intensity L1 covers the rest. `--gray`
makes the model grayscale.

## 2. Event sources compared

| `images_event` | What it is |
|---|---|
| **E2VID** | the same E2VID frames as `images/`, so the event loss adds no new information |
| **Integrated** | per-pixel integration of the raw events at each E2VID frame's timestamp (`integrate_events.py`), warped through the same COLMAP undistortion |

How the EventNeRF lego data is encoded (checked against the renders, documented in
`integrate_events.py`):

- `t` is the integer render index (1 ms); every event between renders *k−1* and *k* has `t = k`.
- RGGB Bayer sensor. Correlation with the renders is 0.96 for RGGB and ~0.5 for the other three layouts.
- Events are on **log(sRGB)**, threshold C = 0.25 (EventNeRF `configs/nerf/lego.txt`).
  EventNeRF's own loss uses `srgb**2.2`, but integrating that way washes out contrast here.
- The t = 0 burst (54,168 events) encodes frame 0 against a flat sRGB grey of 159
  (`init159`), so `log I(k) = log(159/255) + 0.25 · Σp` is exact up to threshold quantisation.

Integrated frames reach **~29.5 dB** against the renders (r_00000/250/500/750/995).
That only holds because the events are noiseless simulation; see §5.

## 3. Results on held-out E2VID views

This is the baseline doc's protocol: 24 held-out views, each scored against its E2VID frame.
It measures consistency with E2VID, not with the true scene.

| Run | Event frames | PSNR | SSIM | LPIPS | Gaussians |
|---|---|---|---|---|---|
| Upstream 3DGS (baseline, rescored) | none | **33.36** | **0.954** | **0.109** | 120 k |
| Event-3DGS `--gray` | none | 32.69 | 0.949 | 0.114 | 136 k |
| Event-3DGS `--gray --event`, scratch, 30 k | E2VID | 23.41 | 0.844 | 0.268 | 2.0 k |
| Event-3DGS warm start, best (1 k) | E2VID | 27.44 | 0.904 | 0.166 | 109 k |
| Event-3DGS warm start, 8 k | E2VID | 26.71 | 0.916 | 0.154 | 90 k |
| Event-3DGS `--gray --event`, scratch, 30 k | Integrated | 25.41 | 0.870 | 0.239 | 2.3 k |
| Event-3DGS warm start, 1 k | Integrated | 26.99 | 0.911 | 0.166 | 111 k |
| Event-3DGS warm start, 8 k | Integrated | 26.71 | 0.911 | 0.176 | 86 k |

"Warm start" means `--start_checkpoint` from the `--gray` run at 30 k, the procedure the
README recommends.

On this metric the event loss looks like a 5–6 dB loss. That turns out to be the metric:
the event loss pulls the model away from E2VID's tone mapping and towards real contrast,
and §4 shows the change is an improvement.

Upstream 3DGS trained on the *integrated* frames scores 34.16 / 0.970 / 0.039 on its own
holdout. Its holdout GT is the integrated frames, so that number is not comparable to
the rows above.

## 4. Results against the true scene

The 201 RGB renders (`scene_rgb/input`, 5 ms spacing) were registered into the E2VID
COLMAP model with `colmap image_registrator`. That step adds images without moving any
existing pose; E2VID poses are bit-for-bit unchanged. 189 / 201 registered, and ATE over
all 380 poses is 1.45 cm. The registered views were undistorted with the same camera
(`scene_e2vid/rgb_eval`).

None of these views is a training view. Scoring is on Rec.601 luma, since every
Event-3DGS model is grayscale. "Tone-fitted" is a per-view least-squares gain + bias on
the render before scoring. E2VID has no absolute intensity, so without the fit the raw
column mostly measures tone mismatch.

| Model | Intensity | Event frames | Raw PSNR | Tone-fitted PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|---|
| Upstream 3DGS (baseline) | E2VID | none | 13.88 | 20.69 | 0.895 | 0.107 |
| Event-3DGS `--gray` | E2VID | none | 13.85 | 20.57 | 0.892 | 0.113 |
| Event-3DGS scratch, 30 k | E2VID | E2VID | 13.70 | 19.30 | 0.822 | 0.208 |
| Event-3DGS warm, 2 k | E2VID | E2VID | 14.04 | 20.61 | 0.880 | 0.127 |
| Event-3DGS warm, 8 k | E2VID | E2VID | 14.35 | 20.93 | 0.888 | 0.117 |
| Event-3DGS scratch, 30 k | E2VID | Integrated | 14.03 | 20.86 | 0.866 | 0.168 |
| Event-3DGS warm, 1 k | E2VID | Integrated | 14.17 | 21.04 | 0.897 | 0.115 |
| **Event-3DGS warm, 8 k** | E2VID | Integrated | 14.26 | **21.59** | **0.910** | **0.106** |
| **Upstream 3DGS** | **Integrated** | none | **31.22** | **31.63** | **0.949** | **0.076** |

SSIM and LPIPS are from the tone-fitted pass.

## 5. Findings

1. **The event loss helps a little when it has real event information.** With
   integrated event frames and a warm start, Event-3DGS beats its own gray-only run by
   +1.0 dB (tone-fitted), the E2VID baseline by +0.9 dB, and the E2VID-event variant by
   +0.7 dB. SSIM and LPIPS improve too. The gain grows over the 8 k steps rather than
   peaking early.
2. **When `images_event` is E2VID, the event loss does almost nothing** (+0.2 dB over
   the baseline). That is expected: it only re-weights information `images/` already
   supplies. Event-3DGS's paper builds `images_event` with a different estimator for
   exactly this reason.
3. **Intensity source dominates.** Swapping E2VID for integrated frames under plain
   3DGS is worth about +11 dB tone-fitted, against +1 dB for the best Event-3DGS loss.
4. **Event-3DGS from scratch always collapses.** With either event source the Gaussian
   count drops to ~2 k by 14 k iterations and most of the truck disappears; it's the
   failure mode its README describes. The warm start avoids it.
5. **The E2VID-holdout metric is misleading for event-supervised methods.** It ranked
   the event runs 5–6 dB below the baseline; against the true scene they are 0.2–0.9 dB
   above it. Future comparisons should score against the true renders where they exist.

### What carries over to ThinkCam captures

Direct integration is exact here only because the events are simulated: no noise, one
threshold for every pixel, and a known starting image (the `init159` burst). Real
sensors have per-pixel threshold mismatch, noise, refractory effects and no known
starting frame, so plain integration drifts. On real captures the intensity source
will still be E2VID or a filter such as high-pass or AKF (which the Event-3DGS README
recommends). The transferable lesson is to put effort into the event-to-intensity step
before the splatting method, and to score against something other than E2VID frames.

## 6. Changes to `external/Event-3DGS`

The submodule now shows as modified; all patches are local:

| File | Change | Why |
|---|---|---|
| `gaussian_renderer/__init__.py` | `antialiasing=False` in raster settings; unpack 3 outputs | Its `submodules/diff-gaussian-rasterization` was never checked out; the env has the current upstream rasterizer (returns `color, radii, invdepths`). Same maths with AA off. |
| `utils/loss_utils.py` | comment out `import torchgeometry` | unused; doesn't install on torch 2.13 |
| `utils/image_utils.py` | import `lpips` inside `LPIPS()` | only eval paths use it |
| `train.py` | `--holdout N` | skip sorted views with idx % N == 0, plus (in `--event`) pairs whose next frame is held out, so held-out frames can't leak in through the event loss |
| `train.py` | `torch.load(..., weights_only=False)` | torch ≥ 2.6 default refuses its own checkpoints |

Not fixed: `render.py` imports `Event_sensor`, which ships a Windows-only `.pyd`. Scoring
goes through `work/eventnerf/eval_splat.py`, which uses Event-3DGS's loader and renderer
directly. That script reproduces the baseline's `metrics.py` numbers exactly
(33.36 / 0.9542 / 0.1085), so Event-3DGS's renderer and the upstream one agree.

Two smaller differences from the baseline: held-out views stay in Event-3DGS's camera
list (they are never sampled), so its scene-extent normalisation includes them. Event
mode also samples indices 2…N−4 only, as upstream code does.

## 7. Follow-up: realistic events, integrated intensity, other methods

Same scene, poses and RGB evaluation as §4.

### 7.1 Event-3DGS with integrated frames as intensity too

`scene_e2vid/e3dgs_int2`: `images/` and `images_event/` both hold the integrated frames.

| Run | Raw PSNR | Tone-fitted PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| Upstream 3DGS on integrated (§4) | 31.22 | 31.63 | 0.949 | 0.076 |
| Event-3DGS `--gray` | 31.29 | **31.75** | 0.949 | **0.075** |
| Event-3DGS warm, 1 k | 29.89 | 30.01 | 0.936 | 0.089 |
| Event-3DGS warm, 4 k | 30.71 | 30.77 | 0.942 | 0.084 |
| Event-3DGS warm, 8 k | 30.84 | 30.91 | 0.943 | 0.084 |

When the intensity frames are already near-exact, the event loss has nothing to add.
It costs about 0.8 dB, recovering slowly from an initial 1.7 dB drop.

### 7.2 Realistic event-to-intensity

`integrate_events.py` gained sensor-emulation flags:
- `--no-init`: drop the t = 0 burst and start from flat grey
- `--alpha`: high-pass leak in rad/s, the event-only form of Scheerlinck's complementary filter
- `--mismatch`: per-pixel threshold error, as a std of C
- `--noise`: background activity, Hz per pixel

Frame quality is gray, tone-fitted PSNR against the renders at 250 / 500 / 750 / 995 ms:

| Setting | Mean PSNR |
|---|---|
| exact (with init burst) | 31.31 |
| no init | 18.44 |
| no init + leak α = 1 / 3 / 10 / 30 rad/s | 18.86 / 18.74 / 18.22 / 17.74 |
| no init + 5% mismatch | 18.44 |
| no init + 5% mismatch + 0.1 Hz noise | 18.41 |
| no init + 5% mismatch + 1 Hz noise | 18.28 |
| no init + 5% mismatch + 1 Hz noise + α = 3 (**"realistic"**) | 18.66 |

Losing the starting image costs 13 dB. Every pixel that began on the truck keeps its
starting error, and a leak only partly forgets it; mismatch and noise are minor next
to that. The flat grey background flatters these numbers, because the grey-start guess
is exactly right there; a textured real scene would be worse.

As an intensity source for upstream 3DGS (`gs_noinit`, `gs_real`):

| Intensity | Raw PSNR | Tone-fitted PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| E2VID (baseline) | 13.88 | **20.69** | **0.895** | **0.107** |
| Integrated, no init | 16.90 | 18.44 | 0.840 | 0.179 |
| Integrated, realistic | 16.80 | 18.66 | 0.841 | 0.278 |

**Under realistic conditions E2VID is the better intensity source.** The +11 dB of §4
came entirely from the simulator's known starting frame.

### 7.3 Event-3DGS with realistic event frames

`scene_e2vid/e3dgs_real`: `images/` = E2VID, `images_event/` = realistic integrated
frames, warm start from the `--gray` 30 k checkpoint.

| Run | Raw PSNR | Tone-fitted PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| E2VID baseline (upstream 3DGS) | 13.88 | 20.69 | 0.895 | 0.107 |
| Event-3DGS `--gray` (warm-start source) | 13.85 | 20.57 | 0.892 | 0.113 |
| Event-3DGS warm, exact event frames, 8 k (§4) | 14.26 | 21.59 | 0.910 | 0.106 |
| Event-3DGS warm, realistic event frames, 1 k | 14.24 | 20.97 | 0.889 | 0.127 |
| Event-3DGS warm, realistic event frames, 4 k | 14.61 | 21.81 | 0.903 | 0.116 |
| **Event-3DGS warm, realistic event frames, 8 k** | **14.72** | **22.09** | **0.907** | 0.114 |

The event-loss gain survives realistic conditions: +1.4 dB over the baseline and
+1.5 dB over its own warm-start source, still climbing at 8 k. It slightly beats the
exact event frames. That is consistent with the loss only using log differences
between consecutive frames: a wrong starting image cancels out of a difference, and
the leak removes slow drift that would otherwise build up. This is one seed, so the
0.5 dB gap over exact frames is within run-to-run noise and shouldn't be read as an
improvement. The +1.4 dB over the baseline is larger than anything else seen here.

**The configuration to carry to real ThinkCam captures:** E2VID intensity, COLMAP poses
from E2VID, `images_event` from leaky integration of the raw events, gray warm start,
then `--event`.

### 7.4 EventSplat and IncEventGS

- **EventSplat** (Yura et al., CVPR 2025, Sony / U of T): no public code, found on
  neither GitHub nor the paper pages. Its method (E2VID initialisation + event loss +
  spline poses) is close to the warm-started Event-3DGS runs above.
- **IncEventGS** (`external/IncEventGS`, WU-CVGL, CVPR 2025): set up, not yet run.
  - `~/envs/iegs`: py3.12, torch 2.13 / cu130. Kept separate from `phase1`.
  - Its bundled gsplat 1.1.1 JIT-builds against CUDA 13.1 for sm_120. That needs
    `CUDA_HOME=/usr/local/cuda-13.1`, plus `~/envs/iegs/bin` on `PATH` for ninja.
  - Marigold weights (`Bingxin/Marigold`, 7.3 GB) are in `~/models` and load through its
    diffusers community pipeline. tiny-cuda-nn is not needed on the 3DGS path.
  - Dataset: `work/eventnerf/to_inceventgs.py` → `lego/inceventgs`, in Replica layout.
    The t = 0 burst is dropped, and events are spread evenly within each 1 ms bin in
    shuffled order, since the simulator orders them spatially.
  - Config: `work/eventnerf/inceventgs_lego.yaml`, starting from their room0 config.
  - Known limitation: lego events are RGGB Bayer and IncEventGS treats them as gray, so
    coloured surfaces get per-channel contrast differences.

  IncEventGS is pose-free, so its renders come at its own estimated poses for every
  4th RGB frame (`img_eval/`); they get scored against those frames.

### 7.5 IncEventGS run on lego (in progress)

Started 2026-10-26 16:22 (initialisation took ~4 min; tracking began 16:26), output in `work/eventnerf/lego/out_inceventgs/lego/`, log at
`out_inceventgs/run.log`. Status as of 18:15: tracking past frame 140 of 200, no errors.

**Patch.** `tools/eval_ate.py:89` called `numpy.linalg.linalg.svd`, which NumPy 2
removed. Changed to `numpy.linalg.svd`. The first attempt crashed on it at the first
pose evaluation; nothing else needed changing. It's the only change to
`external/IncEventGS`.

**Stages seen:**
1. Initialisation on frames 1–5 (25 ms of events, 81 k events, 6 control knots).
   Event loss fell from 0.040 to 0.011 over 4 k iterations, with about 110 k Gaussians.
2. Marigold depth bootstrap: depth estimated from the initial render, and the scene
   restarted from it (about 1 k Gaussians, regrowing), then optimised for another 3.5 k
   iterations.
3. Incremental tracking plus local BA from frame 6 onward. BA runs every second frame
   on a 20-frame sliding window.

**Speed.** About 30 s per frame early on, slowing to 55–75 s per frame by frame 120–140
as the scene grows; BA densification now runs at about 410–445 k Gaussians. GPU use
is about 70% and 3 GB. Expected total: about 3 h for 200 frames plus final global BA,
against 3–5 min for a 30 k Event-3DGS run.

**Pose accuracy so far.** These are tracking poses before global BA. They come from
`tracking/pose_file_{est,gt}_tracking_f141_0200.txt`, Umeyama Sim3-aligned the same way
as `eval_poses.py`:

| Poses | ATE rmse | ATE max | Note |
|---|---|---|---|
| frames 1–141 (~254° of orbit) | **43.3 cm** (17% of 2.52 m bbox diagonal) | 91.1 cm | IncEventGS's own log: 44.2 cm at 143 |
| frames 1–47, aligned alone | 4.25 cm | | |
| frames 47–94, aligned alone | 6.72 cm | | |
| frames 94–141, aligned alone | 6.00 cm | | |
| *COLMAP on E2VID (baseline doc)* | *1.64 cm* | *6.23 cm* | *all 191 frames* |

The pattern is drift: locally the trajectory has the right shape, but error builds up
around the orbit. That's expected of incremental tracking without loop closure. Whether
the final global BA pulls it back is the main open question. For comparison, VGGT on
E2VID (baseline doc) had the *wrong shape* even locally (25–45 cm per third).

**Render quality so far.** Training-view renders are recognisable from the start. By
frame 31 the truck is clean. By frame 143 there is visible doubling and ghosting on the
arm and cab, consistent with the drift. Throughout, coloured surfaces show a fine
checkerboard: the lego events are RGGB Bayer, and IncEventGS models them as gray, so
neighbouring pixels see different contrast steps (the limitation flagged in §7.4).

**Still to do when it finishes:** read the final global-BA ATE, then score
`img_eval/est` against `img_eval/gt`. That's every 4th RGB frame between 15 ms and
995 ms, rendered at IncEventGS's own estimated poses. Score in gray, raw and
tone-fitted, so it lines up with §4 and §7.3. Caveat: those are training-trajectory
views at estimated poses, while §4 used COLMAP-registered poses, so the comparison is
close but not identical.

## 8. Next

- Finish IncEventGS (§7.5): final ATE and render scores.
- Repeat §7.3 with a few seeds to put error bars on the +1.4 dB.
- Apply the §7.3 configuration to a ThinkCam capture. That needs COLMAP-on-E2VID poses
  (Phase 1b) and leaky integration of the HDF5 events. There is no RGB ground truth
  there, so score on held-out E2VID frames plus visual comparison.

## 9. Reproduce

From `work/eventnerf/`, `P=~/envs/phase1/bin/python`, `NPZ=data/nerf/lego/train/events/test_lego1_color_init159_1e-6eps.npz`.

```bash
# event-integrated frames at E2VID timestamps, warped like gs/images
$P integrate_events.py $NPZ lego/integ_e2vid_times --names lego/scene_e2vid/input --stamps lego/e2vid_stamps.txt
colmap image_undistorter --image_path lego/integ_e2vid_times --input_path lego/scene_e2vid/simple/sparse/2 \
  --output_path <tmp> --output_type COLMAP          # then <tmp>/images -> scene_e2vid/e3dgs_int/images_event
# scene_e2vid/e3dgs:     images, images_event, renders -> ../gs/images ; sparse -> ../gs/sparse
# scene_e2vid/e3dgs_int: same, but images_event = undistorted integrated frames
# scene_e2vid/gs_int:    images -> ../e3dgs_int/images_event ; sparse -> ../gs/sparse

# Event-3DGS (run from external/Event-3DGS; L=.../work/eventnerf/lego)
python train.py -s $L/scene_e2vid/e3dgs -m $L/out_e3dgs_gray --gray --holdout 8 --iterations 30000 \
  --save_iterations 3999 7999 13999 20000 30000 --checkpoint_iterations 30000
python train.py -s $L/scene_e2vid/e3dgs_int -m $L/out_e3dgs_int_ft --gray --event --holdout 8 --iterations 8000 \
  --start_checkpoint $L/out_e3dgs_gray/chkpnt30000.pth --save_iterations 999 1999 3999 5999 8000
#   scratch runs: drop --start_checkpoint, --iterations 30000 (out_e3dgs, out_e3dgs_int)
#   E2VID-event variant: -s $L/scene_e2vid/e3dgs (out_e3dgs_ft)

# upstream 3DGS on integrated frames (run from external/gaussian-splatting)
python train.py -s $L/scene_e2vid/gs_int -m $L/out_int --eval --port 6019

# RGB registration into the E2VID model (poses of existing images untouched)
#   scene_e2vid/rgbreg/imgs = symlinks to scene_e2vid/input/* + scene_rgb/input/*; db.db copied from simple/
colmap feature_extractor --database_path rgbreg/db.db --image_path rgbreg/imgs --ImageReader.existing_camera_id 1 --SiftExtraction.use_gpu 0
colmap exhaustive_matcher --database_path rgbreg/db.db --SiftMatching.use_gpu 0
colmap image_registrator --database_path rgbreg/db.db --input_path simple/sparse/2 --output_path rgbreg/sparse
colmap image_undistorter --image_path rgbreg/imgs --input_path rgbreg/sparse --output_path <tmp> --output_type COLMAP
#   scene_e2vid/rgb_eval: undistorted r_*.png + images.txt filtered to r_* + cameras.txt + gs points3D.ply

# scoring
$P eval_splat.py lego/scene_e2vid/e3dgs lego/out_e3dgs_ft 8000                              # E2VID holdout (§3)
$P eval_splat.py lego/scene_e2vid/rgb_eval lego/out_e3dgs_ft 8000 --holdout 1 --gray [--affine] --tag _rgb   # §4
```

Each `eval_splat.py` run writes `<model>/eval_<iter><tag>.json` with per-view scores.

Follow-up runs (§7):

```bash
# realistic event frames at E2VID timestamps, then undistort as above -> scene_e2vid/gs_real/images
$P integrate_events.py $NPZ lego/integ_real --names lego/scene_e2vid/input --stamps lego/e2vid_stamps.txt \
  --no-init --mismatch 0.05 --noise 1 --alpha 3
# scene_e2vid/e3dgs_real: images, renders -> ../gs/images ; images_event -> ../gs_real/images ; sparse -> ../gs/sparse
python train.py -s $L/scene_e2vid/e3dgs_real -m $L/out_e3dgs_real_ft --gray --event --holdout 8 --iterations 8000 \
  --start_checkpoint $L/out_e3dgs_gray/chkpnt30000.pth --save_iterations 999 3999 8000

# IncEventGS (from external/IncEventGS; needs the NumPy-2 patch in tools/eval_ate.py, §7.5)
$P to_inceventgs.py data/nerf/lego/train lego/scene_rgb/input lego/inceventgs
CUDA_HOME=/usr/local/cuda-13.1 PATH=~/envs/iegs/bin:/usr/local/cuda-13.1/bin:$PATH TORCH_CUDA_ARCH_LIST=12.0 \
  ~/envs/iegs/bin/python main.py --config ../../work/eventnerf/inceventgs_lego.yaml
```

## 10. Picking this up

How to carry on from where these runs stopped. Paths are relative to
`work/eventnerf/` unless they start with `external/` or `~`.

### 10.1 What's where

| Thing | Location |
|---|---|
| Scenes (COLMAP + image folders, mostly symlinks into `gs/`) | `lego/scene_e2vid/{gs, e3dgs, e3dgs_int, e3dgs_int2, e3dgs_real, gs_int, gs_noinit, gs_real, rgb_eval, rgbreg}` |
| Trained models | `lego/out_*`: `out_e2vid` (baseline), `out_e3dgs_*` (Event-3DGS), `out_int` / `out_noinit` / `out_real` (upstream 3DGS on integrated frames), `out_inceventgs` |
| Per-run scores | `lego/out_*/eval_<iter>{,_rgb,_rgbaff}.json` from `eval_splat.py` |
| Warm-start checkpoint used by every `_ft` run | `lego/out_e3dgs_gray/chkpnt30000.pth` |
| Scripts | `integrate_events.py`, `eval_splat.py`, `score_pairs.py`, `to_inceventgs.py`, `inceventgs_lego.yaml` |
| Training logs | `lego/train_*.log`; IncEventGS: `lego/out_inceventgs/run.log` |

### 10.2 Environments

- **`~/envs/phase1`** (py3.14, torch 2.13/cu130): Event-3DGS, upstream 3DGS, every
  scoring script, `integrate_events.py`. `P=~/envs/phase1/bin/python`.
- **`~/envs/iegs`** (py3.12, torch 2.13/cu130): IncEventGS only. Before running anything
  in it:
  ```bash
  export CUDA_HOME=/usr/local/cuda-13.1 PATH=$HOME/envs/iegs/bin:/usr/local/cuda-13.1/bin:$PATH TORCH_CUDA_ARCH_LIST=12.0
  ```
  gsplat is JIT-compiled and cached (`~/.cache/torch_extensions`), so only the first
  import is slow.
- Both 3DGS codebases open a viewer socket on port 6009. Two trainings at once
  collide (`OSError: Address already in use`); give the second one `--port 6019`
  or any other free port.

### 10.3 Finishing the IncEventGS run (in progress at time of writing)

Check on it:
```bash
pgrep -f inceventgs_lego.yaml >/dev/null && echo running || echo stopped
tr '\r' '\n' < lego/out_inceventgs/run.log | grep '^Tracking frame' | tail -1     # current frame (of ~200)
tr '\r' '\n' < lego/out_inceventgs/run.log | grep -A5 Traceback | tail -6         # any crash
```

When it stops cleanly:
1. **Final poses.** Look in `lego/out_inceventgs/lego/global_BA/` for the newest
   `pose_file_est_*` / `pose_file_gt_*` pair and its `*_pose.png`. The ATE in the png
   title is IncEventGS's own Sim3-aligned number; §7.5 shows the same computation
   done by hand. Compare it with the 43 cm pre-global-BA figure.
2. **Render scores.** Run
   `$P score_pairs.py lego/out_inceventgs/lego/img_eval/est lego/out_inceventgs/lego/img_eval/gt --json lego/out_inceventgs/scores.json`.
   Use `--by-order` if est and gt filenames differ. This reports the same gray raw and
   tone-fitted PSNR / SSIM / LPIPS as `eval_splat.py --gray [--affine]`; it was checked
   against `eval_splat.py` on three baseline views (15.27 vs 15.28 dB).
3. Put the numbers into §7.5 and the §7.3 comparison table.

**If it crashed or was killed, it has to start over.** `save_model()` / `load_model()`
exist in `main.py` but `run()` never calls them, so there is nothing to resume from.
Rerun the §9 command; it overwrites `out_inceventgs/lego/`, so move the old one aside
first if you want to keep it. A full run is about 3 h. To try settings faster, set
`training_end_index` in `inceventgs_lego.yaml` to about 61 (60 frames = 300 ms, ~100°
of orbit); that also lets you check drift sooner.

### 10.4 Continuing Event-3DGS training

`--start_checkpoint` loads the Gaussians but **restarts the iteration counter, the
learning-rate schedule and densification at 0**. So "continue" here always means a new
warm start; the old run can't be extended in place.

- **The best configuration (§7.3) was still improving at 8 k.** Rerun it from the same
  gray checkpoint with more iterations, not from its own `chkpnt8000.pth`. Chaining
  checkpoints would restart the learning rate high a second time.
  ```bash
  cd external/Event-3DGS; L=$(realpath ../../work/eventnerf/lego)
  ~/envs/phase1/bin/python train.py -s $L/scene_e2vid/e3dgs_real -m $L/out_e3dgs_real_ft16k \
    --gray --event --holdout 8 --iterations 16000 \
    --start_checkpoint $L/out_e3dgs_gray/chkpnt30000.pth \
    --save_iterations 3999 7999 11999 16000 --checkpoint_iterations 16000 --quiet
  cd ../../work/eventnerf
  for i in 3999 7999 11999 16000; do
    $P eval_splat.py lego/scene_e2vid/rgb_eval lego/out_e3dgs_real_ft16k $i --holdout 1 --gray --affine --tag _rgbaff
  done
  ```
  Each 8 k block takes about 3 min to train plus about 2 min to score 189 views.
- **Repeat runs need a seed patch first.** `utils/general_utils.py:safe_state()` hard-codes
  `random.seed(0)`, `np.random.seed(0)` and `torch.manual_seed(0)`, so reruns differ only
  by CUDA nondeterminism. Add a `--seed` argument to `train.py`, pass it into
  `safe_state`, and run 3 seeds of §7.3 and of the `--gray` source so the +1.4 dB has
  an error bar.
- **Don't train Event-3DGS `--event` from scratch.** It collapsed every time (§3, §7.1).
  Always warm-start from a `--gray` checkpoint.
- **Any new scene follows the same pattern:**
  1. undistorted COLMAP model + E2VID frames in `gs/`;
  2. `integrate_events.py --names <gs inputs> --stamps <e2vid stamps> --no-init --alpha 3`;
  3. `colmap image_undistorter` with the same sparse model;
  4. a scene folder with `images` → E2VID and `images_event` → the integrated frames;
  5. `--gray` 30 k, then `--gray --event` warm start.

### 10.5 Moving to a ThinkCam capture

The §7.3 recipe needs four things from a real recording:
1. **E2VID frames + COLMAP poses.** Phase 1b has these for the Sep 9/10 captures
   (`work/sep9_e2vid`, `work/sep10_e2vid`, `work/phase1b`); check which take
   registered best.
2. **Event frames at the E2VID timestamps.** `integrate_events.py` currently reads
   EventNeRF `.npz` with millisecond-binned `t`, an RGGB Bayer mosaic and threshold 0.25.
   A ThinkCam version must instead:
   - read `events.h5` (microsecond `t`, `p` in {0, 1} → ±1, per
     `pipeline/convert_to_inceventgs.py`);
   - skip demosaicing (mono sensor);
   - apply the leak per event, not per 1 ms bin.

   The threshold is unknown, but Event-3DGS's learnable `c` absorbs the scale, so a
   nominal 0.2–0.3 should do.
3. **Undistortion** of those frames through the same COLMAP camera.
4. **Scoring.** There is no RGB ground truth, so use held-out E2VID views
   (`eval_splat.py` default) plus side-by-side renders. §3 showed that metric
   *penalises* the event loss, so read it together with the images.
