# VGGT vs COLMAP — EventNeRF Lego

Camera poses on the EventNeRF synthetic lego truck, from two frame sources, solved two
ways, and scored against ground truth. COLMAP runs are from 2026-09-24; VGGT runs from
2026-09-28.

**Headline:** on RGB frames the two are equivalent (COLMAP 0.34–0.50 cm ATE, VGGT 0.61 cm).
On E2VID reconstructions of the event stream, COLMAP holds up (1.64 cm) and **VGGT fails
outright (65–71 cm on a scene 2.55 m across)**. It fails because the trajectory it predicts has the
wrong shape, not because it drifts: the error is 25–45 cm even inside any one third of the
orbit. For event data, COLMAP on E2VID frames is still the pose source to use.

---

## 1. Data

| | |
|---|---|
| Scene | EventNeRF synthetic `lego`, 346×260 sensor |
| Events | 2,350,774 over 1.0 s (`work/eventnerf/lego/events.txt`) |
| Trajectory | one full 360° orbit, radius 0.9 m (`gt.txt`); bounding-box diagonal 2.55 m, arc length ≈ 5.65 m |
| True intrinsics | fx = fy = 480.55, cx = 173, cy = 130 (`K.txt`) |
| RGB frames | 201 rendered views, `scene_rgb/input` (`rgb_stamps.txt`) |
| E2VID frames | 196 E2VID reconstructions, `scene_e2vid/input` (`e2vid_stamps.txt`) |

`to_e2vid.py` converts the EventNeRF split into E2VID input plus `gt.txt` camera centres on
the same clock as the events.

## 2. Metric

**ATE:** each solved camera centre is paired with the GT centre interpolated at the frame's
timestamp, the solved trajectory is Umeyama Sim3-aligned to GT, and the RMSE of the
residual is reported. The % figure divides by the GT bounding-box diagonal (2.55 m), which
`eval_poses.py` labels "GT path"; it is not the arc length. COLMAP models are scored by
`eval_poses.py`; VGGT by `vggt_poses.py`, which reuses the same alignment code line for
line. "Solved focal" is the recovered fx in native 346-px pixels.

## 3. Results

| Frames | Method | Registered | ATE rmse | ATE max | Solved fx (true 480.55) |
|---|---|---|---|---|---|
| RGB | COLMAP `simple` (SIMPLE_RADIAL) | 201/201 | **0.34 cm** (0.13%) | 0.70 cm | 490.1 |
| RGB | COLMAP `auto` (OPENCV) | 201/201 | 0.50 cm (0.19%) | 1.22 cm | 484.6 |
| RGB | VGGT, stride 4 | 51/51 | **0.61 cm** (0.24%) | 1.74 cm | 476.5 |
| E2VID | COLMAP `simple` (SIMPLE_RADIAL) = `gs` | 191/196 | **1.64 cm** (0.64%) | 6.23 cm | 520.6 |
| E2VID | COLMAP `calib` (K fixed to GT) | 190/196 | 2.73 cm (1.07%) | 10.48 cm | 480.55 (fixed) |
| E2VID | COLMAP `auto` (OPENCV) | split: 99 + 94 | 28.5 / 33.9 cm | 53 / 63 cm | 636 / 160 |
| E2VID | VGGT, stride 4 | 49/49 | 71.5 cm (28.1%) | 118.7 cm | 416.1 |
| E2VID | VGGT, stride 3 | 66/66 | 65.2 cm (25.6%) | 144.9 cm | 412.0 |

VGGT's "registered" column is always all of its inputs: it is feed-forward and returns a
pose for every frame whether or not the pose is any good.

### COLMAP details

All from `sfm.sh` (CPU SIFT, exhaustive matching, incremental mapper, single shared
camera), largest sub-model scored.

| Model | Points | Mean track | Mean reproj |
|---|---|---|---|
| RGB `simple` | 10,971 | 8.87 | 0.46 px |
| RGB `auto` | 10,885 | 8.94 | 0.45 px |
| E2VID `simple` | 8,393 | 6.26 | 0.65 px |
| E2VID `calib` | 8,547 | 6.24 | 0.65 px |
| E2VID `auto` (larger half) | 3,486 | 6.41 | 0.74 px |

- **Camera model matters on E2VID, not on RGB.** With the 8-parameter OPENCV model the
  E2VID reconstruction splits into two half-orbits with nonsense focals (636 and 160); with
  SIMPLE_RADIAL it stays in one piece. The distortion terms have too much freedom for the
  noisier E2VID matches.
- **Fixing K to the truth made E2VID worse** (2.73 vs 1.64 cm). SIMPLE_RADIAL settles on
  fx = 520.6, 8% long, and still wins — it is absorbing something about how E2VID frames
  are formed that the true pinhole K doesn't model.
- E2VID loses 5–6 frames that RGB registers, and tracks are ~30% shorter (6.3 vs 8.9).

### 3DGS on the COLMAP poses

Gaussian splatting (`external/gaussian-splatting`, 30 k iterations, `--eval` holding out
every 8th image → 24 test views) trained on the undistorted `gs` models:

| Source | Poses from | Test PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| RGB | RGB `simple` | 40.28 | 0.993 | 0.007 |
| E2VID | E2VID `simple` | 33.36 | 0.954 | 0.109 |

The E2VID test images are themselves E2VID frames (`out_e2vid/test/ours_30000/gt`), not
RGB renders, so this measures how self-consistent the splat is with the E2VID frames. It
does not measure fidelity to the true scene, and the two rows are not directly comparable.

## 4. VGGT details

`facebook/VGGT-1B`, bf16, RTX 5080 16 GB. Frames go through VGGT's own loader (width →
518, so 346×260 → 518×392); intrinsics are scaled back to native pixels.

| Run | Frames | Inference | Peak VRAM |
|---|---|---|---|
| E2VID stride 4 | 49 | 5.1 s | 10.0 GiB |
| E2VID stride 3 | 66 | 7.9 s | 10.8 GiB |
| RGB stride 4 | 51 | 5.5 s | 10.1 GiB |

Memory limits it to roughly 70 frames per pass on this card (the VGGT README lists
21 GB at 100 frames), so all runs subsample. More frames did not help E2VID: stride 3
is only marginally better than stride 4.

### Why E2VID fails

- **Not an evaluation bug.** The identical scoring path gives 0.61 cm and a focal within
  1% on RGB.
- **The shape is wrong.** GT is a circle. The spread of camera distance from the
  trajectory centroid (std/mean) is 0.02 for VGGT-on-RGB and **0.32** for VGGT-on-E2VID.
- **It is not drift or a missed loop closure.** Aligning each third of the orbit
  independently still leaves 44.9 / 25.4 / 38.8 cm local ATE (RGB: 0.4 / 0.7 / 0.4 cm).
  Error is lowest near 90° into the orbit (8–13 cm) and worst at the loop ends (up to
  145 cm).
- **Focal is 14% short** (412–416 vs 480.55), in the opposite direction from COLMAP's 8%
  long, which is further evidence that the geometry it infers from these frames is
  inconsistent.

The likely cause is domain gap: E2VID frames are grayscale, low-contrast, carry
reconstruction artifacts and smear, and nothing like them is in VGGT's training data.
COLMAP only needs locally repeatable SIFT keypoints, which E2VID frames still provide
(§3), while VGGT's learned prior depends on how the whole image looks.

## 5. Where next

- **VGGT → COLMAP BA** (`demo_colmap.py --use_ba`, which re-tracks with VGGSfM and bundle
  adjusts): worth one run, but it starts from poses that are 25–45 cm wrong locally, so
  it may converge to the wrong basin.
- **VGGT on RGB is a viable fast path** where RGB exists: ~5 s vs ~1–2 min CPU COLMAP,
  at 0.6 cm. It does not remove the need for an event-native pose source.
- COLMAP `simple` on E2VID (1.64 cm) remains the reference for event-only poses.

## 6. Reproduce

From `work/eventnerf/`, with `~/envs/phase1/bin/python` (VGGT cloned to
`work/vggt/repo`, installed `pip install --no-deps -e .` to keep torch 2.13/cu130):

```bash
# COLMAP (CAM_MODEL defaults to OPENCV = `auto`; `simple` is SIMPLE_RADIAL per the saved
# cameras.bin — exact invocations weren't logged, `calib` also fixed/unrefined K)
CAM_MODEL=SIMPLE_RADIAL ./sfm.sh lego/scene_e2vid simple
python eval_poses.py lego/scene_e2vid/simple/sparse/2 lego/e2vid_stamps.txt lego/gt.txt

# VGGT
python vggt_poses.py lego/scene_e2vid/input lego/e2vid_stamps.txt lego/gt.txt lego/vggt_e2vid_s3 --stride 3
python vggt_poses.py lego/scene_rgb/input   lego/rgb_stamps.txt   lego/gt.txt lego/vggt_rgb_s4   --stride 4
```

Each VGGT run writes `poses.txt`, `points.ply` (confidence-filtered depth unprojection)
and `summary.json` (ATE, per-frame error, intrinsics, timing) to its output dir.
