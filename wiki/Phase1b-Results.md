# Phase 1b — Measured Results

What [`Phase1b-MotionCompensation.md`](Phase1b-MotionCompensation.md) specified, built and
run against `20260602_102004_demo_scene_orbit_1` (51.2 M events, 64.1 s, 798 Kev/s).

Two tools, both in `pipeline/` beside the existing converters:

| Tool | Does |
|---|---|
| `pipeline/accumulate_frames.py` | events → frames. Step 1 (plain accumulation) and Step 2 (`--motion-comp`, contrast maximization, optional `--mc-tiles`). |
| `pipeline/frame_metrics.py` | every Phase1b §5 gate: `determinism`, `sharpness`, `flow`, `colmap`, plus `mc-probe` for the Phase1b §4.1 `VERIFY-FIRST`. `all` runs them cheapest-first. |

**Headline:** the frames are deterministic — the gate E2VID failed — and the motion model
was verified rather than assumed. But the multi-view gate fails, and it fails for a reason
that no reconstruction method can reach: **at ~1 event per active pixel per window, local
image appearance is not repeatable between windows.** SIFT correspondences between windows
that share no events are indistinguishable from noise at every window length tested, 20 k
to 1.6 M events, with and without compensation — random flow statistics throughout, and
zero matches agreeing with an independently measured shift at 200 k and above. This is
Phase1b §6's second risk, confirmed with numbers: the fix is at the sensor.

---

## 1. Gate results

| Phase1b §5 gate | Step 1 (plain) | Step 2 (`--motion-comp`) |
|---|---|---|
| **Determinism** (bit-identical, different history) | **PASS** — 25/25 windows @ 50 k, 8/8 @ 200 k | **PASS** cold; **FAIL** warm-started (see §3) |
| Sharpness (variance ≥ baseline) | — | **PASS** — ×1.06 median on the raw window, ×1.29 on the fastest frames |
| SIFT matches ≥ 30/pair | **FAIL** — median 4–17 | **FAIL** — median 12–18 |
| Median flow scales with gap | FAIL at ≤100 k, PASS at 200 k | **FAIL** — 383 px @0.16 s → 428 px @0.64 s |
| H inlier ratio < 0.9 | PASS (0.27–0.65) | **PASS** (0.27–0.33) |
| **COLMAP** ≥90% registered, track > 3 | **FAIL** — 2/238, track 2.00 | **FAIL** — 3/238, track 2.67 |

The flow and inlier "passes" are hollow and the §7 diagnosis says why: the matches those
numbers are computed from are spurious. A median flow of ~400 px on a 1280-wide image is
what uniformly random matching produces — and the flow gate flips between PASS and FAIL
depending on which frame pairs are sampled, which is itself the signature of a statistic
computed over noise. Read the COLMAP row; it is the one that does not flip.

## 2. The window-length sweep

Phase1b §3 asks for 20 k / 50 k / 100 k / 200 k picked on measured numbers. Held constant:
250 frames over the same 20 s segment (`--stride 64000`), so the frame *rate* is fixed and
only the exposure varies — otherwise window length and frame count move together and
neither can be attributed.

| events/frame | exposure | SIFT keypoints/frame | median matches/pair | verdict |
|---|---|---|---|---|
| 20 k | 25 ms | 588 | 4 | too sparse to detect features |
| 50 k | 62 ms | 2,593 | 6 | — |
| 100 k | 125 ms | 6,372 | 10 | — |
| 200 k | 250 ms | 8,049 | 17 | best of a failing set |

Keypoint count rises with window length throughout, and match count with it, but no
setting reaches the ≥30 gate and none of the matches survives §7's consistency check. 200 k
is used for the COLMAP runs as the most favourable case, not as a passing one.

## 3. Determinism caught a real defect — in this implementation

Phase1b §2 Step 2 recommends warm-starting each window's search from the previous
window's solution ("a large speedup"). It is, and it is also **E2VID's defect in
miniature**: the frame stops being a function of its own events. The gate caught it
immediately — same events, different preceding history, **MAE 27/255**, the same order as
E2VID's own 15–31.

`--mc-warm-start` therefore exists but is **off by default**, and the runtime banner says
which mode is active. Cold search costs ~1.7× (81 s vs 46 s for 238 frames at 200 k
events) and passes the gate bit-exact. Per-tile motion keeps its seed from the *same*
window's global fit, so tiles do not reintroduce the dependence.

## 4. The optimiser needed a fix the plan did not anticipate

A plain coarse-to-fine grid over `(vx, vy)` — which is what Phase1b §2 Step 2 describes — returns
`v ≈ 0` with a variance gain of ×1.00 on this data. That looks exactly like a static
camera, so it is not self-announcing.

The cause: **the objective's peak is only as wide in velocity as an edge is in pixels.**
A 1 px edge over a half-window of 0.12 s is a peak ~8 px/s across, while the coarse level
steps 500 px/s — the search steps straight over it, every time. The fix is to make each
level's *spatial* bin as wide as that level's step displaces an event at the window edge,
so the peak is as broad as the grid is coarse. With binning, the same windows solve to
500–2200 px/s at gains up to ×2.24. The velocities were then confirmed independently:
brute-force NCC between two disjoint windows 213 ms apart peaks at (157, −161) px, which
is 1050 px/s in the same direction the contrast search reported.

## 5. Phase1b §4.1 `VERIFY-FIRST` — one global motion is enough

`pipeline.frame_metrics mc-probe`, 100 k-event windows, six samples across the take:

```
        t_mid_s   v_global(px/s)      gain     +3x3 tiles   tile v spread
        0.05      (    23,  -852)     x1.45      x1.03          42 px/s
        12.13     (   273,   -23)     x1.10      x1.06         105 px/s
        25.95     (  1549,  -586)     x1.28      x1.02         364 px/s
        37.80     ( -1078,    23)     x1.01      x1.01         295 px/s
        51.53     (   766,  -609)     x1.19      x1.09         263 px/s
        64.06     (   195,   395)     x1.19      x1.02          87 px/s
```

Per-tile motion adds **×1.04** over a single global `v`. At this window length the 2-DOF
model is adequate and the piecewise path (`--mc-tiles`) is not needed — measured, not
assumed. What does *not* hold is constant velocity over a **long** window: at 455 ms the
coarse-scale optimum makes full-resolution variance *worse* (×0.97), so the window length,
not the spatial model, is the binding constraint.

## 6. COLMAP

238 frames, 200 k events/frame, 20 s segment, sequential matcher, same feature settings as
the Phase 1 attempts in `work/phase1/*/run_colmap.sh`. `work/` is gitignored, so the
commands are written out under **Reproducing** rather than only referenced.

| Frames | Registered | Mean track length | Points |
|---|---|---|---|
| Plain accumulation (`work/phase1b/sw_200000`) | **2 / 238 (1%)** | **2.00** | 103 |
| Motion-compensated (`work/phase1b/mccold_200000`) | **3 / 238 (1%)** | **2.67** | 6 |
| *E2VID, for reference (Phase1b §0)* | *—* | *2.0* | *—* |

Compensation moves the reconstruction from two images to three. The gate wants 214 and a
track length above 3.

The match database says why, and it is not a shortage of features:

| | keypoints/image | pairs with any match | median inliers on verified pairs |
|---|---|---|---|
| Plain | 15,073 | 524 / 1,649 | 46 |
| Compensated | 14,401 | 649 / 1,649 | 17 |

Roughly 15 k keypoints per frame and a third of pairs "verified" — yet the mapper cannot
grow past three images. Splitting the verified pairs by how far apart the two frames are
(stride 64 k inside a 200 k window, so neighbours share 68% of their events) shows why:

| frame gap | event overlap | plain: pairs / median inliers | compensated: pairs / median inliers |
|---|---|---|---|
| 1 | 68% | 104 / **386** | 60 / 17 |
| 2 | 36% | 76 / **318** | 61 / 17 |
| 3 | 4% | 57 / **323** | 44 / 17 |
| 4–15 | 0% | 204 / 32 | 258 / 17 |
| 16+ | 0% | 83 / 16 | 226 / 16 |

For plain accumulation the only strong pairs are the ones that **share events** — 300+
inliers where the frames are near-duplicates, collapsing to 16–32 (RANSAC's noise floor)
the moment they do not. Near-duplicate pairs carry no baseline, so they cannot triangulate;
everything else is noise. There is no rung in between, and that is what stalls the mapper.

Compensation flattens the table to ~17 everywhere, including the overlapping pairs. That
looks like a regression and is not: each window is warped by its own velocity, so two
overlapping windows no longer share a coordinate frame and their *artificial*
zero-baseline similarity disappears. What is left is the honest signal — and it sits at
the noise floor, which is §7 in one sentence.

## 7. Why it fails, and why more post-processing will not fix it

Six measurements, all pointing the same way:

**Compensation works, and it is not enough.** Warping by the solved velocity raises
full-window variance ×1.06 (median, up to ×1.29 at 3550 px/s) and lifts the share of
events landing on a multiply-hit pixel from 27% to 33% — real sharpening, visible in the
frames. It moves COLMAP from 2 registered images to 3.

**Local appearance does not repeat.** Two frames that share **84%** of their events
correlate at only **0.04** per-pixel (0.45 after σ=3 blur). Each window's events are a
Poisson sample of the edge set, so two windows illuminate *different* pixels of the same
edge.

**Global structure does repeat.** Brute-force NCC between windows sharing **no** events
peaks cleanly at 0.24–0.58, consistently across scales, at the shift the motion search
predicts. The geometry is in the data; the pixels just do not carry it locally.

**SIFT correspondences are noise.** Between disjoint windows, the match-displacement
histogram is flat — the modal 20 px bin holds 3 of 640 matches at ratio 0.9, 6 of 2403 at
0.95. Zero matches agreed with the NCC-measured shift, at 200 k / 400 k / 800 k / 1.6 M
events per window, compensated or not.

**Blur buys matches, not correspondences.** σ=1 raises median matches 9 → 39 (plain) and
17 → 59 (compensated) while the fundamental-matrix inlier ratio *falls* 0.71 → 0.27. More
matches, all spurious. It is not added to the tool for that reason.

**Longer exposure does not raise density where it matters.** A trajectory-compensated 2 s
exposure (32 slices registered by FFT cross-correlation and integrated) reaches 73% sensor
coverage but still only **1.95 events per active pixel** — because the camera is moving,
extra time paints a wider swath rather than building up the same pixels. SIFT still found
1 consistent match out of 37.

That last one is the crux. Per-pixel event count is set by **how many events one edge
crossing fires**, which is a bias/contrast-threshold property of the sensor — not
something any accumulation window, warp, or filter can increase after the fact. Phase1b §6
predicted exactly this, and it is now measured rather than suspected.

## 8. A note on the smoke take

`20260602_101754_demo_scene_test_orbit_1` (5.3 s) is **structureless** — accumulations at
20 k–400 k events at six positions across the take show noise and a dead left band, no
scene content at all. It still works as a mechanics smoke test (it is what the accumulator
was first run against), but it cannot serve as the Phase1b §5 geometric smoke gate. All numbers
here are from the 64 s take.

## 9. What to do next

1. **Re-capture with the bias changes** in [`CaptureGuide.md`](CaptureGuide.md). That is
   the only lever that raises events per edge crossing, which is the quantity every gate
   above is starved of. Target: enough that a window has ≥5 events per active pixel.
2. **Re-run the gates on the new take, unchanged.** Both tools are take-agnostic:
   `pipeline.accumulate_frames --input <rec> --out <scene>/input` then
   `pipeline.frame_metrics all --frames <scene> --input <rec> --scene <scene>`. The cheapest
   disqualifier runs first, so a bad take is rejected in seconds rather than after COLMAP.
3. **Keep the determinism gate in front of any future front end.** It cost minutes, it
   caught this implementation's own warm-start defect, and it is the test that would have
   disqualified E2VID before three configurations were run against it.
4. **Phase1b §7.3 stands.** Phase 2's EventSplat front end assumes event-to-video
   initialisation and would inherit the same problem; resolve before planning it.

---

## Reproducing

```bash
V=~/envs/phase1/bin/python          # numpy, h5py, opencv, torch; colmap on PATH
REC=recordings/20260602_102004_demo_scene_orbit_1

# Step 1 — plain accumulation
$V -m pipeline.accumulate_frames --input $REC --out work/phase1b/sw_200000/input \
    --events-per-frame 200000 --stride 64000 --start-s 10 --duration-s 20

# Step 2 — contrast maximization (cold search; add --mc-warm-start only for previews)
$V -m pipeline.accumulate_frames --input $REC --out work/phase1b/mccold_200000/input \
    --events-per-frame 200000 --stride 64000 --start-s 10 --duration-s 20 --motion-comp

# Phase1b §5 gates, cheapest first
$V -m pipeline.frame_metrics all --frames work/phase1b/mccold_200000 --input $REC \
    --gaps 0.16 0.32 0.64
$V -m pipeline.frame_metrics mc-probe --input $REC -n 100000 --samples 6 --tiles 3x3

# The COLMAP gate. Same feature settings as the Phase 1 attempts in
# work/phase1/*/run_colmap.sh, so the numbers compare with the E2VID run.
# (work/ is gitignored, so this is written out rather than only referenced.)
S=work/phase1b/mccold_200000
colmap feature_extractor --database_path $S/db.db --image_path $S/input \
    --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV \
    --SiftExtraction.peak_threshold 0.002 --SiftExtraction.max_num_features 16384 \
    --SiftExtraction.use_gpu 0
colmap sequential_matcher --database_path $S/db.db \
    --SequentialMatching.overlap 15 --SequentialMatching.quadratic_overlap 1 \
    --SiftMatching.use_gpu 0
mkdir -p $S/sparse && colmap mapper --database_path $S/db.db \
    --image_path $S/input --output_path $S/sparse \
    --Mapper.init_min_tri_angle 2 --Mapper.init_min_num_inliers 30 \
    --Mapper.abs_pose_min_num_inliers 15 --Mapper.min_num_matches 12 \
    --Mapper.filter_min_tri_angle 0.5 --Mapper.init_max_error 6
$V -m pipeline.frame_metrics colmap --scene $S
```

Runtimes on this machine: accumulation 2.5 s for 238 frames, motion-compensated 81 s
(46 s with `--mc-warm-start`, which fails the determinism gate); COLMAP ~13 min per scene
on CPU SIFT.
