# ThinkCam Phase 1b — Motion-Compensated Event Frames

> **Built and measured.** Tools: `accumulate_frames.py`, `frame_metrics.py`.
> Results, including which gates passed and what the failures diagnose:
> [`Phase1b-Results.md`](Phase1b-Results.md).

This plan replaces the **event-to-video stage** of [`Phase1.md`](Phase1.md) §5 Step 3.
Everything else in that plan stands: the exporter, COLMAP self-calibration, the upstream
3DGS baseline, and the acceptance gates are unchanged. What changes is how
`events.h5` becomes intensity frames — pretrained E2VID was measured and found unusable
for this sensor, for a reason that no parameter can reach. This document specifies the
non-learned replacement.

---

## 0. Why E2VID was abandoned

The decisive experiment: feed E2VID the **same events** twice, differing only in what
preceded them (one stream carrying 20 s of prior history, one starting fresh at the same
instant). A scene-driven reconstructor returns identical images.

| frame | MAE (0-255) | correlation | SIFT matches between the two |
|---|---|---|---|
| 20 | 30.6 | 0.109 | 6 |
| 60 | 24.5 | 0.054 | 2 |
| 100 | 17.0 | 0.242 | 3 |
| 140 | 14.7 | 0.580 | 0 |

Identical input, near-uncorrelated output. E2VID's checkpoint is `E2VIDRecurrent` with
ConvLSTM blocks (`external/rpg_e2vid/model/model.py:82-99`) carrying hidden state across
windows, and on this data **the state dominates the events**.

That single mechanism explains every failure observed:

| Observation | Explanation |
|---|---|
| COLMAP mean track length **2.0** (`work/phase1/wide`) | The same scene point looks different at different times as state evolves, so features cannot persist past two views. |
| Median feature flow **~0.2 px** over 0.4 s while raw events move **353 px / 0.5 s** | SIFT was locking onto the LSTM state's imprint, which is anchored in image coordinates and drifts independently of camera motion. |
| COLMAP labelling 276-1342 pairs `WATERMARK` | That is COLMAP's category for features that do not move between images — the state imprint, precisely. |
| Three configurations failed identically (1280×720, 320×180, 427×240 +CLAHE) | Window size, resolution and contrast do not touch hidden state. |
| E2VID reconstructs its **own** reference data perfectly | Reference density is 10.23 ev/px/s; ours is 0.87. With dense events the events drive the output and state is a minor refinement. At 12× sparser, state wins. |

**Root cause, one sentence:** our event density is too low for the scene signal to
overcome E2VID's recurrent state, so its frames are not a function of the scene and
cannot support multi-view geometry.

Two independent fixes follow. Raising event density at the sensor is covered in
[`CaptureGuide.md`](CaptureGuide.md) and needs a new capture. **This document covers the
other one: a reconstruction method with no hidden state**, which works on the takes
already recorded.

---

## 1. The approach

Accumulate events into an image directly, warping them by an estimated motion so edges
add up sharp instead of smearing. Two properties matter, and both are things E2VID lacks:

```
Deterministic  ... output depends ONLY on the events inside the window.
                   Same scene, same viewpoint -> same appearance -> features track.
Scale-free ..... no training distribution, so 1280x720 is not out-of-regime.
```

The technique is contrast maximization (Gallego et al., *A Unifying Contrast Maximization
Framework for Event Cameras*): warp events by a candidate motion, accumulate, and score
the result's sharpness. The correct motion is the one that maximises it.

```
events window (x, y, t, p)
  │  warp:  x' = x - (t - t_ref) * v          v = per-window motion parameters
  ▼
accumulate into H x W image
  │  score: Var(image)                        sharp = high variance
  ▼
optimise v  ->  motion-compensated frame
```

---

## 2. Build order

Cheapest first. **Step 1 may be sufficient on its own — measure before building Step 2.**

### Step 1 — Plain short-window accumulation

No motion estimation at all. Short enough windows that motion is small, accumulated
directly. This is already known to show real scene structure: at 20,000 events (22 ms)
the raw accumulation shows a chair, a doorway and wall edges clearly, where every E2VID
frame at every setting showed none.

New `accumulate_frames.py` at the repo root, beside `export_e2vid_input.py`:

- Reuse `event_chunks()` from `export_e2vid_input.py:85` for the streaming read, and
  `_resolve_input` from `convert_to_inceventgs.py:47-54` so the CLI matches the others.
- Window by **event count** (`--events-per-frame`), sliding with `--stride` for overlap.
- Accumulate, normalise, write 3-channel PNG (Phase1 §3.3 — a 1-channel ground truth
  broadcasts silently against a 3-channel render).
- Run the §5 metrics. If mean track length exceeds 3 and COLMAP registers, **stop here**.

### Step 2 — Contrast maximization

Only if Step 1's frames are too motion-blurred. Add a `--motion-comp` flag:

- **Model:** start with 2-DOF global translation `(vx, vy)`. It is the correct first
  model — the dominant term over a short window — and has a cheap, robust search.
- **Objective:** image variance of the accumulated frame. Maximise.
- **Optimiser:** coarse grid over `(vx, vy)` then local refinement. With 2 parameters this
  beats gradient descent for robustness and needs no differentiable accumulation. Move to
  a torch/GPU differentiable warp only if the search proves too slow.
- **Warm start** each window from the previous window's solution — camera motion is
  continuous, and this is a large speedup.

### Step 3 — Feed the existing pipeline

Unchanged from [`Phase1.md`](Phase1.md) §5 Steps 4-5: frames into `<scene>/input/`, then
`convert.py`, then `train.py`/`render.py`/`metrics.py`.

---

## 3. Design decisions

**Accumulate event COUNT, not signed polarity.** This is the load-bearing decision and it
is not obvious. An edge crossed left-to-right fires the opposite polarity to the same edge
crossed right-to-left. On an orbit the camera passes edges in different directions, so a
**signed** accumulation makes the same physical edge appear bright in one view and dark in
another — destroying exactly the multi-view consistency this whole approach exists to
provide. Accumulate `|events|` per pixel. ThinkCam stores `p` in `{0,1}`
(`thinkcam/raw_recorder.py:96`), so simply ignore the column.

**Normalise identically across every frame.** Per-frame min/max stretching makes
appearance depend on window content, reintroducing view-dependence through the back door.
Use a **fixed** mapping — a global percentile pair computed once over the take, then
applied unchanged to every frame.

**Window length is the central trade-off**, and unlike E2VID there is no required
events-per-pixel: too short is noisy, too long is motion-blurred, and Step 2 exists
specifically to push the usable window longer. Sweep `--events-per-frame` over
20k / 50k / 100k / 200k and pick on the §5 metrics, not by eye — judging event
reconstructions visually was wrong twice during Phase 1.

**Full resolution.** Motion compensation removes the smear that forced downsampling, and
1280×720 yields far more features than 320×180 did. No `--downsample` needed.

**Overlapping windows are allowed.** Unlike E2VID's sequential state, windows here are
independent, so a sliding stride costs only compute and gives COLMAP more frames and
longer tracks.

---

## 4. `VERIFY-FIRST` items

**4.1 `VERIFY-FIRST` — does a single global motion suffice?** Contrast maximization with
one motion per window assumes the whole image moves together. An orbit around a subject
produces **depth-dependent** parallax — which is the very signal SfM needs. A single global
`v` will sharpen the dominant depth and smear the rest. Measure the residual: if
foreground and background cannot both be sharp, either shorten the window until the
approximation holds, or estimate motion per tile and warp piecewise. **Do not assume the
2-DOF model is adequate — confirm it on a real window before building on it.**

**4.2 `VERIFY-FIRST` — is Step 2 needed at all?** Step 1 might pass the gates unaided.
Run the §5 metrics on plain accumulation first and only build the optimiser if they fail.

---

## 5. Acceptance tests

Ordered so the cheapest disqualifier runs first.

- **Determinism (the test E2VID failed):** reconstruct a window twice from streams with
  different preceding history. Output must be **bit-identical**. Any deviation means state
  or normalisation has leaked in. This is a hard gate, not a metric.
- **Sharpness:** image variance after compensation ≥ before (Step 2 only). If not, the
  motion model is wrong.
- **Multi-view consistency:** SIFT matches between frames ~0.4 s apart, and — the number
  that actually predicts SfM — **median feature flow must be non-zero and scale with time
  separation**. E2VID sat at ~0.2 px regardless of gap; anything scene-locked will not.
- **Homography/fundamental inlier ratio** below ~0.9, indicating real parallax rather than
  degenerate geometry.
- **The gate:** COLMAP registers ≥90% of frames with **mean track length > 3**. Track
  length is the metric that exposed the E2VID failure (it was exactly 2.0) and is the
  single best indicator that features persist across views.
- **Smoke first:** `20260602_101754_demo_scene_test_orbit_1` (5.3 s) end-to-end before
  the 64 s take.

---

## 6. Risks

- **Depth-dependent parallax breaks the global motion model** (§4.1). The most likely
  failure, and the one to measure first.
- **Sparse events may simply be too sparse.** At 0.87 ev/px/s a 20 ms window touches ~2%
  of the sensor. Motion compensation sharpens what is there; it cannot manufacture edges.
  If Step 1 and Step 2 both fail the §5 gates, the honest conclusion is that this take
  lacks the density to reconstruct, and the fix is at the sensor — not further post-processing.
- **Compute.** A per-window search over 2,500 windows is real work. Warm-starting and a
  coarse-to-fine grid should keep it to minutes; if not, move the warp to the GPU.

---

## 7. Open decisions for the operator

1. **Try this before re-capturing, or in parallel?** This plan works on takes already
   recorded; the bias changes need a session. They are independent and both worth doing.
2. **How much implementation before falling back to a new capture?** Step 1 is small.
   Step 2 with a tile-wise model (§4.1) is substantially more. Suggest gating Step 2 on
   Step 1's measured numbers.
3. **If this works, does Phase 2 still need event-to-video?** The report's Phase 2 front
   end (EventSplat) assumes event-to-video initialization and would inherit the same
   recurrent-state problem. Worth resolving before Phase 2 is planned in detail.

---

## Checklist per attempt

- [ ] Determinism gate passed — same events, different history, **bit-identical** output.
- [ ] Accumulated **unsigned** event counts, not signed polarity.
- [ ] Normalisation is **global**, not per-frame.
- [ ] Window length chosen on **measured metrics**, not visual inspection.
- [ ] Median feature flow is **non-zero** and grows with time separation.
- [ ] COLMAP mean track length **> 3** (E2VID achieved exactly 2.0).
- [ ] Smoke take passed end-to-end before the 64 s run.
