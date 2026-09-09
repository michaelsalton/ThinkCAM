#!/usr/bin/env python3
"""LK feature tracks over accumulated event frames -> COLMAP correspondences.

This replaces COLMAP's own SIFT stage for event frames, and the reason is
measured (docs/Phase1b-Results.md §7): between two windows that share no
events, SIFT descriptor matching produces NOTHING. The match-displacement
histogram is flat, zero matches agree with an independently measured shift, and
COLMAP registered 2 of 238 frames at mean track length 2.0.

Tracking is a different primitive and it does work. On the same frames, LK with
a forward-backward check yields 563 surviving tracks at one step of 0.032 s.
Descriptors ask "do these two patches look alike?" -- and at ~1 event per active
pixel two windows sample DIFFERENT pixels of the same edge, so the answer is no.
Tracking only asks "where did this patch go since last frame?", over a step small
enough that the sampling has not turned over.

The measured limit, which sets what to expect here: track lifetime is ~0.4 s
REGARDLESS of step size (0.032 / 0.066 / 0.122 s all die at 3-4 steps). It is not
step size that kills tracks -- at a median 763 px/s the patch has left, changed
scale and been resampled by then. So tracks are short and their baselines are
short, which is exactly what triangulation needs to be long. Run it and read the
mean track length; do not assume it clears 3.

Output goes into a COLMAP database the normal way:

    feature_importer   <- per-image keypoint text files this writes
    matches_importer   <- pair/match text file this writes (--match_type raw,
                          so COLMAP still does its own geometric verification)
    mapper             <- unchanged

Everything downstream of the database (mapper, image_undistorter, 3DGS) is
untouched, so a model produced this way drops into Phase1.md §5 Step 4 onward.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("opencv-python is required: pip install opencv-python")

from frame_metrics import load_frames


def prepare(path, blur):
    """Blur + stretch. LK needs gradients; a raw count image is 4 grey levels."""
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        sys.exit(f"Unreadable: {path}")
    g = g.astype(np.float32)
    if blur:
        g = cv2.GaussianBlur(g, (0, 0), blur)
    return cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def track(items, args):
    """[(frame_idx, x, y)] per track id. Reseeds as tracks die."""
    lk = dict(winSize=(args.win, args.win), maxLevel=args.levels,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    tracks = {}
    active_ids = []
    active_pts = np.zeros((0, 1, 2), np.float32)
    next_id = 0
    prev = None

    for k, (path, _t) in enumerate(items):
        cur = prepare(path, args.blur)

        if prev is not None and len(active_ids):
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, active_pts, None, **lk)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, nxt, None, **lk)
            fb = np.linalg.norm(back - active_pts, axis=2).ravel()
            inside = ((nxt[:, 0, 0] >= 0) & (nxt[:, 0, 0] < cur.shape[1]) &
                      (nxt[:, 0, 1] >= 0) & (nxt[:, 0, 1] < cur.shape[0]))
            # The FB check is what separates a track from a coincidence; without
            # it LK reports a confident answer for every point, everywhere.
            keep = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < args.fb) & inside
            active_ids = [i for i, ok in zip(active_ids, keep) if ok]
            active_pts = nxt[keep].reshape(-1, 1, 2)
            for i, p in zip(active_ids, active_pts):
                tracks[i].append((k, float(p[0, 0]), float(p[0, 1])))

        if len(active_ids) < args.min_active:
            # Mask a radius around live tracks so reseeding adds coverage
            # instead of duplicating what is already tracked.
            mask = np.full(cur.shape, 255, np.uint8)
            for p in active_pts:
                cv2.circle(mask, (int(p[0, 0]), int(p[0, 1])), args.min_dist, 0, -1)
            new = cv2.goodFeaturesToTrack(
                cur, maxCorners=args.max_corners - len(active_ids),
                qualityLevel=args.quality, minDistance=args.min_dist, mask=mask)
            if new is not None:
                for p in new:
                    tracks[next_id] = [(k, float(p[0, 0]), float(p[0, 1]))]
                    active_ids.append(next_id)
                    next_id += 1
                active_pts = (np.concatenate([active_pts, new]) if len(active_pts)
                              else new).astype(np.float32)

        prev = cur
        if (k + 1) % 100 == 0:
            print(f"    frame {k + 1}/{len(items)}  {len(active_ids)} live tracks, "
                  f"{len(tracks)} total")

    return tracks


def export(tracks, items, out_dir, min_len, max_gap):
    """COLMAP text features + a raw match list. Returns (n_kp, n_pairs, stats)."""
    os.makedirs(out_dir, exist_ok=True)
    kept = {i: obs for i, obs in tracks.items() if len(obs) >= min_len}

    # Per-frame keypoint tables, and each observation's index within its frame.
    per_frame = {}
    index_of = {}
    for tid, obs in kept.items():
        for (k, x, y) in obs:
            lst = per_frame.setdefault(k, [])
            index_of[(tid, k)] = len(lst)
            lst.append((x, y))

    feat_dir = os.path.join(out_dir, "features")
    os.makedirs(feat_dir, exist_ok=True)
    for k, (path, _t) in enumerate(items):
        name = os.path.basename(path)
        pts = per_frame.get(k, [])
        with open(os.path.join(feat_dir, name + ".txt"), "w") as f:
            # COLMAP's text feature format: header, then x y scale orientation
            # plus a 128-d descriptor. Descriptors are never read on this path
            # (--match_type raw supplies the matches), so they are zeros.
            f.write(f"{len(pts)} 128\n")
            zeros = " ".join(["0"] * 128)
            for (x, y) in pts:
                f.write(f"{x:.4f} {y:.4f} 1.00 0.00 {zeros}\n")

    # Every pair of frames a track appears in, out to max_gap. Short-range pairs
    # are the reliable ones; long-range pairs are what SfM needs for baseline.
    pairs = {}
    for tid, obs in kept.items():
        frames = [k for (k, _x, _y) in obs]
        for a in range(len(frames)):
            for b in range(a + 1, len(frames)):
                if frames[b] - frames[a] > max_gap:
                    break
                key = (frames[a], frames[b])
                pairs.setdefault(key, []).append(
                    (index_of[(tid, frames[a])], index_of[(tid, frames[b])]))

    match_path = os.path.join(out_dir, "matches.txt")
    with open(match_path, "w") as f:
        for (a, b), ms in sorted(pairs.items()):
            f.write(f"{os.path.basename(items[a][0])} {os.path.basename(items[b][0])}\n")
            for i, j in ms:
                f.write(f"{i} {j}\n")
            f.write("\n")

    lens = np.array([len(o) for o in kept.values()]) if kept else np.array([0])
    # Lifetime in SECONDS as well as in frames. Frames alone are not comparable
    # across runs -- a window of 25 k events is a different amount of time at
    # every event rate -- and it is seconds that the ~0.4 s measured limit (see
    # this module's docstring) and the largest fragment's span are quoted in.
    t_of = [t for _p, t in items]
    lifetimes = np.array([t_of[obs[-1][0]] - t_of[obs[0][0]]
                          for obs in kept.values()]) if kept else np.array([0.0])
    stats = {
        "tracks_total": len(tracks),
        "tracks_kept": len(kept),
        "track_len_mean": float(lens.mean()),
        "track_len_max": int(lens.max()),
        "track_lifetime_s_mean": float(lifetimes.mean()),
        "track_lifetime_s_median": float(np.median(lifetimes)),
        "track_lifetime_s_max": float(lifetimes.max()),
        "observations": int(sum(len(v) for v in per_frame.values())),
        "pairs": len(pairs),
        "matches_per_pair_median": float(np.median([len(v) for v in pairs.values()]))
        if pairs else 0.0,
    }
    return feat_dir, match_path, stats


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True,
                    help="frame dir or scene dir (PNGs in <scene>/input/)")
    ap.add_argument("--out", required=True, help="scene dir for the COLMAP database")
    ap.add_argument("--blur", type=float, default=2.0,
                    help="Gaussian sigma before tracking (default 2.0). A raw "
                         "count image has ~4 grey levels and no gradients.")
    ap.add_argument("--max-corners", type=int, default=3000)
    ap.add_argument("--quality", type=float, default=0.01)
    ap.add_argument("--min-dist", type=int, default=8)
    ap.add_argument("--min-active", type=int, default=2000,
                    help="reseed whenever fewer tracks than this are alive")
    ap.add_argument("--win", type=int, default=31, help="LK window (default 31)")
    ap.add_argument("--levels", type=int, default=4, help="LK pyramid levels")
    ap.add_argument("--fb", type=float, default=2.0,
                    help="forward-backward error threshold in px (default 2.0)")
    ap.add_argument("--min-track-len", type=int, default=3,
                    help="drop tracks seen in fewer frames (default 3; COLMAP "
                         "cannot triangulate a 2-view track into anything stable)")
    ap.add_argument("--max-gap", type=int, default=10,
                    help="widest frame gap to emit a pair for (default 10)")
    ap.add_argument("--run-colmap", action="store_true",
                    help="import into a database and run the mapper")
    ap.add_argument("--camera-model", default="OPENCV")
    args = ap.parse_args()

    _man, items = load_frames(args.frames)
    print(f"\n  {len(items)} frames from {args.frames}")
    tracks = track(items, args)
    feat_dir, match_path, stats = export(
        tracks, items, args.out, args.min_track_len, args.max_gap)

    print(f"\n  tracks: {stats['tracks_kept']:,} kept of {stats['tracks_total']:,} "
          f"(>= {args.min_track_len} views)")
    print(f"  mean track length {stats['track_len_mean']:.2f}  "
          f"max {stats['track_len_max']}  observations {stats['observations']:,}")
    print(f"  lifetime mean {stats['track_lifetime_s_mean']:.3f} s  "
          f"median {stats['track_lifetime_s_median']:.3f} s  "
          f"max {stats['track_lifetime_s_max']:.3f} s  (measured limit ~0.4 s)")
    print(f"  pairs {stats['pairs']:,}  median matches/pair "
          f"{stats['matches_per_pair_median']:.0f}")
    with open(os.path.join(args.out, "tracks.json"), "w") as f:
        json.dump(stats, f, indent=1)

    if not args.run_colmap:
        print(f"\n  features -> {feat_dir}\n  matches  -> {match_path}\n")
        return

    img_dir = os.path.dirname(items[0][0])
    db = os.path.join(args.out, "db.db")
    if os.path.exists(db):
        os.remove(db)
    steps = [
        ["colmap", "feature_importer", "--database_path", db,
         "--image_path", img_dir, "--import_path", feat_dir,
         "--ImageReader.single_camera", "1",
         "--ImageReader.camera_model", args.camera_model],
        ["colmap", "matches_importer", "--database_path", db,
         "--match_list_path", match_path, "--match_type", "raw",
         "--SiftMatching.use_gpu", "0"],
    ]
    sparse = os.path.join(args.out, "sparse")
    os.makedirs(sparse, exist_ok=True)
    steps.append(
        ["colmap", "mapper", "--database_path", db, "--image_path", img_dir,
         "--output_path", sparse,
         "--Mapper.init_min_tri_angle", "2",
         "--Mapper.init_min_num_inliers", "30",
         "--Mapper.abs_pose_min_num_inliers", "15",
         "--Mapper.min_num_matches", "12",
         "--Mapper.filter_min_tri_angle", "0.5",
         "--Mapper.init_max_error", "6"])

    for cmd in steps:
        print(f"\n  $ {' '.join(cmd[:2])} ...")
        r = subprocess.run(cmd, capture_output=True, text=True)
        log = os.path.join(args.out, cmd[1] + ".log")
        with open(log, "w") as f:
            f.write(r.stdout + r.stderr)
        if r.returncode:
            sys.exit(f"  FAILED (see {log}):\n{r.stdout[-1500:]}{r.stderr[-1500:]}")
        print(f"    ok -> {log}")

    print(f"\n  Now: python frame_metrics.py colmap --scene {args.out}\n")


if __name__ == "__main__":
    main()
