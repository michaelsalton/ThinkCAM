#!/usr/bin/env python3
"""Render a COLMAP sparse model -- points and camera poses -- to a PNG.

Three ways exist to look at a reconstruction; this is the third:

  1. colmap gui --import_path <model> --database_path <db> --image_path <imgs>
     Interactive, needs a display.
  2. colmap model_converter --input_path <model> --output_path out.ply \
         --output_type PLY
     Then MeshLab / CloudCompare / Blender.
  3. This: a static three-view PNG, no display and no extra packages, which is
     what works over a terminal or in a report.

Panels are orthographic along the scene's own principal axes rather than the
raw XYZ of the model, because COLMAP's world frame is arbitrary -- gravity is
not up and the axes carry no meaning. Cameras are drawn as their centres, joined
in image order to show the trajectory, with a stub along each viewing direction.

Point brightness encodes track length: how many images saw that point. On event
frames that is the number worth looking at, since a cloud made of 2-view points
is what a failed reconstruction produces (docs/Phase1b-Results.md §6).
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("opencv-python is required: pip install opencv-python")


def load_model(path):
    """(points Nx4 [xyz + track length], cameras [(C, R, name)]).

    Accepts a .bin model by converting to TXT in a temp dir -- parsing COLMAP's
    binary format by hand would be one more thing to get subtly wrong.
    """
    tmp = None
    if not os.path.exists(os.path.join(path, "points3D.txt")):
        if not os.path.exists(os.path.join(path, "points3D.bin")):
            sys.exit(f"No COLMAP model in {path}")
        if not shutil.which("colmap"):
            sys.exit("colmap is not on PATH (needed to read a .bin model).")
        tmp = tempfile.mkdtemp(prefix="model_txt_")
        r = subprocess.run(["colmap", "model_converter", "--input_path", path,
                            "--output_path", tmp, "--output_type", "TXT"],
                           capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"model_converter failed:\n{r.stderr[-800:]}")
        path = tmp

    pts = []
    with open(os.path.join(path, "points3D.txt")) as f:
        for line in f:
            if line.startswith("#"):
                continue
            v = line.split()
            if len(v) < 8:
                continue
            # TRACK[] is (IMAGE_ID, POINT2D_IDX) pairs after the 8 fixed fields.
            pts.append([float(v[1]), float(v[2]), float(v[3]), (len(v) - 8) // 2])

    cams = []
    with open(os.path.join(path, "images.txt")) as f:
        lines = [l for l in f if not l.startswith("#")]
    # Each image is two lines: pose line, then its 2D points (ignored here).
    for i in range(0, len(lines), 2):
        v = lines[i].split()
        if len(v) < 9:
            continue
        w, x, y, z = (float(a) for a in v[1:5])
        t = np.array([float(a) for a in v[5:8]])
        R = np.array([
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]])
        cams.append((-R.T @ t, R, v[-1]))   # camera centre in world coordinates

    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    if not pts:
        sys.exit("Model has no 3D points.")
    return np.array(pts), sorted(cams, key=lambda c: c[2])


def principal_axes(P):
    """Scene axes by PCA. COLMAP's world frame is arbitrary; this at least puts
    the widest spread across the page."""
    X = P - P.mean(0)
    _u, _s, vt = np.linalg.svd(X, full_matrices=False)
    return vt


def panel(P, tracks, cams, ax_u, ax_v, size, title, pct):
    img = np.full((size, size, 3), 18, np.uint8)
    uv = np.stack([P @ ax_u, P @ ax_v], 1)
    cuv = np.stack([np.array([c[0] for c in cams]) @ ax_u,
                    np.array([c[0] for c in cams]) @ ax_v], 1)

    # Scale on a percentile of the POINTS so a few outliers cannot shrink the
    # scene to a dot, then include the cameras so the trajectory stays on-page.
    lo = np.percentile(uv, 100 - pct, axis=0)
    hi = np.percentile(uv, pct, axis=0)
    lo = np.minimum(lo, cuv.min(0))
    hi = np.maximum(hi, cuv.max(0))
    span = np.maximum(hi - lo, 1e-9).max()
    m = size * 0.08
    to_px = lambda q: ((q - lo) / span * (size - 2 * m) + m).astype(np.int32)

    q = to_px(uv)
    tmax = max(2, tracks.max())
    for (px, py), tl in zip(q, tracks):
        if 0 <= px < size and 0 <= py < size:
            # 2-view points dim, well-observed points bright.
            s = (tl - 2) / max(1, tmax - 2)
            c = (90 + int(120 * s), 150 + int(105 * s), 110 + int(60 * s))
            cv2.circle(img, (px, size - 1 - py), 1, c, -1, cv2.LINE_AA)

    cq = to_px(cuv)
    for i in range(len(cq) - 1):
        cv2.line(img, (cq[i][0], size - 1 - cq[i][1]),
                 (cq[i + 1][0], size - 1 - cq[i + 1][1]), (60, 60, 235), 1, cv2.LINE_AA)
    for (c, R, _n), (px, py) in zip(cams, cq):
        look = c + R[2] * span * 0.06          # +Z of the camera is its view axis
        lx, ly = to_px(np.array([look @ ax_u, look @ ax_v]))
        cv2.line(img, (px, size - 1 - py), (int(lx), size - 1 - int(ly)),
                 (80, 200, 255), 1, cv2.LINE_AA)
        cv2.circle(img, (px, size - 1 - py), 3, (80, 200, 255), -1, cv2.LINE_AA)

    cv2.rectangle(img, (0, 0), (size, 22), (0, 0, 0), -1)
    cv2.putText(img, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (235, 235, 235), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="COLMAP model dir (sparse/0), .bin or .txt")
    ap.add_argument("--out", required=True, help="output PNG")
    ap.add_argument("--size", type=int, default=560, help="panel size in px")
    ap.add_argument("--percentile", type=float, default=98.0,
                    help="point percentile used for framing (default 98)")
    ap.add_argument("--ply", default=None,
                    help="also write a PLY here for MeshLab/CloudCompare")
    args = ap.parse_args()

    P, cams = load_model(args.model)
    xyz, tracks = P[:, :3], P[:, 3].astype(int)
    axes = principal_axes(xyz)

    panels = [
        panel(xyz, tracks, cams, axes[0], axes[1], args.size, "principal 1-2", args.percentile),
        panel(xyz, tracks, cams, axes[0], axes[2], args.size, "principal 1-3", args.percentile),
        panel(xyz, tracks, cams, axes[2], axes[1], args.size, "principal 3-2", args.percentile),
    ]
    cv2.imwrite(args.out, np.hstack(panels))

    C = np.array([c[0] for c in cams])
    path = float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum()) if len(C) > 1 else 0.0
    depth = float(np.median(np.linalg.norm(xyz - C.mean(0), axis=1)))
    print(f"\n  {len(P)} points, {len(cams)} cameras  -> {args.out}")
    print(f"  mean track length {tracks.mean():.2f}   "
          f"{100 * (tracks > 2).mean():.0f}% of points seen by 3+ images")
    print(f"  camera path {path:.2f}, scene depth {depth:.2f} "
          f"(depth/baseline {depth / max(1e-9, np.linalg.norm(C.max(0) - C.min(0))):.1f})")

    if args.ply:
        if not shutil.which("colmap"):
            sys.exit("colmap is not on PATH.")
        subprocess.run(["colmap", "model_converter", "--input_path", args.model,
                        "--output_path", args.ply, "--output_type", "PLY"],
                       capture_output=True, text=True)
        print(f"  PLY -> {args.ply}")
    print()


if __name__ == "__main__":
    main()
