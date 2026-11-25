"""Compare final ATE across runs:  python -m models.compare_ate work/runs/<config name>

For each run folder, takes the newest pose file (global BA, else local BA, else
tracking), Sim3-aligns the estimated camera positions onto ground truth (Umeyama),
and prints ATE RMSE / max in cm.
"""
import argparse
from pathlib import Path

import numpy as np


def positions(path):
    return np.loadtxt(path).reshape(-1, 3, 4)[:, :, 3]       # KITTI rows: 3x4 [R|t]


def ate_cm(est, gt):
    """Sim3 (Umeyama) align est onto gt; return (rmse, max) in cm."""
    mu_e, mu_g = est.mean(0), gt.mean(0)
    xe, xg = est - mu_e, gt - mu_g
    U, S, Vt = np.linalg.svd(xg.T @ xe / len(est))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (xe ** 2).sum(1).mean()
    err = np.linalg.norm(s * xe @ R.T + mu_g - gt, axis=1)
    return 100 * np.sqrt((err ** 2).mean()), 100 * err.max()


def final_pose_pair(run):
    for stage in ("global_BA", "BA", "tracking"):
        ests = sorted((run / stage).glob("pose_file_est_*.txt"), key=lambda p: p.stat().st_mtime)
        if ests:
            gt = ests[-1].with_name(ests[-1].name.replace("pose_file_est_", "pose_file_gt_"))
            return stage, ests[-1], gt
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    print(f"{'run':<28}{'stage':<11}{'frames':>7}{'ATE cm':>9}{'max cm':>9}  file")
    for run in sorted(p for p in args.root.iterdir() if p.is_dir()):
        pair = final_pose_pair(run)
        if pair is None:
            print(f"{run.name:<28}(no pose files yet)")
            continue
        stage, est, gt = pair
        e, g = positions(est), positions(gt)
        rmse, mx = ate_cm(e, g)
        print(f"{run.name:<28}{stage:<11}{len(e):>7}{rmse:>9.2f}{mx:>9.2f}  {est.name}")


if __name__ == "__main__":
    main()
