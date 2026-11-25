"""E2EGS edge-weighted event loss (Kim et al., CVPR 2026, arXiv 2603.14684), Sec. 3.2 + 3.4.

Parameter values are not given in the paper; the defaults below are guesses.

Edge map, per training window:
  1. split the window's events into T equal-count slices -> event maps E_1..E_T
  2. D_t = |G_sigma * E_t - G_sigma * E_{t-1}|                          (Eq. 1)
  3. C(P) = max_t Var(D_t(P)) over p x p patches with overlap rho      (Eq. 2)
  4. keep patches with C > tau (tau adaptive: mean + tau_std * std of C)
  5. per pixel: max strength over the overlapping patches covering it -> M_raw
  6. normalise, Gaussian smooth, threshold, morphological closing -> M in [0,1]
Loss weight: w = 1 + beta * M                                           (Eq. 5)
"""
import os

import imageio
import numpy as np
import torch
import torch.nn.functional as F

EDGE_LOSS_DEFAULTS = {
    "beta": 1.0,          # w = 1 + beta * M
    "n_slices": 4,        # T
    "sigma": 1.0,         # G_sigma in Eq. 1 (pixels)
    # Tuned by eye on Replica room0 (2026-11-24): with var + max-normalisation a few
    # high-contrast patches dominated and M missed most object edges.
    "patch_size": 8,      # p
    "overlap": 0.75,      # rho
    "tau_std": 0.0,       # tau = mean(C) + tau_std * std(C)
    "strength": "std",    # per-patch edge strength written to M: "var" (Eq. 2) or "std" (sqrt, less dominated by a few patches)
    "norm_quantile": 0.9, # normalise M_raw by this quantile of its non-zero values (1.0 = max), then clamp to 1
    "post_sigma": 1.0,    # smoothing of M_raw (0 = off)
    "post_thresh": 0.1,   # zero out M < post_thresh after normalising
    "close_size": 5,      # morphological closing kernel (odd; 0/1 = off)
    "save_every": 200,    # save events|edge-map png every N calls per stage (0 = never)
}


def _gaussian_blur(img, sigma):
    """img [N,1,H,W]; separable Gaussian, replicate padding."""
    if sigma <= 0:
        return img
    r = max(1, int(3.0 * sigma + 0.5))
    x = torch.arange(-r, r + 1, device=img.device, dtype=img.dtype)
    k = torch.exp(-(x ** 2) / (2.0 * sigma ** 2))
    k = k / k.sum()
    img = F.conv2d(F.pad(img, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(img, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))


def _closing(img, k):
    if k <= 1:
        return img
    dil = F.max_pool2d(img, k, stride=1, padding=k // 2)
    return -F.max_pool2d(-dil, k, stride=1, padding=k // 2)


@torch.no_grad()
def compute_edge_map(slices, cfg):
    """slices [T,H,W] event maps of consecutive sub-windows -> M [H,W] in [0,1]."""
    T, H, W = slices.shape
    B = _gaussian_blur(slices.float().unsqueeze(1), cfg["sigma"])
    D = (B[1:] - B[:-1]).abs()                                  # Eq. 1

    p = int(cfg["patch_size"])
    s = max(1, int(round(p * (1.0 - cfg["overlap"]))))
    mean = F.avg_pool2d(D, p, stride=s)
    var = (F.avg_pool2d(D * D, p, stride=s) - mean * mean).clamp_min(0)
    C = var.max(dim=0, keepdim=True).values                     # Eq. 2, [1,1,h,w]

    tau = C.mean() + cfg["tau_std"] * C.std()
    S = C.sqrt() if cfg["strength"] == "std" else C
    strength = torch.where(C > tau, S, torch.zeros_like(S))

    # Put each patch's score at its top-left pixel; a p x p max-pool looking up-left
    # then gives every pixel the max over all patches that cover it.
    U = torch.zeros(1, 1, H, W, device=C.device, dtype=C.dtype)
    h, w = strength.shape[-2:]
    U[..., 0:h * s:s, 0:w * s:s] = strength
    M = F.max_pool2d(F.pad(U, (p - 1, 0, p - 1, 0)), p, stride=1)

    nz = M[M > 0]
    q = float(cfg["norm_quantile"])
    scale = (nz.float().quantile(q) if q < 1.0 and nz.numel() > 0 else M.max()).to(M.dtype)
    M = (M / scale.clamp_min(1e-12)).clamp(max=1.0)
    M = _gaussian_blur(M, cfg["post_sigma"])
    M = M / M.max().clamp_min(1e-12)
    M = torch.where(M >= cfg["post_thresh"], M, torch.zeros_like(M))
    return _closing(M, int(cfg["close_size"])).clamp(0, 1)[0, 0]


def split_events(events_stream, n_slices):
    """Equal-count split along the (time-sorted) event axis."""
    n = events_stream.shape[0]
    n_slices = max(2, min(int(n_slices), n))
    b = torch.linspace(0, n, n_slices + 1).long().tolist()
    return [events_stream[b[i]:b[i + 1]] for i in range(n_slices)]


def weighted_sq_err(err_sq, weight):
    """Apply an [H,W] weight to an [H,W] or [H,W,C] squared-error map."""
    if weight is None:
        return err_sq
    if err_sq.dim() == 3 and weight.dim() == 2:
        weight = weight.unsqueeze(-1)
    return weight.to(err_sq.dtype) * err_sq


@torch.no_grad()
def edge_weight(self, events_stream, stage):
    """Attached to IncEventGS's SLAM class by run_iegs.py. Returns w = 1 + beta*M."""
    cfg = self.config["edge_loss"]
    slices = [self.post_process_event_image(sl, _sigma=self.config["blur_sigma"])
              for sl in split_events(events_stream, cfg["n_slices"])]
    M = compute_edge_map(torch.stack(slices), cfg)

    counts = self.__dict__.setdefault("_edge_map_count", {})
    n = counts.get(stage, 0)
    counts[stage] = n + 1
    if cfg["save_every"] and n % cfg["save_every"] == 0:
        d = os.path.join(self.config["data"]["output"], self.config["data"]["exp_name"], "edge_maps")
        os.makedirs(d, exist_ok=True)
        ev = sum(slices).abs()
        ev = (ev / ev.max().clamp_min(1e-12)).clamp(0, 1)
        vis = torch.cat([ev, M], dim=1).cpu().numpy()     # events | edge map
        imageio.imwrite(os.path.join(d, f"{stage}_{n:05d}.png"), (vis * 255).astype(np.uint8))

    return 1.0 + cfg["beta"] * M
