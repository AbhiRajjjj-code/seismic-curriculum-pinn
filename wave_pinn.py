"""
Fourier-feature PINN for the 2D acoustic wave equation in a two-layer medium

    u_tt = c(y)^2 (u_xx + u_yy) + q(x, y, t),      c = 1 km/s for y < 1 km, 2 km/s above

The only data used is one finite-difference (FD) snapshot (u, u_t) at T0 = 150 ms. From there the
PINN predicts the wavefield up to T_END, and is scored against a fine FD run (relative L2 error).

Method (details in BTP_report.pdf):
  1. Time marching. [T0, T_END] is cut into windows of DT = 25 ms. Each window is trained from
     scratch and its end state (u, u_t) is the initial condition of the next window.
  2. Hard initial condition. Inside a window, with tau = t - t_start,
         u(x, y, tau) = u0 + tau * v0 + (tau / DT)^2 * N(x, y, tau)
     so u and u_t are exact at tau = 0 for any network N. There is no IC loss, only the PDE residual.
  3. Domain decomposition. A window is not solved by one network but by one small Fourier-feature
     MLP per 0.4 km tile (+ 0.1 km halo). In 25 ms a wave moves at most 50 m, so a patch network is
     correct except in a 60 m rim along its edge; overlapping patches are blended with weights
     that are zero in that rim. All patch networks of a group are trained in one batched model.
  4. Interface inputs (--kink-feature=1). Two extra inputs that are C^1 at the interface and have a
     jump in the 2nd / 3rd y-derivative, because u_yy of the true wavefield jumps there.
     --kink-local=L multiplies both by exp(-(d / L)^2), d = distance above the interface, so they
     act only near the interface and do not give the networks a smooth ramp far above it.
  5. Absorbing edges (--pad-x, --pad-top). The PINN solves on a larger domain, so the wave leaves
     the physical domain instead of piling up at its edge. Scoring uses the physical domain only.

Layout follows CausalPINNs/NS/NS.py: network, PINN class (neural_net, residual_net,
residuals_and_weights, loss, step, train), then the time-marching script.

Defaults = the final recipe (interface inputs on and faded over 0.2 km, padded edges 0.35 / 0.15 km,
10000 Adam steps per window, fixed Fourier frequencies, no causal weights). Older runs (steps 2-5 in
PROJECT_SUMMARY.txt) switched these on by flag; the old recipe is --kink-feature=0 --kink-local=0 --pad-x=0 --pad-top=0.

Usage:  python wave_pinn.py --t-end=0.6
        python wave_pinn.py --resume=results/state_600ms.npz --t-end=1.0 --steps=20000
        (all flags: python wave_pinn.py --help;  the final run: run_all.sh / run_all.bat)
"""
import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

from fd_solver import run_fd_solver

# ---------------------------------------------------------------------------------------------
# Problem set-up (km, s, km/s)
# ---------------------------------------------------------------------------------------------
C_LOW, C_UP, INTERFACE_Y = 1.0, 2.0, 1.0          # velocity below / above the interface
SOURCE_X, SOURCE_Y = 1.0, 0.5                     # source position
FREQUENCY, SOURCE_SIGMA = 15.0, 0.03              # Ricker peak frequency (Hz), Gaussian source width (km)
SOURCE_T0 = 1.0 / FREQUENCY                       # Ricker time shift
OMEGA = 2.0 * math.pi * FREQUENCY
U_REF = 1.0 / OMEGA ** 2                          # the network works with u / U_REF, which is O(1)
FD_NX, FD_NY = 451, 676                           # reference FD grid on x in [0, 2], y in [-1, 2] (dx = 4.4 m)

# ---------------------------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Fourier-feature PINN, hard-IC time marching on patches")
parser.add_argument("--t0", type=float, default=0.150, help="time of the FD snapshot used as initial condition (s)")
parser.add_argument("--t-end", type=float, default=1.0, help="final time (s)")
parser.add_argument("--dt", type=float, default=0.025, help="window length (s)")
parser.add_argument("--steps", type=int, default=10000, help="Adam steps per group of patches per window")
parser.add_argument("--lr", type=float, default=2e-3, help="initial learning rate (cosine decay to 1%% of it)")
parser.add_argument("--features", type=int, default=128, help="number of random Fourier frequencies")
parser.add_argument("--hidden", type=int, default=128, help="width of the 4 hidden layers")
parser.add_argument("--sigma-xy", type=float, default=12.0, help="std of the Fourier frequencies in x, y (cycles/km)")
parser.add_argument("--sigma-t", type=float, default=15.0, help="std of the Fourier frequencies in t (cycles/s)")
parser.add_argument("--train-freq", type=int, default=0,
                    help="1: the Fourier frequencies B are trained too (each patch its own copy, same start values)")
parser.add_argument("--freq-lr", type=float, default=1.0, help="learning rate of B as a multiple of --lr")
parser.add_argument("--arch", default="fourier", choices=["fourier", "vanilla", "modified"], help="patch network type")
parser.add_argument("--tile", type=float, default=0.4, help="tile size (km); patch = tile + halo")
parser.add_argument("--patch-pts", type=int, default=1800, help="collocation points per patch per step, inside the wave region")
parser.add_argument("--patch-far", type=int, default=600, help="collocation points per patch per step, uniform in the patch")
parser.add_argument("--group", type=int, default=12, help="patch networks trained together (GPU memory)")
parser.add_argument("--quiet", type=float, default=3e-4, help="skip tiles whose expected correction is below this fraction of peak |u|")
parser.add_argument("--lowpass", type=float, default=35.0, help="cut-off of the hand-off low-pass filter (cycles/km)")
parser.add_argument("--kink-feature", type=int, default=1, help="1: give the networks the two interface inputs (0 = off)")
parser.add_argument("--kink-local", type=float, default=0.2,
                    help="> 0: fade the interface inputs out above the interface over this length (km); 0 = no fade")
parser.add_argument("--pad-x", type=float, default=0.35, help="extra domain on the left and right (km)")
parser.add_argument("--pad-top", type=float, default=0.15, help="extra domain above the top edge (km)")
parser.add_argument("--causal-tol", type=float, default=0.0, help="> 0: causal weights inside a window (Wang et al. 2022)")
parser.add_argument("--causal-slices", type=int, default=16, help="time slices per window for the causal weights")
parser.add_argument("--resume", default=None, help="state file (.npz) of an earlier run to continue from")
parser.add_argument("--n-win", type=int, default=999, help="stop after this many windows")
parser.add_argument("--snap-ms", type=int, default=0, help="> 0: also save pinn_<t>ms.npz (to plot) whenever t is a multiple of this (ms)")
parser.add_argument("--tag", default="", help="suffix for the output file names")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

T0, T_END, DT = args.t0, args.t_end, args.dt
C_MAX = max(C_LOW, C_UP)
SUPPORT_FRAC = 1e-3                 # the wave "exists" where |u| or |u_t| exceeds this fraction of its peak
MARGIN = 0.06                       # km added around the wave when choosing where to place collocation points
PAD_T = 0.03                        # s of slack on the first-arrival time (the wavelet has a finite length)
RIM = C_MAX * DT + 0.01             # km along a patch edge where the patch network is not valid
RAMP = 0.08                         # km over which neighbouring patches are blended
HALO = RIM + 0.5 * RAMP             # km by which a patch extends beyond its tile
GRAD_CLIP = 10.0                    # per-patch gradient norm clip


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------------------------
# Physics: velocity model and source
# ---------------------------------------------------------------------------------------------
def velocity_numpy(x, y):
    return np.where(np.asarray(y) < INTERFACE_Y, C_LOW, C_UP)


def source_numpy(x, y, t):
    """q(x, y, t) = Gaussian in space * Ricker wavelet in time."""
    term = np.pi * FREQUENCY * (t - SOURCE_T0)
    gauss = np.exp(-((x - SOURCE_X) ** 2 + (y - SOURCE_Y) ** 2) / (2.0 * SOURCE_SIGMA ** 2))
    return gauss * ((1.0 - 2.0 * term ** 2) * np.exp(-term ** 2))


def source_torch(x, y, t):
    term = torch.pi * FREQUENCY * (t - SOURCE_T0)
    gauss = torch.exp(-((x - SOURCE_X) ** 2 + (y - SOURCE_Y) ** 2) / (2.0 * SOURCE_SIGMA ** 2))
    return gauss * ((1.0 - 2.0 * term ** 2) * torch.exp(-term ** 2))


# ---------------------------------------------------------------------------------------------
# Grid helpers (used for the initial condition of every window)
# ---------------------------------------------------------------------------------------------
def rel_l2(a, b):
    return np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30)


def field_at(fd, t):
    """FD wavefield at time t (linear interpolation between stored time steps)."""
    tt = fd["t"]
    j = min(max(int(np.searchsorted(tt, t)), 1), len(tt) - 1)
    w = (t - tt[j - 1]) / (tt[j] - tt[j - 1])
    return ((1 - w) * fd["wavefield"][j - 1] + w * fd["wavefield"][j]).astype(np.float64)


def laplacian4(f, h):
    """4th-order finite-difference Laplacian on the grid."""
    out = np.zeros_like(f)
    c = (-1.0, 16.0, -30.0, 16.0, -1.0)
    for ax in (0, 1):
        s = [slice(2, -2), slice(2, -2)]
        acc = 0.0
        for k, ck in zip(range(-2, 3), c):
            sl = list(s)
            sl[ax] = slice(2 + k, f.shape[ax] - 2 + k)
            acc = acc + ck * f[tuple(sl)]
        out[2:-2, 2:-2] += acc / (12.0 * h * h)
    return out


def build_fields(u, v, h, c2):
    """
    (u, v) on the grid -> channels [u, v, A, B] with A = c^2 lap(u) = u_tt and B = c^2 lap(v) = u_ttt.
    The residual of the Taylor part u0 + tau v0 needs A and B. c^2 lap is continuous across the
    interface (lap itself jumps by the c^2 ratio), so it can be interpolated there.
    """
    return np.stack([u, v, c2 * laplacian4(u, h), c2 * laplacian4(v, h)])


def lowpass(f, h, kc):
    """Super-Gaussian low-pass at kc cycles/km. The 15 Hz wave has almost no energy above ~35 cycles/km."""
    kx = np.fft.fftfreq(f.shape[0], d=h)[:, None]
    ky = np.fft.fftfreq(f.shape[1], d=h)[None, :]
    return np.real(np.fft.ifft2(np.fft.fft2(f) * np.exp(-(np.hypot(kx, ky) / kc) ** 8)))


def travel_time(x, y, n_xi=801):
    """
    First-arrival time (s) from the source for one flat interface (Fermat's principle): straight ray
    in the source layer; elsewhere the minimum over interface points of (first arrival on the
    interface, including running along it at the fast speed) + (straight leg from there).
    Used only as a mask: no wave can exist where it has not arrived yet.
    """
    X, Y = np.meshgrid(x, y, indexing="ij")
    T = np.hypot(X - SOURCE_X, Y - SOURCE_Y) / C_LOW
    above = Y >= INTERFACE_Y
    xi = np.linspace(x[0], x[-1], n_xi)
    leg1 = np.hypot(xi - SOURCE_X, INTERFACE_Y - SOURCE_Y) / C_LOW
    t_if = (leg1[None, :] + np.abs(xi[:, None] - xi[None, :]) / C_MAX).min(axis=1)
    for mask, c_leg in ((above, C_UP), (~above, C_LOW)):
        ix, iy = np.nonzero(mask)
        best = np.full(len(ix), np.inf)
        for i in range(0, len(ix), 20000):
            sl = slice(i, i + 20000)
            leg2 = np.hypot(x[ix[sl], None] - xi[None, :], y[iy[sl], None] - INTERFACE_Y) / c_leg
            best[sl] = (t_if[None, :] + leg2).min(axis=1)
        T[ix, iy] = np.minimum(T[ix, iy], best)
    return T


def smoothstep(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


def make_tiles(lo, hi):
    """Patch intervals along one axis: tiles of ~args.tile covering [lo, hi], each widened by HALO."""
    n = max(1, int(np.ceil((hi - lo) / args.tile - 1e-9)))
    edges = np.linspace(lo, hi, n + 1)
    return [(max(lo, edges[i] - HALO), min(hi, edges[i + 1] + HALO)) for i in range(n)]


def axis_weight(c, p0, p1, lo, hi):
    """Blend weight along one axis: 0 in the rim, 1 in the core, no rim at a domain edge. Neighbours sum to 1."""
    w = np.ones_like(c)
    if p0 > lo + 1e-9:
        w = w * smoothstep((c - p0 - RIM) / RAMP)
    if p1 < hi - 1e-9:
        w = w * smoothstep((p1 - c - RIM) / RAMP)
    return w * ((c >= p0) & (c <= p1))


def wave_region(fields, x, arrival, t_end):
    """Cells where the field can be non-zero by the end of the window."""
    h = x[1] - x[0]
    support = (np.abs(fields[:2]) > SUPPORT_FRAC * np.abs(fields[:2]).max(axis=(1, 2), keepdims=True)).any(0)
    region = ndimage.distance_transform_edt(~support) * h <= C_MAX * DT + MARGIN
    return region & (arrival <= t_end + PAD_T)


class GridFields:
    """The window's IC channels [u0, v0, A, B] on the grid, sampled at arbitrary points (bicubic)."""

    def __init__(self, x, y, fields, device):
        self.x0, self.x1 = float(x[0]), float(x[-1])
        self.y0, self.y1 = float(y[0]), float(y[-1])
        arr = torch.tensor(fields, dtype=torch.float32, device=device)   # (4, nx, ny)
        self.grid = arr.permute(0, 2, 1).unsqueeze(0).contiguous()       # (1, 4, ny, nx)

    def sample(self, x, y):
        gx = 2.0 * (x - self.x0) / (self.x1 - self.x0) - 1.0
        gy = 2.0 * (y - self.y0) / (self.y1 - self.y0) - 1.0
        g = torch.cat([gx, gy], dim=1).view(1, 1, -1, 2)
        out = F.grid_sample(self.grid, g, mode="bicubic", padding_mode="zeros", align_corners=True)
        return out[0, :, 0, :]                                            # (4, N)


# ---------------------------------------------------------------------------------------------
# Network: P independent MLPs (one per patch) evaluated in one pass
# ---------------------------------------------------------------------------------------------
class PatchMLP(nn.Module):
    """
    Returns the correction (tau / DT)^2 * N(x, y, tau) for every patch. Input (P, n, 3), output (P, n, 1).
    Every weight carries a leading patch dimension; the patches share nothing.

    arch:
        "fourier"   [sin, cos](2 pi B p), p = (x - xc, y - yc, tau), B fixed random ~ N(0, sigma^2);
                    then 4 tanh layers                                                    (default)
                    train_freq: B becomes a trained parameter, one copy per patch (P, 3, F)
        "vanilla"   4 tanh layers on the scaled coordinates
        "modified"  scaled coordinates, MLP with the gating branches U, V of Wang et al. 2021
                    (the network of NS.py):  h <- h * U + (1 - h) * V after every hidden layer
    kink_y: interface height. With it the network gets two more inputs r^2/(1+r^2), r^3/(1+r^3),
        r = max(y - kink_y, 0) / 0.1 km. With kink_local = L > 0 both are multiplied by exp(-(0.1 r / L)^2).
    The last layer starts at zero, so training starts from the plain Taylor step u0 + tau v0.
    """

    def __init__(self, centers, t_start, window, sigma_xy, sigma_t, n_features, hidden_dim,
                 n_hidden=4, seed=0, kink_y=None, kink_scale=0.1, kink_local=0.0, arch="fourier", train_freq=False):
        super().__init__()
        self.arch, self.window = arch, float(window)
        self.kink_y, self.kink_scale, self.kink_local = kink_y, float(kink_scale), float(kink_local)
        n_patch = len(centers)
        gen = torch.Generator().manual_seed(seed)
        B = torch.randn(3, n_features, generator=gen)
        B[0:2] *= sigma_xy
        B[2:3] *= sigma_t
        if train_freq:
            self.B = nn.Parameter(B.repeat(n_patch, 1, 1))
        else:
            self.register_buffer("B", B)
        self.register_buffer("t_start", torch.tensor(float(t_start)))
        self.register_buffer("centers", torch.tensor(centers, dtype=torch.float32).view(n_patch, 1, 2))

        n_input = (2 * n_features if arch == "fourier" else 3) + (2 if kink_y is not None else 0)
        sizes = [n_input] + [hidden_dim] * n_hidden + [1]
        gain = nn.init.calculate_gain("tanh")

        def linear(n_in, n_out, zero=False):
            std = 0.0 if zero else gain * math.sqrt(2.0 / (n_in + n_out))            # Xavier
            return (nn.Parameter(torch.randn(n_patch, n_in, n_out, generator=gen) * std),
                    nn.Parameter(torch.zeros(n_patch, 1, n_out)))

        self.weights, self.biases = nn.ParameterList(), nn.ParameterList()
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            w, b = linear(n_in, n_out, zero=(i == len(sizes) - 2))
            self.weights.append(w)
            self.biases.append(b)
        if arch == "modified":
            (self.w_u, self.b_u), (self.w_v, self.b_v) = linear(n_input, hidden_dim), linear(n_input, hidden_dim)

    def forward(self, points, patch=None):
        """points (P, n, 3); or, with patch=k, points (n, 3) evaluated by network k only."""
        sel = slice(None) if patch is None else slice(patch, patch + 1)
        if patch is not None:
            points = points.unsqueeze(0)
        tau = points[..., 2:3] - self.t_start
        p = torch.cat([points[..., 0:2] - self.centers[sel], tau], dim=-1)
        if self.arch == "fourier":
            z = 2.0 * math.pi * (torch.bmm(p, self.B[sel]) if self.B.dim() == 3 else p @ self.B)
            h = torch.cat([torch.sin(z), torch.cos(z)], dim=-1)
        else:                                            # coordinates scaled to about [-1, 1]
            h = torch.cat([p[..., 0:2] / 0.3, 2.0 * tau / self.window - 1.0], dim=-1)
        if self.kink_y is not None:
            r = torch.clamp(points[..., 1:2] - self.kink_y, min=0.0) / self.kink_scale
            k2, k3 = r ** 2 / (1.0 + r ** 2), r ** 3 / (1.0 + r ** 3)
            if self.kink_local > 0.0:
                fade = torch.exp(-(r * self.kink_scale / self.kink_local) ** 2)
                k2, k3 = k2 * fade, k3 * fade
            h = torch.cat([h, k2, k3], dim=-1)
        last = len(self.weights) - 1
        if self.arch == "modified":
            U = torch.tanh(torch.baddbmm(self.b_u[sel], h, self.w_u[sel]))
            V = torch.tanh(torch.baddbmm(self.b_v[sel], h, self.w_v[sel]))
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = torch.baddbmm(b[sel], h, w[sel])
            if i < last:
                h = torch.tanh(h)
                if self.arch == "modified":
                    h = h * U + (1.0 - h) * V
        out = (tau / self.window) ** 2 * h
        return out if patch is None else out[0]


# ---------------------------------------------------------------------------------------------
# PINN for one window and one group of patches
# ---------------------------------------------------------------------------------------------
class PINN:
    def __init__(self, boxes, fields, t_start, x, y, region, y_if, device):
        """boxes: list of patches (x0, x1, y0, y1). fields: IC channels [u0, v0, A, B] on the grid (x, y)."""
        self.device, self.t_start, self.y_if = device, t_start, y_if
        self.boxes, self.n_patch, self.h = boxes, len(boxes), x[1] - x[0]

        # collocation cells of every patch: the part of the patch where the wave can be
        cell_lists = []
        for (x0, x1, y0, y1) in boxes:
            ix, iy = np.nonzero(region & ((x[:, None] >= x0) & (x[:, None] <= x1) & (y[None, :] >= y0) & (y[None, :] <= y1)))
            cell_lists.append(np.stack([x[ix], y[iy]], 1))
        self.counts = torch.tensor([len(c) for c in cell_lists], device=device)
        self.cells = torch.zeros(self.n_patch, int(self.counts.max()), 2, device=device)
        for k, c in enumerate(cell_lists):
            self.cells[k, :len(c)] = torch.tensor(c, dtype=torch.float32, device=device)
        self.lo = torch.tensor([[b[0], b[2]] for b in boxes], dtype=torch.float32, device=device).view(self.n_patch, 1, 2)
        self.hi = torch.tensor([[b[1], b[3]] for b in boxes], dtype=torch.float32, device=device).view(self.n_patch, 1, 2)
        self.rows = torch.arange(self.n_patch, device=device)[:, None]

        self.grid = GridFields(x, y, fields, device)
        self.net = PatchMLP([(0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3])) for b in boxes], t_start, DT,
                            args.sigma_xy, args.sigma_t, args.features, args.hidden,
                            kink_y=y_if if args.kink_feature else None, kink_local=args.kink_local, arch=args.arch,
                            train_freq=bool(args.train_freq)).to(device)
        self.freq = [self.net.B] if args.train_freq else []
        self.params = [q for q in self.net.parameters() if all(q is not f for f in self.freq)]
        groups = [{"params": self.params}] + ([{"params": self.freq, "lr": args.lr * args.freq_lr}] if self.freq else [])
        self.optimizer = torch.optim.Adam(groups, lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, args.steps, eta_min=0.01 * args.lr)

    def sample_points(self):
        """Fresh random collocation points (x, y, t) for every patch, and tau = t - t_start."""
        n_in, n_far = args.patch_pts, args.patch_far
        idx = (torch.rand(self.n_patch, n_in, device=self.device) * self.counts[:, None]).long().clamp(max=self.cells.shape[1] - 1)
        xy_in = self.cells[self.rows, idx] + (torch.rand(self.n_patch, n_in, 2, device=self.device) - 0.5) * self.h
        xy_far = self.lo + (self.hi - self.lo) * torch.rand(self.n_patch, n_far, 2, device=self.device)
        xy = torch.cat([xy_in, xy_far], 1)
        tau = DT * torch.rand(self.n_patch, n_in + n_far, 1, device=self.device)
        return torch.cat([xy, self.t_start + tau], 2), tau

    def neural_net(self, pts):
        """Correction (tau/DT)^2 N. The full field is u0 + tau v0 + this."""
        return self.net(pts)

    def residual_net(self, pts, tau):
        """
        PDE residual u_tt - c^2 lap(u) - q of u = u0 + tau v0 + correction, per point (P, n).
        The Taylor part contributes -(A + tau B) (its u_tt is 0), taken from the grid; the derivatives
        of the correction come from autograd. Also returns the residual with the correction set to 0.
        """
        pts = pts.requires_grad_(True)
        u = self.neural_net(pts)
        g = torch.autograd.grad(u.sum(), pts, create_graph=True)[0]
        u_xx = torch.autograd.grad(g[..., 0].sum(), pts, create_graph=True)[0][..., 0]
        u_yy = torch.autograd.grad(g[..., 1].sum(), pts, create_graph=True)[0][..., 1]
        u_tt = torch.autograd.grad(g[..., 2].sum(), pts, create_graph=True)[0][..., 2]
        flat = pts.detach().view(-1, 3)
        gf = self.grid.sample(flat[:, 0:1], flat[:, 1:2])
        c = torch.where(flat[:, 1] < self.y_if, C_LOW, C_UP).view(self.n_patch, -1)
        src = (source_torch(flat[:, 0:1], flat[:, 1:2], flat[:, 2:3]) / U_REF).view(self.n_patch, -1)
        res_taylor = -(gf[2] + tau.reshape(-1) * gf[3]).view(self.n_patch, -1) - src
        res = res_taylor + u_tt - c ** 2 * (u_xx + u_yy)
        return res / OMEGA ** 2, res_taylor / OMEGA ** 2

    def residuals_and_weights(self, sq, base, tau):
        """
        Causal weights of Wang et al. 2022 inside the window: the window is cut into M time slices with
        mean squared residual L_i, and slice i gets w_i = exp(-tol * sum_{j<i} L_j). L is measured
        relative to the Taylor-only residual of the patch, so tol has no units.
        """
        M = args.causal_slices
        sl = (tau.view(self.n_patch, -1) / DT * M).long().clamp(max=M - 1)
        zero = torch.zeros(self.n_patch, M, device=self.device)
        L = zero.scatter_add(1, sl, sq) / zero.scatter_add(1, sl, torch.ones_like(sq)).clamp(min=1.0)
        W = torch.exp(-args.causal_tol * (torch.cumsum(L, 1) - L) / (M * base[:, None] + 1e-30)).detach()
        return L, W

    def loss(self, pts, tau):
        """Per patch: training objective, plain mean squared residual, Taylor-only residual, smallest causal weight."""
        res, res_taylor = self.residual_net(pts, tau)
        sq, base = res ** 2, (res_taylor ** 2).mean(1)
        if args.causal_tol <= 0.0:
            return sq.mean(1), sq.mean(1), base, 1.0
        L, W = self.residuals_and_weights(sq, base, tau)
        return (W * L).mean(1), sq.mean(1), base, W.min().item()

    def step(self):
        objective, plain, base, w_min = self.loss(*self.sample_points())
        self.optimizer.zero_grad(set_to_none=True)
        objective.sum().backward()                       # patches are independent: each gets its own gradient
        for params in (self.params, self.freq):           # trained frequencies (if any) are clipped on their own
            if not params:
                continue
            norm = torch.sqrt(sum(q.grad.reshape(self.n_patch, -1).pow(2).sum(1) for q in params))
            scale = torch.clamp(GRAD_CLIP / (norm + 1e-12), max=1.0).view(self.n_patch, 1, 1)
            for q in params:
                q.grad.mul_(scale)
        self.optimizer.step()
        self.scheduler.step()
        return plain, base, w_min

    def train(self, n_iter, tag):
        t_clock = time.time()
        B0 = self.net.B.detach().clone() if self.freq else None
        for it in range(1, n_iter + 1):
            plain, base, w_min = self.step()
            if it == 1 or it % max(1, n_iter // 4) == 0:
                ratio = (plain / (base + 1e-30)).detach()
                drift = ""
                if self.freq:                            # how far the frequencies moved: rms change / rms start value
                    d = ((self.net.B.detach() - B0).pow(2).mean((1, 2)) / B0.pow(2).mean((1, 2))).sqrt()
                    drift = f"  frequency change median={d.median().item():.3f} max={d.max().item():.3f}"
                print(f"    [{tag}] it {it:5d} [{time.time() - t_clock:5.0f}s] residual / Taylor-only residual per patch: "
                      f"median={ratio.median().item():.2e} worst={ratio.max().item():.2e}"
                      + (f"  smallest causal weight={w_min:.3f}" if args.causal_tol > 0.0 else "") + drift, flush=True)

    def predict(self, k, xs, ys, t):
        """Correction and its time derivative from patch network k at the points (xs, ys) and time t."""
        n = len(xs)
        uN, vN = np.zeros(n), np.zeros(n)
        for i in range(0, n, 20000):
            sl = slice(i, i + 20000)
            p = torch.tensor(np.stack([xs[sl], ys[sl], np.full(len(xs[sl]), t)], 1),
                             dtype=torch.float32, device=self.device).requires_grad_(True)
            out = self.net(p, patch=k)
            g = torch.autograd.grad(out.sum(), p)[0]
            uN[sl] = out.detach().cpu().numpy()[:, 0]
            vN[sl] = g[:, 2].detach().cpu().numpy()
        return uN, vN


# ---------------------------------------------------------------------------------------------
# One window: choose the patches, train them, blend, hand over
# ---------------------------------------------------------------------------------------------
def march_window(fields, t_start, x, y, c2, y_if, arrival, device, tag):
    h = x[1] - x[0]
    t_end = t_start + DT
    X, Y = np.meshgrid(x, y, indexing="ij")
    region = wave_region(fields, x, arrival, t_end)

    # expected size of the correction at tau = DT is ~ DT^2/2 * |u_tt|, with u_tt = c^2 lap(u) + q
    src = max((np.abs(source_numpy(X, Y, t)) for t in (t_start, t_start + 0.5 * DT, t_end)),
              key=lambda a: a.max()) / U_REF
    forcing = 0.5 * DT ** 2 * (np.abs(fields[2]) + DT * np.abs(fields[3]) + src) * region
    peak = np.abs(fields[0]).max()

    x_lo, x_hi, y_lo, y_hi = float(x[0]), float(x[-1]), float(y[0]), float(y[-1])
    active = []                                        # (box, blend weight) of every tile that has a wave in it
    for px0, px1 in make_tiles(x_lo, x_hi):
        for py0, py1 in make_tiles(y_lo, y_hi):
            w = axis_weight(x, px0, px1, x_lo, x_hi)[:, None] * axis_weight(y, py0, py1, y_lo, y_hi)[None, :]
            own = (w > 0) & region
            if own.any() and forcing[own].max() >= args.quiet * peak:   # else the Taylor step already is the answer
                active.append(((px0, px1, py0, py1), w * own))

    uN, vN = np.zeros_like(fields[0]), np.zeros_like(fields[0])
    t_clock = time.time()
    for g0 in range(0, len(active), args.group):
        group = active[g0:g0 + args.group]
        model = PINN([b for b, _ in group], fields, t_start, x, y, region, y_if, device)
        model.train(args.steps, f"{tag} patches {g0 + 1}-{g0 + len(group)}/{len(active)}")
        for k, (_, w) in enumerate(group):
            ix, iy = np.nonzero(w)
            pu, pv = model.predict(k, x[ix], y[iy], t_end)
            uN[ix, iy] += w[ix, iy] * pu
            vN[ix, iy] += w[ix, iy] * pv
        del model
        torch.cuda.empty_cache()
    print(f"  [{tag}] {len(active)} patches trained in {time.time() - t_clock:.0f}s")

    # hand-off to the next window: VALUES of u and u_t only, masked to where the wave can be, low-passed.
    # A and B are re-taken on the grid (2nd/3rd derivatives of a PDE-trained network are not reliable).
    keep_region = region & (arrival <= t_end + 0.5 * PAD_T)
    keep = smoothstep(ndimage.distance_transform_edt(keep_region) * h / 0.03)
    u1 = lowpass((fields[0] + DT * fields[1] + uN) * keep, h, args.lowpass) * keep
    v1 = lowpass((fields[1] + vN) * keep, h, args.lowpass) * keep
    return build_fields(u1, v1, h, c2), region


# ---------------------------------------------------------------------------------------------
# Time marching
# ---------------------------------------------------------------------------------------------
def main():
    set_seed(args.seed)
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  arch={args.arch}  window={DT*1000:.0f} ms  steps={args.steps}  tile={args.tile} km "
          f"(patch = tile + {HALO:.2f} km halo)  pts/patch={args.patch_pts}+{args.patch_far}  group={args.group}  "
          f"lowpass={args.lowpass}  kink-feature={args.kink_feature}  kink-local={args.kink_local}  train-freq={args.train_freq}  causal-tol={args.causal_tol}")

    print("Running fine FD reference ...")
    fd = run_fd_solver(velocity=velocity_numpy, source_function=source_numpy, nx=FD_NX, ny=FD_NY, t_max=T_END + 0.01)
    h_t = fd["dt"]
    h = fd["x"][1] - fd["x"][0]
    s = 1.0 / U_REF

    # Absorbing edges: the networks solve on a domain padded by --pad-x (left, right) and --pad-top, with the
    # layers continued into the padding, so the wave simply leaves. Inside the physical domain that is what a
    # perfect absorbing edge gives. Everything is scored on the physical domain only (`inner`).
    npx, npy = int(round(args.pad_x / h)), int(round(args.pad_top / h))
    x = np.concatenate([fd["x"][0] - h * np.arange(npx, 0, -1), fd["x"], fd["x"][-1] + h * np.arange(1, npx + 1)])
    y = np.concatenate([fd["y"], fd["y"][-1] + h * np.arange(1, npy + 1)])
    inner = (slice(None), slice(npx, npx + FD_NX), slice(0, FD_NY))
    core = (slice(None), slice(npx + 3, npx + FD_NX - 3), slice(3, FD_NY - 3))   # Laplacian channels: skip the seam rows
    X, Y = np.meshgrid(x, y, indexing="ij")
    c2 = velocity_numpy(X, Y) ** 2
    if npx or npy:
        print(f"Padded domain: x in [{x[0]:.2f}, {x[-1]:.2f}], y in [{y[0]:.2f}, {y[-1]:.2f}] km ({len(x)} x {len(y)} cells)")
    y_if = INTERFACE_Y
    if args.kink_feature:
        # The FD grid switches c between two rows, so the reference's interface sits half a cell (2.2 m) below
        # INTERFACE_Y. The interface inputs are sharp, so the networks must use that same height.
        j = int(np.nonzero(np.diff(c2[0]))[0][0]) + 1
        y_if = float(0.5 * (y[j - 1] + y[j]))
        print(f"Interface height used by the networks: {y_if:.4f} km (grid rows {j - 1}/{j})")

    def pad(f):
        return np.pad(f, [(0, 0)] * (f.ndim - 2) + [(npx, npx), (0, npy)])

    def truth(t):                                        # FD fields, zero in the padding (only `inner` is meaningful)
        return build_fields(pad(field_at(fd, t) * s),
                            pad((field_at(fd, t + h_t) - field_at(fd, t - h_t)) * s / (2 * h_t)), h, c2)

    if args.resume:
        st = np.load(args.resume)
        fields, t_start = st["fields"], float(st["t"])
        if fields.shape[1:] == (FD_NX, FD_NY):           # state saved on the physical domain: valid while the wave is inside it
            fields = pad(fields)
        assert fields.shape[1:] == X.shape, f"state grid {fields.shape[1:]} does not match this run's grid {X.shape}"
        print(f"Resumed from {args.resume} at t = {t_start*1000:.0f} ms")
    else:
        t_start = T0
        fields = truth(t_start)                          # the only data the PINN gets: FD (u, u_t) at T0
    n_win = min(int(round((T_END - t_start) / DT)), args.n_win)

    tt_map = travel_time(x, y)
    sup = (np.abs(fields[:2]) > SUPPORT_FRAC * np.abs(fields[:2]).max(axis=(1, 2), keepdims=True)).any(0)
    arrival = tt_map + (t_start - tt_map[sup].max())     # = t_start at the IC front
    log = f"results/errors{args.tag}.csv"
    if not os.path.exists(log):
        with open(log, "w") as f:
            f.write("t_ms,rel_l2_pinn,rel_l2_taylor_only,rel_l2_frozen\n")
    for k in range(n_win):
        t_next = t_start + DT
        new, region = march_window(fields, t_start, x, y, c2, y_if, arrival, device, f"win {k+1}/{n_win}")
        ref = truth(t_next)
        taylor = (fields[0] + DT * fields[1]) * region
        errs = (rel_l2(new[inner][0], ref[inner][0]), rel_l2(taylor[inner[1:]], ref[inner][0]),
                rel_l2(fields[inner][0], ref[inner][0]))
        print(f"  t={t_next*1000:.0f} ms  relL2 vs fine FD: PINN={errs[0]:.4f}  Taylor-only={errs[1]:.4f}  frozen={errs[2]:.4f}")
        print("    channel relL2 vs FD truth: " + "  ".join(
            f"{n}={rel_l2(new[core][i], ref[core][i]):.3f}" for i, n in enumerate(["u", "v", "Lu", "Lv"])), flush=True)
        with open(log, "a") as f:
            f.write(f"{t_next*1000:.0f},{errs[0]:.4f},{errs[1]:.4f},{errs[2]:.4f}\n")
        fields, t_start = new, t_next
        np.savez(f"results/state_latest{args.tag}.npz", fields=fields, t=t_start)      # resume point
        t_ms = int(round(t_start * 1000))
        if args.snap_ms > 0 and t_ms % args.snap_ms == 0 and k < n_win - 1:         # the last one is saved below
            np.savez(f"results/pinn_{t_ms}ms{args.tag}.npz", x=fd["x"], y=fd["y"],
                     u=fields[inner][0] * U_REF, ref=field_at(fd, t_start))

    t_ms = int(round(t_start * 1000))
    u_in, ref_in = fields[inner][0], field_at(fd, t_start)
    print("\nFinal relL2 vs fine FD at %d ms: %.4f" % (t_ms, rel_l2(u_in, ref_in * s)))
    np.savez(f"results/state_{t_ms}ms{args.tag}.npz", fields=fields, t=t_start)
    np.savez(f"results/pinn_{t_ms}ms{args.tag}.npz", x=fd["x"], y=fd["y"], u=u_in * U_REF, ref=ref_in)
    print(f"Saved results/state_{t_ms}ms{args.tag}.npz (to resume) and results/pinn_{t_ms}ms{args.tag}.npz (to plot)")


if __name__ == "__main__":
    main()
