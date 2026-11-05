"""Rescore the final run against the 2x finer FD (901 x 1351), at every 50 ms snapshot.
    python cluster/rescore_fine.py      (run from the project root, ~minutes on a laptop CPU)
Writes full_run/tables/rescored_vs_fine_fd.csv and full_run/data/fine_fd_snapshots.npz.
"""
import glob, re, sys, os
import numpy as np
sys.path.insert(0, "cluster")
from fd_solver import run_fd_solver

# same set-up as wave_pinn.py (copied so that this script needs no torch)
FD_NX, FD_NY, FREQ, SIG, X0, Y0 = 451, 676, 15.0, 0.03, 1.0, 0.5
velocity_numpy = lambda x, y: np.where(np.asarray(y) < 1.0, 1.0, 2.0)
def source_numpy(x, y, t):
    term = np.pi * FREQ * (t - 1.0 / FREQ)
    return np.exp(-((x - X0) ** 2 + (y - Y0) ** 2) / (2.0 * SIG ** 2)) * ((1.0 - 2.0 * term ** 2) * np.exp(-term ** 2))

OUT = "full_run/data/fine_fd_snapshots.npz"
if not os.path.exists(OUT):
    fine = run_fd_solver(velocity=velocity_numpy, source_function=source_numpy, nx=2 * FD_NX - 1, ny=2 * FD_NY - 1, t_max=1.01)
    snaps = {}
    for f in glob.glob("full_run/data/seed42/pinn_*ms_full_s42.npz"):
        t = int(re.search(r"pinn_(\d+)ms", f).group(1)); n = t / 1000 / fine["dt"]; k = int(np.floor(n)); w = n - k
        snaps[str(t)] = ((1 - w) * fine["wavefield"][k] + w * fine["wavefield"][k + 1])[::2, ::2]
    np.savez_compressed(OUT, **snaps)
fine = np.load(OUT)
rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)
with open("full_run/tables/rescored_vs_fine_fd.csv", "w") as f:
    f.write("t_ms,seed42_vs_ref,seed43_vs_ref,seed42_vs_fine,seed43_vs_fine,ref_vs_fine\n")
    for t in sorted(int(k) for k in fine.files):
        row = []
        d = {s: np.load(f"full_run/data/seed{s}/pinn_{t}ms_full_s{s}.npz") for s in (42, 43)}
        ref, fi = d[42]["ref"], fine[str(t)]
        row = [rel(d[42]["u"], ref), rel(d[43]["u"], ref), rel(d[42]["u"], fi), rel(d[43]["u"], fi), rel(ref, fi)]
        f.write(f"{t}," + ",".join(f"{v:.4f}" for v in row) + "\n")
print(open("full_run/tables/rescored_vs_fine_fd.csv").read())
