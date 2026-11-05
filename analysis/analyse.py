"""
Where is the error of a PINN snapshot? Usage: python analyse.py results/pinn_800ms_x.npz [more files]
Prints, for each file: total relative L2; share of the squared error and local error above the
interface (y > 1.1 km), in the strip 0.9..1.1 km and below 0.9 km; the smooth offset above the
interface (mean error there / rms of the reference there); and the share of the error energy
below 2 cycles/km. Also appends one line per file to results/analysis.csv.
"""
import os
import sys

import numpy as np

print(f"{'file':42s} {'total':>6s} | {'above: share  local':>20s} | {'strip: share  local':>20s} | "
      f"{'below: share  local':>20s} | {'offset':>6s} {'<2cyc':>6s}")
rows = []
for path in sys.argv[1:]:
    d = np.load(path)
    x, y, u, ref = d["x"], d["y"], d["u"], d["ref"]
    err = u - ref
    total = np.linalg.norm(err) / np.linalg.norm(ref)
    zones = {"above": y > 1.1, "strip": (y >= 0.9) & (y <= 1.1), "below": y < 0.9}
    out = [total]
    for m in zones.values():
        out += [np.sum(err[:, m] ** 2) / np.sum(err ** 2), np.linalg.norm(err[:, m]) / np.linalg.norm(ref[:, m])]
    up = zones["above"]
    offset = err[:, up].mean() / np.sqrt(np.mean(ref[:, up] ** 2))
    h = x[1] - x[0]
    E = np.abs(np.fft.fft2(err)) ** 2
    kx, ky = np.fft.fftfreq(len(x), h), np.fft.fftfreq(len(y), h)
    low = np.hypot(*np.meshgrid(kx, ky, indexing="ij")) < 2.0
    out += [offset, E[low].sum() / E.sum()]
    rows.append((os.path.basename(path), out))
    print(f"{os.path.basename(path):42s} {out[0]:6.4f} | {out[1]:9.2f} {out[2]:9.3f} | {out[3]:9.2f} {out[4]:9.3f} | "
          f"{out[5]:9.2f} {out[6]:9.3f} | {out[7]:+6.3f} {out[8]:6.2f}")
os.makedirs("results", exist_ok=True)
new = not os.path.exists("results/analysis.csv")
with open("results/analysis.csv", "a") as f:
    if new:
        f.write("file,total,above_share,above_local,strip_share,strip_local,below_share,below_local,offset_above,low_freq_share\n")
    for name, out in rows:
        f.write(name + "," + ",".join(f"{v:.4f}" for v in out) + "\n")
