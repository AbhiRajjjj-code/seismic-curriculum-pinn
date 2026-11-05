"""The four failures met on the way, as PINN fields from the saved runs -> full_run/images/report/challenges.png
(top: PINN, bottom: |PINN - FD|).   python cluster/make_challenges.py"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cases = [("taylor_marching_150_1000", "A. One network, 1000 ms:\nnoise blows up"),
         ("patch_marching_150_800", "B. Patches, 800 ms:\nnoise on the interface"),
         ("patch_marching_150_950_kf", "C. + interface inputs, 950 ms:\nwave piles up at the edges"),
         ("patch_marching_150_1000_kfpad", "D. + padded edges, 1000 ms:\nsmooth offset above interface")]
fig, axs = plt.subplots(2, 4, figsize=(16, 10.5))
for i, (f, title) in enumerate(cases):
    d = np.load(f"results/{f}.npz"); x, y, u, ref = d["x"], d["y"], d["u"], d["ref"]
    ext = [x[0], x[-1], y[0], y[-1]]; v = np.abs(ref).max(); rel = np.linalg.norm(u - ref) / np.linalg.norm(ref)
    for r, (fld, cm, lo, hi) in enumerate([(u, "seismic", -v, v), (np.abs(u - ref), "hot", 0, 0.3 * v)]):
        a = axs[r, i]; a.imshow(fld.T, origin="lower", extent=ext, cmap=cm, vmin=lo, vmax=hi)
        a.axhline(1, color="yellow", ls="--", lw=0.8); a.set_xticks([]); a.set_yticks([])
    axs[0, i].set_title(title, fontsize=13); axs[1, i].set_xlabel(f"rel L2 = {rel:.3f}", fontsize=13)
axs[0, 0].set_ylabel("PINN (colour scale = FD peak)", fontsize=12); axs[1, 0].set_ylabel("|PINN - FD| (0 to 30% of FD peak)", fontsize=12)
fig.tight_layout(); fig.savefig("full_run/images/report/challenges.png", dpi=110)
print("saved")
