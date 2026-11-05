"""
Figures for the wave PINN.

Snapshot (fine FD | PINN | absolute error) from a file written by wave_pinn.py:
    python plot_results.py results/pinn_1000ms_stage3.npz
Error against time from one or more error tables (stages of one march are joined):
    python plot_results.py --curve results/errors_stage1.csv results/errors_stage2.csv results/errors_stage3.csv
Each figure is saved next to its (first) input file as a .png.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_snapshot(path):
    d = np.load(path)
    x, y, pred, ref = d["x"], d["y"], d["u"], d["ref"]
    rel = np.linalg.norm(pred - ref) / np.linalg.norm(ref)
    vmax = np.abs(ref).max()
    extent = [x.min(), x.max(), y.min(), y.max()]
    t_ms = [p[:-2] for p in path.replace(".npz", "").replace("\\", "/").split("/")[-1].split("_") if p.endswith("ms")][0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for ax, field, cmap, vmin, title in [
        (axes[0], ref, "seismic", -vmax, f"fine FD (reference), t = {t_ms} ms"),
        (axes[1], pred, "seismic", -vmax, "PINN"),
        (axes[2], np.abs(pred - ref), "hot", 0.0, f"|PINN - FD|   rel L2 = {rel:.3f}"),
    ]:
        im = ax.imshow(field.T, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.axhline(1.0, color="yellow", lw=0.8, ls="--")            # the interface
        ax.set_xlabel("x (km)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046)
    axes[0].set_ylabel("y (km)")
    plt.tight_layout()
    out = path.replace(".npz", ".png")
    plt.savefig(out, dpi=110)
    print(f"Saved {out}  (rel L2 = {rel:.4f})")


def plot_curve(paths):
    rows = np.concatenate([np.atleast_2d(np.genfromtxt(p, delimiter=",", skip_header=1)) for p in paths])
    rows = rows[np.argsort(rows[:, 0])]
    t, err = rows[:, 0], rows[:, 1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(t, err, "o-", color="tab:blue", label="curriculum PINN (hard IC, patches, 25 ms windows)")
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.text(t.min(), 1.1, "1.0 = as bad as predicting zero", color="gray", fontsize=9)
    for t_mark, text in ((500, "wave reaches the interface"), (850, "wave reaches the side edges")):
        if t.min() < t_mark < t.max():
            ax.axvline(t_mark, color="darkgoldenrod", ls=":", lw=1.2)
            ax.text(t_mark - 5, 0.5, text, color="darkgoldenrod", fontsize=9, ha="right")
    ax.set_ylim(top=2.0)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("relative L2 error vs fine FD")
    ax.set_title("PINN error against time, started from the 150 ms snapshot")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()
    out = paths[0].replace(".csv", ".png")
    plt.savefig(out, dpi=120)
    print(f"Saved {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "--curve":
        plot_curve(sys.argv[2:])
    else:
        for p in sys.argv[1:]:
            plot_snapshot(p)
