"""Report / slide figures -> full_run/images/report.   python cluster/make_figures.py  (from the project root)
setup.png (model + patches + snapshot), method.png (pipeline), ablation.png (stage-by-stage), time_strip.png (FD | PINN | error)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch

OUT = "full_run/images/report"
D = lambda s, t: np.load(f"full_run/data/seed{s}/pinn_{t}ms_full_s{s}.npz")
BLUE, RED, GREY, GREEN, ORANGE = "#2a6fb0", "#c0392b", "#7f8c8d", "#2e8b57", "#e08a1e"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# ---------------------------------------------------------------- setup
d = D(42, 400); x, y = d["x"], d["y"]; ext = [x[0], x[-1], y[0], y[-1]]
fig, ax = plt.subplots(1, 2, figsize=(14, 6.4))
a = ax[0]
a.add_patch(Rectangle((0, -1), 2, 2, color="#cfe3f5")); a.add_patch(Rectangle((0, 1), 2, 1, color="#f6dcc0"))
a.add_patch(Rectangle((-0.35, -1), 2.7, 3.15, fill=False, ls="--", ec=GREY, lw=1.3))
for i in np.arange(0, 2.01, 0.4): a.plot([i, i], [-1, 2], color="white", lw=0.8)
for j in np.arange(-1, 2.01, 0.4): a.plot([0, 2], [j, j], color="white", lw=0.8)
a.axhline(1, color="k", lw=2); a.plot(1.0, 0.5, "*", ms=22, color=RED, mec="k")
a.text(1.0, 0.57, "source (15 Hz Ricker)", ha="center", color=RED, fontsize=10)
a.text(1.0, -0.55, "c = 1 km/s", ha="center", fontsize=14); a.text(1.0, 1.5, "c = 2 km/s", ha="center", fontsize=14)
a.text(0.02, 1.03, "interface y = 1 km", fontsize=10)
a.text(-0.33, 2.2, "dashed: padded edges (0.35 km sides, 0.15 km top)", fontsize=9, color=GREY)
a.text(0.04, -0.97, "white grid: 0.4 km tiles, one small network each", fontsize=9)
a.set_xlim(-0.45, 2.45); a.set_ylim(-1.1, 2.35); a.set_aspect("equal"); a.set_xlabel("x (km)"); a.set_ylabel("y (km)")
a.set_title("Problem: 2D acoustic wave, two layers")
a = ax[1]; v = np.abs(d["ref"]).max()
im = a.imshow(d["ref"].T, origin="lower", extent=ext, cmap="seismic", vmin=-v, vmax=v)
a.axhline(1, color="yellow", ls="--", lw=1); a.set_xlabel("x (km)"); a.set_title("FD reference at t = 400 ms (the target)")
plt.colorbar(im, ax=a, fraction=0.04)
fig.tight_layout(); fig.savefig(f"{OUT}/setup.png", dpi=130); plt.close(fig)

# ---------------------------------------------------------------- method
fig, ax = plt.subplots(figsize=(15, 7.2)); ax.set_xlim(0, 15); ax.set_ylim(0, 7.2); ax.axis("off")
def box(x0, y0, w, h, txt, fc, fs=10.5, bold=False):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.05,rounding_size=0.15", fc=fc, ec="k", lw=1.2))
    ax.text(x0 + w / 2, y0 + h / 2, txt, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")
def arrow(p, q): ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=18, lw=1.6, color="k"))
box(0.2, 5.3, 2.7, 1.5, "FD snapshot at 150 ms\n(u, u_t) = the only data", "#e8e8e8", bold=True)
box(3.6, 5.3, 3.2, 1.5, "Cut time into 25 ms windows\n(curriculum: each window starts\nfrom the previous prediction)", "#dbe9f7")
box(7.5, 5.3, 3.4, 1.5, "Hard initial condition\nu = u0 + τ·u_t0 + (τ/25ms)²·N(x,y,τ)\nstarts exact for any network N", "#dff0e0", fs=9.5)
box(11.5, 5.3, 3.3, 1.5, "Many small networks\none per 0.4 km tile (+ halo),\nFourier-feature MLP each", "#f9e7cc")
arrow((2.9, 6.05), (3.6, 6.05)); arrow((6.8, 6.05), (7.5, 6.05)); arrow((10.9, 6.05), (11.5, 6.05))
box(11.5, 2.9, 3.3, 1.7, "Training: only the wave-equation\nresidual (no wave data), Adam,\n10 000 steps per window", "#f6d6d6")
box(7.5, 2.9, 3.4, 1.7, "Blend tiles (weights sum to 1),\nlow-pass filter at 35 cycles/km\n→ state (u, u_t) at window end", "#dbe9f7")
box(3.6, 2.9, 3.2, 1.7, "Hand the state to the next\nwindow. Repeat 34 times:\n150 ms → 1000 ms", "#dff0e0", bold=True)
arrow((13.15, 5.3), (13.15, 4.6)); arrow((11.5, 3.75), (10.9, 3.75)); arrow((7.5, 3.75), (6.8, 3.75))
ax.add_patch(FancyArrowPatch((5.2, 4.6), (5.2, 5.3), arrowstyle="-|>", mutation_scale=18, lw=1.6, color="k", connectionstyle="arc3,rad=0"))
ax.text(5.35, 4.95, "next window", fontsize=9)
box(0.2, 0.3, 4.6, 1.9, "Two tricks for the interface\n• extra inputs r²/(1+r²), r³/(1+r³) that know\n  where the layer boundary is\n• faded out above it (exp(-(r/0.2 km)²))", "#f9e7cc", fs=10)
box(5.3, 0.3, 4.2, 1.9, "Padded edges\nsolve on a larger domain so the\nwave leaves instead of piling up\n(score only the real domain)", "#dbe9f7", fs=10)
box(10.0, 0.3, 4.8, 1.9, "Score: relative L2 error against a\nfinite-difference run (never used in\ntraining, except the 150 ms start)", "#e8e8e8", fs=10)
ax.set_title("Curriculum PINN: how one run works", fontsize=15)
fig.savefig(f"{OUT}/method.png", dpi=130, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- ablation (measured values only, see PROJECT_SUMMARY.txt / full_run/tables)
fig, axs = plt.subplots(1, 2, figsize=(14, 5.2))
def bars(a, labels, vals, cols, title):
    pos = np.arange(len(vals))[::-1]
    a.barh(pos, vals, color=cols, ec="k")
    a.set_yticks(pos); a.set_yticklabels(labels); a.set_xscale("log"); a.set_xlim(0.05, 30)
    for p_, v in zip(pos, vals): a.text(v * 1.08, p_, f"{v:.3f}" if v < 1 else f"{v:.2f}", va="center", fontsize=10)
    a.axvline(1, color=GREY, ls=":"); a.set_title(title); a.set_xlabel("relative L2 error vs FD (log scale; 1 = predicting zero)")
bars(axs[0], ["one network + hard IC", "+ patches", "+ interface inputs", "+ padded edges + fade\n(final, mean of 2 seeds)"],
     [0.358, 0.169, 0.150, 0.123], [RED, ORANGE, ORANGE, GREEN], "At 800 ms")
bars(axs[1], ["one network + hard IC", "patches + interface inputs\n(no padding, at 950 ms)", "+ padded edges", "+ fade of interface inputs\n(final, mean of 2 seeds)"],
     [7.90, 0.365, 0.259, 0.161], [RED, ORANGE, ORANGE, GREEN], "At 1000 ms")
fig.suptitle("What each ingredient bought. Not shown: soft-IC PINNs, ~1.0 from the first window (they predict ~zero)", fontsize=12)
fig.tight_layout(); fig.savefig(f"{OUT}/ablation.png", dpi=130); plt.close(fig)

# ---------------------------------------------------------------- time strip
ts = [200, 400, 600, 800, 1000]
fig, axs = plt.subplots(3, 5, figsize=(17, 11.5))
emax = max(np.abs(D(42, t)["u"] - D(42, t)["ref"]).max() for t in ts)
for i, t in enumerate(ts):
    d = D(42, t); v = np.abs(d["ref"]).max(); e = np.abs(d["u"] - d["ref"]); rel = np.linalg.norm(d["u"] - d["ref"]) / np.linalg.norm(d["ref"])
    for r, (f, cm, lo, hi) in enumerate([(d["ref"], "seismic", -v, v), (d["u"], "seismic", -v, v), (e, "hot", 0, emax)]):
        a = axs[r, i]; im = a.imshow(f.T, origin="lower", extent=ext, cmap=cm, vmin=lo, vmax=hi); a.axhline(1, color="yellow", ls="--", lw=0.7)
        a.set_xticks([]); a.set_yticks([])
        if r == 0: a.set_title(f"t = {t} ms", fontsize=14)
        if r == 2: a.set_xlabel(f"rel L2 = {rel:.3f}", fontsize=12)
        if i == 0: a.set_ylabel(["FD reference", "PINN", "|error| (one scale)"][r], fontsize=13)
fig.suptitle("Final run, seed 42: the wave crosses the interface (y = 1 km, dashed) and the PINN follows", fontsize=15)
fig.tight_layout(); fig.savefig(f"{OUT}/time_strip.png", dpi=110); plt.close(fig)
print("saved", OUT)

# ---------------------------------------------------------------- error vs time, two references (needs cluster/rescore_fine.py first)
r = np.loadtxt("full_run/tables/rescored_vs_fine_fd.csv", delimiter=",", skiprows=1); t = r[:, 0]
fig, a = plt.subplots(figsize=(9, 5.5))
a.fill_between(t, r[:, 1], r[:, 2], color="tab:blue", alpha=.15); a.plot(t, r[:, 1], color="tab:blue", label="vs our FD reference (451x676), seed 42"); a.plot(t, r[:, 2], "--", color="tab:blue", label="seed 43")
a.fill_between(t, r[:, 3], r[:, 4], color="tab:green", alpha=.15); a.plot(t, r[:, 3], color="tab:green", label="vs 2x finer FD (901x1351), seed 42"); a.plot(t, r[:, 4], "--", color="tab:green", label="seed 43")
a.plot(t, r[:, 5], ":", color="gray", label="reference FD vs finer FD (the reference's own error)")
a.set_xlabel("t (ms)"); a.set_ylabel("relative L2 error"); a.grid(alpha=.3); a.legend(); a.set_title("Final run scored against two FD references")
fig.tight_layout(); fig.savefig(f"{OUT}/error_vs_time_two_references.png", dpi=130); plt.close(fig)
