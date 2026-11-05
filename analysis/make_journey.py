"""Error against time for every stage of the project (numbers from the run logs, PROJECT_SUMMARY.txt updates 5-16, and
full_run/tables). -> full_run/images/report/journey.png.   python cluster/make_journey.py"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T = lambda a, n: list(range(a, a + 25 * n, 25))
single = T(175, 34), [0.0056, 0.0127, 0.0233, 0.0327, 0.0430, 0.0550, 0.0684, 0.0829, 0.0981, 0.1145, 0.1323, 0.1510, 0.1708,
                      0.1899, 0.2064, 0.2269, 0.2444, 0.2512, 0.262, 0.272, 0.279, 0.286, 0.295, 0.306, 0.322, 0.358, 0.472,
                      0.851, 1.54, 2.11, 2.64, 3.65, 5.42, 7.90]
patch = T(175, 26), [0.0035, 0.0073, 0.0106, 0.0143, 0.0179, 0.0217, 0.0249, 0.0281, 0.0316, 0.0354, 0.0392, 0.0435, 0.0480,
                     0.0524, 0.0562, 0.0673, 0.0822, 0.0884, 0.0934, 0.0975, 0.1034, 0.1077, 0.1153, 0.1274, 0.1432, 0.1688]
kink = T(625, 14), [0.0952, 0.1009, 0.1083, 0.1167, 0.1240, 0.1310, 0.1391, 0.1502, 0.1634, 0.1797, 0.2148, 0.2738, 0.3295, 0.3648]
pad = T(825, 8), [0.1631, 0.1753, 0.1878, 0.2024, 0.2179, 0.2316, 0.2449, 0.2594]
f42 = np.loadtxt("full_run/tables/errors_full_s42.csv", delimiter=",", skiprows=1, usecols=(0, 1))
f43 = np.loadtxt("full_run/tables/errors_full_s43.csv", delimiter=",", skiprows=1, usecols=(0, 1))

fig, ax = plt.subplots(figsize=(10, 5.8))
ax.semilogy(*single, "o-", ms=3, color="#c0392b", label="1  one network, hard IC, 25 ms windows")
ax.semilogy(*patch, "s-", ms=3, color="#e08a1e", label="2  + patch networks")
ax.semilogy(*kink, "^-", ms=3, color="#8e44ad", label="3  + interface inputs (from 600 ms)")
ax.semilogy(*pad, "d-", ms=3, color="#2a6fb0", label="4  + padded edges (from 800 ms)")
ax.fill_between(f42[:, 0], np.minimum(f42[:, 1], f43[:, 1]), np.maximum(f42[:, 1], f43[:, 1]), color="#2e8b57", alpha=.25)
ax.semilogy(f42[:, 0], f42[:, 1], "-", lw=2.2, color="#2e8b57", label="5  + fade of interface inputs (final, 2 seeds)")
ax.semilogy(f43[:, 0], f43[:, 1], "--", lw=1.5, color="#2e8b57")
ax.axhline(1, color="gray", ls=":"); ax.text(160, 1.12, "1.0 = predicting zero (soft-IC PINNs ended here)", color="gray", fontsize=9)
for tm, txt in ((500, "wave reaches\nthe interface"), (850, "wave reaches\nthe side edges")):
    ax.axvline(tm, color="darkgoldenrod", ls=":", lw=1); ax.text(tm + 6, 0.0035, txt, color="darkgoldenrod", fontsize=8.5)
ax.set_xlabel("time (ms)"); ax.set_ylabel("relative L2 error vs FD (log scale)"); ax.set_ylim(0.002, 12); ax.set_xlim(150, 1010)
ax.grid(alpha=.3, which="both"); ax.legend(loc="upper left", fontsize=9)
ax.set_title("How the error changed as each fix was added")
fig.tight_layout(); fig.savefig("full_run/images/report/journey.png", dpi=140)
print("saved")
