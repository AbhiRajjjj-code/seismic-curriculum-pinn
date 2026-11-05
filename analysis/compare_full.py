"""Final run (step 6) against the earlier runs: figures and a table into full_run/images/compare and full_run/tables.

    python cluster/compare_full.py
"""
import numpy as np
import matplotlib.pyplot as plt

from compare import OLD, FIX_CSV, side_by_side

F = "full_run"
CMP = f"{F}/images/compare"


def table(path):
    return {int(t): e for t, e, *_ in np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)}


s42, s43 = table(f"{F}/tables/errors_full_s42.csv"), table(f"{F}/tables/errors_full_s43.csv")
piece = {}
for f in FIX_CSV:
    piece.update(table(f))

# error against time
fig, ax = plt.subplots(figsize=(9, 5.5))
ts = sorted(s42)
ax.fill_between(ts, [min(s42[t], s43[t]) for t in ts], [max(s42[t], s43[t]) for t in ts], color="tab:blue", alpha=0.15)
ax.plot(ts, [s42[t] for t in ts], "-", color="tab:blue", label="final run, seed 42")
ax.plot(ts, [s43[t] for t in ts], "--", color="tab:blue", label="final run, seed 43")
ax.plot(sorted(piece), [piece[t] for t in sorted(piece)], "o", ms=4, color="tab:green",
        label="earlier boundary-fix runs (600 ms -> 1 s, pieced together)")
ax.plot(sorted(OLD), [OLD[t] for t in sorted(OLD)], "o-", ms=4, color="tab:red", label="old recipe (no fade)")
ax.set_xlabel("t (ms)")
ax.set_ylabel("rel L2 vs FD")
ax.set_title("Final run, 150 ms -> 1 s")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(f"{CMP}/error_vs_time_final.png", dpi=120)
plt.close(fig)

with open(f"{F}/tables/final_vs_earlier.csv", "w") as f:
    f.write("t_ms,seed42,seed43,mean,old_recipe,earlier_fix\n")
    for t in ts:
        f.write(f"{t},{s42[t]:.4f},{s43[t]:.4f},{(s42[t] + s43[t]) / 2:.4f},"
                f"{OLD.get(t, float('nan')):.4f},{piece.get(t, float('nan')):.4f}\n")

# maps: old recipe vs final (seed 42, same seed as the old runs), and the two seeds against each other
side_by_side("results/patch_marching_150_800_kf.npz", f"{F}/data/seed42/pinn_800ms_full_s42.npz",
             f"{CMP}/compare_800ms_old_vs_final.png", "old recipe", "final run (seed 42)", "old - final")
side_by_side("results/patch_marching_150_1000_kfpad.npz", f"{F}/data/seed42/pinn_1000ms_full_s42.npz",
             f"{CMP}/compare_1000ms_old_vs_final.png", "old recipe", "final run (seed 42)", "old - final")
side_by_side(f"{F}/data/seed42/pinn_1000ms_full_s42.npz", f"{F}/data/seed43/pinn_1000ms_full_s43.npz",
             f"{CMP}/compare_1000ms_seed42_vs_seed43.png", "seed 42", "seed 43", "seed 42 - seed 43")
print(f"Saved {CMP}/error_vs_time_final.png and {F}/tables/final_vs_earlier.csv")
