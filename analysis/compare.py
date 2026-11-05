"""Old run vs boundary-fixed run: the side-by-side maps and the error-vs-time curve, into results/cluster/images.

    python cluster/compare.py                      # everything below
    python cluster/compare.py OLD.npz NEW.npz OUT  # one extra side-by-side map
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

IMG = "results/cluster/images"

# rel L2 per window (ms -> value). Old: laptop runs, PROJECT_SUMMARY updates 13 and 16 (no fade, 600 -> 800 -> 1000 ms).
# Fix: cluster steps 2 and 2b (--kink-local=0.2, same start state, seed 42).
OLD = dict(zip(range(625, 1001, 25), [0.0952, 0.1009, 0.1083, 0.1167, 0.1240, 0.1310, 0.1391, 0.1502,
                                      0.1631, 0.1753, 0.1878, 0.2024, 0.2179, 0.2316, 0.2449, 0.2594]))
FIX_CSV = ["results/cluster/errors_step2.csv", "results/cluster/errors_step2b.csv"]

# side-by-side maps: time -> (old run, fixed run); a time is skipped until both files exist
PAIRS = {
    800: ("results/patch_marching_150_800_kf.npz", "results/cluster/pinn_800ms_step2.npz"),
    875: ("results/cluster/pinn_875ms_step3_s43.npz", "results/cluster/pinn_875ms_step5b_plain_s43.npz"),  # both seed 43
    1000: ("results/patch_marching_150_1000_kfpad.npz", "results/cluster/pinn_1000ms_step2b.npz"),
}


def rel(u, ref):
    return np.linalg.norm(u - ref) / np.linalg.norm(ref)


def side_by_side(old_file, new_file, out, old_label="old run", new_label="boundary fix",
                 diff_label="old - fixed (what the fix removed)"):
    old, new = np.load(old_file), np.load(new_file)
    x, y, ref = old["x"], old["y"], old["ref"]
    assert np.allclose(ref, new["ref"]), "the two files are not at the same time"
    ext = [x[0], x[-1], y[0], y[-1]]
    vmax = np.abs(ref).max()
    emax = max(np.abs(old["u"] - ref).max(), np.abs(new["u"] - ref).max())
    panels = [
        (ref, "FD reference", "seismic", -vmax, vmax),
        (old["u"], f"PINN, {old_label}", "seismic", -vmax, vmax),
        (new["u"], f"PINN, {new_label}", "seismic", -vmax, vmax),
        (np.abs(old["u"] - ref), f"|error|, {old_label}  (rel L2 = {rel(old['u'], ref):.3f})", "hot", 0, emax),
        (np.abs(new["u"] - ref), f"|error|, {new_label}  (rel L2 = {rel(new['u'], ref):.3f})", "hot", 0, emax),
        (old["u"] - new["u"], diff_label, "seismic", -emax, emax),
    ]
    fig, axs = plt.subplots(2, 3, figsize=(15, 12))
    for ax, (f, title, cmap, lo, hi) in zip(axs.flat, panels):
        im = ax.imshow(f.T, origin="lower", extent=ext, cmap=cmap, vmin=lo, vmax=hi, aspect="equal")
        ax.axhline(1.0, color="yellow", ls="--", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Saved {out}  (old {rel(old['u'], ref):.4f}, fix {rel(new['u'], ref):.4f})")


def error_vs_time():
    fix = {}
    for f in FIX_CSV:
        for t, e, *_ in np.loadtxt(f, delimiter=",", skiprows=1, ndmin=2):
            fix[int(t)] = e
    ts = sorted(OLD)
    with open(f"{IMG}/old_vs_fix.csv", "w") as f:
        f.write("t_ms,old,fix,reduction_pct\n")
        for t in ts:
            if t in fix:
                f.write(f"{t},{OLD[t]:.4f},{fix[t]:.4f},{100 * (1 - fix[t] / OLD[t]):.1f}\n")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ts, [OLD[t] for t in ts], "o-", color="tab:red", label="old run (no fade)")
    ft = [t for t in ts if t in fix]
    ax.plot(ft, [fix[t] for t in ft], "o-", color="tab:blue", label="boundary fix (--kink-local=0.2)")
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("rel L2 vs FD")
    ax.set_title("Error over time, 600 ms -> 1 s")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{IMG}/error_vs_time_old_vs_fix.png", dpi=120)
    plt.close(fig)
    print(f"Saved {IMG}/error_vs_time_old_vs_fix.png and {IMG}/old_vs_fix.csv")


if __name__ == "__main__":
    os.makedirs(IMG, exist_ok=True)
    if len(sys.argv) == 4:
        side_by_side(*sys.argv[1:])
    else:
        error_vs_time()
        for t, (o, n) in PAIRS.items():
            if os.path.exists(o) and os.path.exists(n):
                side_by_side(o, n, f"{IMG}/compare_{t}ms_old_vs_fix.png")
