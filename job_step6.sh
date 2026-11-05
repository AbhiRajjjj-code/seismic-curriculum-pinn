#!/bin/bash
#SBATCH --job-name=step6_full
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.log
# Step 6: the final clean run, 150 ms -> 1 s with one recipe from the start. Seed as argument: sbatch job_step6.sh 42
#   fading interface inputs (--kink-local=0.2) and padded edges throughout;
#   10000 steps per window up to 600 ms, then 20000 (the late windows are where the error is);
#   no causal weights, fixed Fourier frequencies (neither gave a measurable gain in steps 4, 4b, 5, 5b).
# One snapshot (pinn_<t>ms_full_s<seed>.npz) every 50 ms; one error table for the whole run.
S=${1:?seed}
export PYTHONUNBUFFERED=1
RECIPE="--kink-feature=1 --kink-local=0.2 --pad-x=0.35 --pad-top=0.15 --seed=$S --snap-ms=50 --tag=_full_s$S"
venv/bin/python wave_pinn.py --t-end=0.6 $RECIPE || exit 1
venv/bin/python wave_pinn.py --resume=results/state_600ms_full_s$S.npz --t-end=1.0 --steps=20000 $RECIPE || exit 1
for f in results/pinn_*ms_full_s$S.npz; do venv/bin/python plot_results.py $f; done
venv/bin/python plot_results.py --curve results/errors_full_s$S.csv
venv/bin/python analyse.py results/pinn_*ms_full_s$S.npz
