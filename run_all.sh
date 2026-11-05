#!/bin/sh
# The final run, 150 ms -> 1 s (about 10.7 hours on one cluster GPU). The defaults of wave_pinn.py are the final recipe.
# Seed as argument:  sh run_all.sh 42
set -e
S=${1:-42}
python wave_pinn.py --t-end=0.6 --seed=$S --snap-ms=50 --tag=_full_s$S
python wave_pinn.py --resume=results/state_600ms_full_s$S.npz --t-end=1.0 --steps=20000 --seed=$S --snap-ms=50 --tag=_full_s$S
for f in results/pinn_*ms_full_s$S.npz; do python plot_results.py $f; done
python plot_results.py --curve results/errors_full_s$S.csv
