#!/bin/bash
# Reproduce the simulation runs behind Figures 5-7.
#
# Produces, for 5 random seeds each:
#   - 13_7_3_seed{1..5}    : preset (M13, M7, M3)          -> Fig 5, Fig 6, Fig 7
#   - 13_5_3_seed{1..5}    : preset (M13, M5, M3)          -> Fig 7
#   - fixed_weights_seed{1..5} : (M13, M7, M3) control     -> Fig 7
#
# Each run performs 1000 simulation steps and writes results to
#   particle_visualizations/<run_name>/   (reconstruction errors, PCA snapshots)
#   saved_particles/<run_name>/           (population state per step)
#
# Usage:
#   bash run_experiments.sh
#
# Set PYTHON to your interpreter if needed, e.g.:
#   PYTHON="conda run -n SAEH_env python" bash run_experiments.sh
set -e

PYTHON="${PYTHON:-python}"
MAX_ITER="${MAX_ITER:-1000}"
SEEDS=(1 2 3 4 5)

mkdir -p logs

run_one () {
    local run_name="$1"; shift
    echo "=== ${run_name} ($(date +%H:%M:%S)) ==="
    rm -rf "particle_visualizations/${run_name}" "saved_particles/${run_name}"
    $PYTHON particle.py --run_name "${run_name}" --max_iter "${MAX_ITER}" "$@" \
        > "logs/${run_name}.log" 2>&1
}

for s in "${SEEDS[@]}"; do
    run_one "13_7_3_seed${s}"       --seed "$s" --preset 13-7-3
    run_one "13_5_3_seed${s}"       --seed "$s" --preset 13-5-3
    run_one "fixed_weights_seed${s}" --seed "$s" --preset 13-7-3 --fixed_weights
done

echo "All runs complete."
