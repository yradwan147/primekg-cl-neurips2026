#!/bin/bash
# Run 14 rerun: NC for SI/Distillation/MIR (failed due to missing method choices)
# Fixed: added si, distillation, mir_replay to run_nc.py METHODS list
# KGQA jobs are still running — do NOT resubmit those
# Total: 15 jobs

mkdir -p slurm/slurm_logs results

SEEDS=(42 123 456 789 1024)

echo "=== Run 14 Rerun: NC for new baselines (15 jobs) ==="

for METHOD in si distillation mir_replay; do
    for SEED in "${SEEDS[@]}"; do
        sbatch -J nc_${METHOD}_s${SEED} slurm/run_nc.sh $METHOD $SEED
    done
done

echo "Done! Submitted 15 NC jobs."
echo "Monitor: squeue -u \$USER"
