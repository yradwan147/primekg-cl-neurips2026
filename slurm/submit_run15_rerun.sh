#!/bin/bash
# Resubmit failed RotatE jobs (ComplexFloat evaluation fix applied)
set -e
mkdir -p slurm/slurm_logs

SEEDS="42 123 456 789 1024"

echo "=== Resubmitting failed RotatE + pending ComplEx jobs ==="

for SEED in $SEEDS; do
    echo "Submitting: Naive RotatE seed=$SEED"
    sbatch -J "ns_rot_s${SEED}" slurm/run_baseline.sh naive_sequential RotatE "$SEED"
    echo "Submitting: EWC RotatE seed=$SEED"
    sbatch -J "ewc_rot_s${SEED}" slurm/run_baseline.sh ewc RotatE "$SEED"
done

echo ""
echo "=== 10 RotatE resubmitted ==="
