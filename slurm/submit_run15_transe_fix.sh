#!/bin/bash
# Rerun TransE baselines with correct L1 scoring (was using L2)
# Only Naive + EWC for now to check if AP changes significantly
set -e
mkdir -p slurm/slurm_logs

SEEDS="42 123 456 789 1024"

echo "=== TransE scoring fix: Naive + EWC ==="
for SEED in $SEEDS; do
    sbatch -J "ns_trE_s${SEED}" slurm/run_baseline.sh naive_sequential TransE "$SEED"
    sbatch -J "ewc_trE_s${SEED}" slurm/run_baseline.sh ewc TransE "$SEED"
done
echo "=== 10 TransE fix jobs submitted ==="
