#!/bin/bash
# =============================================================
# Run 15c: IncDE baseline on PrimeKG-CL
# Purpose: Add AAAI 2024 CKGE baseline to both papers
# Total: 5 seeds
# =============================================================

set -e
mkdir -p slurm/slurm_logs

SEEDS="42 123 456 789 1024"

echo "=== Run 15c: IncDE on PrimeKG-CL (10 snapshots) ==="
echo "Jobs: 5 seeds"
echo ""

for SEED in $SEEDS; do
    JOBNAME="incde_s${SEED}"
    echo "Submitting: IncDE seed=${SEED} -> ${JOBNAME}"
    sbatch -J "$JOBNAME" slurm/run_incde.sh "$SEED"
done

echo ""
echo "=== All 5 IncDE jobs submitted ==="
echo "Monitor: squeue -u \$USER --format='%.10i %.14j %.8T %.10M'"
echo "Results in: external/IncDE/logs/ and results/incde_TransE_seed*.log"
