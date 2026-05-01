#!/bin/bash
# Run 11 RESUBMIT: Only the DistMult jobs that OOM'd
# 35 jobs total:
#   - 4 existing baselines × 5 seeds × DistMult = 20
#   - 3 new baselines × 5 seeds × DistMult = 15
#
# Usage: bash slurm/submit_run11_distmult_only.sh

set -e

SEEDS="42 123 456 789 1024"

echo "=== Run 11 RESUBMIT: DistMult only (OOM-fixed) ==="
echo ""

JOB_COUNT=0

for BASELINE in naive_sequential joint_training ewc experience_replay si distillation mir_replay; do
    for SEED in $SEEDS; do
        SHORT=$(echo $BASELINE | cut -c1-3)
        JOB_NAME="${SHORT}_dm_s${SEED}"
        echo "  $BASELINE DistMult seed=$SEED"
        sbatch -J $JOB_NAME \
            -o slurm/slurm_logs/${JOB_NAME}_%J.out \
            slurm/run_baseline.sh $BASELINE DistMult $SEED
        JOB_COUNT=$((JOB_COUNT + 1))
    done
done

echo ""
echo "Submitted $JOB_COUNT jobs."
