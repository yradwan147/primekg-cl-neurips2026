#!/bin/bash
# Run 11: Re-run all baselines with DistMult decoder for fair comparison
# Total: 20 jobs (4 baselines × 5 seeds)
# LKGE excluded — requires TransE backbone
#
# Usage: bash slurm/submit_run11_distmult.sh

set -e

SEEDS="42 123 456 789 1024"
BASELINES="naive_sequential joint_training ewc experience_replay"

echo "=== Run 11: DistMult baselines ==="
echo "Baselines: $BASELINES"
echo "Seeds: $SEEDS"
echo ""

JOB_COUNT=0

for BASELINE in $BASELINES; do
    for SEED in $SEEDS; do
        # Short name for job
        SHORT=$(echo $BASELINE | cut -c1-3)
        JOB_NAME="${SHORT}_dm_s${SEED}"

        echo "Submitting: $BASELINE DistMult seed=$SEED  (job: $JOB_NAME)"
        sbatch -J $JOB_NAME \
            -o slurm/slurm_logs/${JOB_NAME}_%J.out \
            slurm/run_baseline.sh $BASELINE DistMult $SEED

        JOB_COUNT=$((JOB_COUNT + 1))
    done
done

echo ""
echo "Submitted $JOB_COUNT jobs total."
echo "Monitor with: squeue -u \$USER"
echo "Results will be in: results/${BASELINE}_DistMult_seed*.json"
