#!/bin/bash
# Run 11: New baselines — SI, Distillation, MIR Replay
# Both TransE and DistMult decoders
# Plus DistMult re-runs of existing baselines
#
# Total: 50 jobs
#   - 4 existing baselines × 5 seeds × 1 decoder (DistMult) = 20 jobs
#   - 3 new baselines × 5 seeds × 2 decoders = 30 jobs
#
# Usage: bash slurm/submit_run11_new_baselines.sh

set -e

SEEDS="42 123 456 789 1024"

echo "=== Run 11: New Baselines + DistMult Re-runs ==="
echo ""

JOB_COUNT=0

# --- Part A: DistMult re-runs of existing baselines ---
echo "--- Part A: DistMult re-runs of existing baselines ---"
for BASELINE in naive_sequential joint_training ewc experience_replay; do
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

# --- Part B: New baselines (both decoders) ---
echo ""
echo "--- Part B: New baselines ---"
for BASELINE in si distillation mir_replay; do
    for MODEL in TransE DistMult; do
        for SEED in $SEEDS; do
            SHORT=$(echo $BASELINE | cut -c1-3)
            MOD=$(echo $MODEL | cut -c1-2)
            JOB_NAME="${SHORT}_${MOD}_s${SEED}"
            echo "  $BASELINE $MODEL seed=$SEED"
            sbatch -J $JOB_NAME \
                -o slurm/slurm_logs/${JOB_NAME}_%J.out \
                slurm/run_baseline.sh $BASELINE $MODEL $SEED
            JOB_COUNT=$((JOB_COUNT + 1))
        done
    done
done

echo ""
echo "Submitted $JOB_COUNT jobs total."
echo "Monitor with: squeue -u \$USER"
