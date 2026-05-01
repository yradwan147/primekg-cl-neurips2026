#!/bin/bash
# =========================================================================
# Stratified eval — Wave 2: seeds 123, 456, 789, 1024 for all 6 cells.
# Submit after seed-42 wave validates cleanly. 24 jobs total.
# =========================================================================
set -e
mkdir -p slurm/slurm_logs results_stratified

SEEDS="123 456 789 1024"
METHODS="naive_sequential ewc joint_training"
DECODERS="DistMult RotatE"

for METHOD in $METHODS; do
  for DECODER in $DECODERS; do
    for SEED in $SEEDS; do
      JOBNAME="strat_${METHOD:0:4}_${DECODER:0:1}_s${SEED}"
      sbatch -J "$JOBNAME" slurm/run_baseline_stratified.sh "$METHOD" "$DECODER" "$SEED" 2>&1 | tail -1
    done
  done
done

echo ""
echo "Submitted 24 jobs; monitor with: squeue -u \$USER -n strat_naiv_D_s123,...,strat_join_R_s1024"
