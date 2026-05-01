#!/bin/bash
# =========================================================================
# Stratified eval on persistent / removed / added test triples
#
# Focused scope (per supervisor request): a representative slice of the
# main-results matrix rather than the full 6×4 grid. Retrains each (method,
# decoder) from scratch with --eval-stratified, which writes per-stratum
# MRR (persistent / removed / added) to the JSON alongside the standard
# results_matrix.
#
# Methods:   Naive Sequential, EWC, Joint Training (oracle)
# Decoders:  DistMult (CMKL's home family) and RotatE (overall winner)
# Seeds:     42, 123, 456, 789, 1024
# Total:     3 × 2 × 5 = 30 jobs
# =========================================================================
set -e
mkdir -p slurm/slurm_logs results_stratified

SEEDS="42 123 456 789 1024"
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
echo "Submitted $(echo $METHODS $DECODERS $SEEDS | wc -w) jobs; monitor with:"
echo "  squeue -u \$USER -o '%.10i %.20j %.8T %.10M'"
echo "Results land in: results_stratified/<method>_<decoder>_seed<seed>.json"
