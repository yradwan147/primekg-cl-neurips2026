#!/bin/bash
# =============================================================
# Run 16d-g: Fill the decoder × CL strategy matrix
#
# Currently have:
#   Naive × all 4 decoders ✓
#   EWC × all 4 decoders (RotatE pending)
#   Joint × TransE, DistMult (adding RotatE, ComplEx in run16a-c)
#
# This run adds:
#   SI, Distillation, Experience Replay, MIR × {RotatE, ComplEx}
# = 4 methods × 2 decoders × 5 seeds = 40 jobs
# =============================================================
set -e
mkdir -p slurm/slurm_logs

SEEDS="42 123 456 789 1024"
METHODS="si distillation experience_replay mir_replay"
DECODERS="RotatE ComplEx"

for METHOD in $METHODS; do
  # short abbreviation for job name
  case $METHOD in
    si) SHORT="si" ;;
    distillation) SHORT="dist" ;;
    experience_replay) SHORT="er" ;;
    mir_replay) SHORT="mir" ;;
  esac

  for DEC in $DECODERS; do
    DSHORT=$(echo $DEC | cut -c1-3 | tr '[:upper:]' '[:lower:]')
    for SEED in $SEEDS; do
      JOBNAME="${SHORT}_${DSHORT}_s${SEED}"
      sbatch -J "$JOBNAME" slurm/run_baseline.sh "$METHOD" "$DEC" "$SEED" 2>&1 | tail -1
    done
  done
done
