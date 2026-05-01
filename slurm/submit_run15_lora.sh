#!/bin/bash
# =============================================================
# Run 15b: LoRA text adapter experiment for Paper B
# Purpose: Address greedy modality problem by making text pathway trainable
# Tests: CMKL-MoE-DistMult with LoRA ranks 16, 64
# Total: 2 ranks × 5 seeds = 10 jobs
# =============================================================

set -e
mkdir -p slurm/slurm_logs

SEEDS="42 123 456 789 1024"
LORA_RANKS="16 64"

echo "=== Run 15b: LoRA text adapter experiments ==="
echo "Jobs: 2 ranks × 5 seeds = 10 jobs"
echo ""

for RANK in $LORA_RANKS; do
  for SEED in $SEEDS; do
    JOBNAME="cmkl_lora${RANK}_s${SEED}"
    echo "Submitting: LoRA rank=${RANK} seed=${SEED} -> ${JOBNAME}"
    sbatch -J "$JOBNAME" slurm/run_cmkl.sh DistMult "$SEED" moe "--text-lora-rank $RANK"
  done
done

echo ""
echo "=== All 10 LoRA jobs submitted ==="
echo "Monitor: squeue -u \$USER --format='%.10i %.14j %.8T %.10M'"
