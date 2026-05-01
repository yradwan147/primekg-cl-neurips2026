#!/bin/bash
# Run 9: MoE Fusion — replace gated cross-attention with Mixture-of-Experts
#
# Motivation: Run 8 showed text_only (AP=0.136) >> CMKL Full (AP=0.060).
# The gated fusion dilutes the strong text signal via interpolation + MLP bottleneck.
# MoE uses per-modality expert MLPs + learned router with masked softmax.
# Guarantee: MoE >= best single modality (router can learn weight=1 for any expert).
#
# No precompute needed — Run 8 features (Morgan FPs + BiomedBERT) are still valid.
#
# Usage: bash slurm/submit_run9.sh

set -e
mkdir -p slurm/slurm_logs results

SEEDS="42 123 456 789 1024"
COUNT=0

echo "============================================================"
echo "Run 9: MoE Fusion (15 jobs total)"
echo "============================================================"
echo ""
echo "No precompute needed — reusing Run 8 features."
echo ""

# ---------------------------------------------------------------
# Group 1: CMKL-MoE-DistMult LP (5 jobs)
# ---------------------------------------------------------------
echo "Group 1: CMKL-MoE-DistMult LP (5 jobs, 5 seeds)"
echo "-------------------------------------------------"

for SEED in $SEEDS; do
    echo "  Submitting: MoE-DistMult seed=$SEED"
    sbatch -J r9_dm_s${SEED} slurm/run_cmkl.sh DistMult $SEED moe
    COUNT=$((COUNT + 1))
done

# ---------------------------------------------------------------
# Group 2: CMKL-MoE-TransE LP (5 jobs)
# ---------------------------------------------------------------
echo ""
echo "Group 2: CMKL-MoE-TransE LP (5 jobs, 5 seeds)"
echo "-----------------------------------------------"

for SEED in $SEEDS; do
    echo "  Submitting: MoE-TransE seed=$SEED"
    sbatch -J r9_te_s${SEED} slurm/run_cmkl.sh TransE $SEED moe
    COUNT=$((COUNT + 1))
done

# ---------------------------------------------------------------
# Group 3: CMKL-MoE NC (5 jobs)
# ---------------------------------------------------------------
echo ""
echo "Group 3: CMKL-MoE Node Classification (5 jobs, 5 seeds)"
echo "--------------------------------------------------------"

for SEED in $SEEDS; do
    echo "  Submitting: MoE NC seed=$SEED"
    sbatch -J r9_nc_s${SEED} slurm/run_nc.sh cmkl $SEED moe
    COUNT=$((COUNT + 1))
done

echo ""
echo "============================================================"
echo "Submitted $COUNT jobs total"
echo "  Group 1:  5 MoE-DistMult LP (5 seeds)"
echo "  Group 2:  5 MoE-TransE LP (5 seeds)"
echo "  Group 3:  5 MoE NC (5 seeds)"
echo "============================================================"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  watch -n 30 'grep -h \"\\[PROGRESS\\]\\|\\[SUCCESS\\]\\|\\[FAILED\\]\" slurm/slurm_logs/r9_*.out | sort | tail -20'"
echo ""
echo "Check router weights in logs:"
echo "  grep \"Router:\" slurm/slurm_logs/r9_dm_s42_*.out | tail -10"
echo ""
echo "Check for failures:"
echo "  grep -l \"\\[FAILED\\]\" slurm/slurm_logs/r9_*.out 2>/dev/null"
