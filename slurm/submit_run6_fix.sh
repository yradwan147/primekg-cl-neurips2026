#!/bin/bash
# Run 6 Fix: Re-run 20 jobs that failed in Run 6
#
# Three bugs fixed:
#   1. Device mismatch (CPU features vs CUDA model) in multihop eval and NC
#      -> run_cmkl.py and run_nc.py now move features to model device
#   2. Distillation OOM (50K × 129K all-entity scoring matrix = 24 GiB)
#      -> cmkl.py now subsamples to batch_size (512) triples for distillation
#
# Run 6 LP training results ARE SAVED (cmkl_DistMult/TransE_seed*.json).
# But model weights are NOT checkpointed, so multihop eval requires retraining.
#
# Jobs to re-run (20 total):
#   - 10 CMKL LP (DistMult + TransE, 5 seeds each) — retrain + multihop
#       LP results will overwrite existing (same training, just adds multihop)
#   - 5 CMKL NC (5 seeds) — full re-run with device fix
#   - 5 Distillation ablation (5 seeds) — full re-run with OOM fix
#
# Usage: bash slurm/submit_run6_fix.sh

set -e
mkdir -p slurm/slurm_logs results

SEEDS="42 123 456 789 1024"
COUNT=0

# Check that features exist
if [ ! -f "data/benchmark/features/edge_index.pt" ]; then
    echo "ERROR: edge_index.pt not found!"
    exit 1
fi

echo "============================================================"
echo "Run 6 Fix: Re-run 20 failed jobs (device + OOM bugs fixed)"
echo "============================================================"
echo ""

# ---------------------------------------------------------------
# Group 1: CMKL-DistMult LP + multihop (5 jobs)
#   LP results exist but multihop crashed; must retrain to get model
# ---------------------------------------------------------------
echo "Group 1: CMKL-DistMult LP + multihop (5 jobs)"
echo "-----------------------------------------------"

for SEED in $SEEDS; do
    echo "  Submitting: CMKL-DistMult seed=$SEED (retrain + multihop)"
    sbatch -J cmkl_dm_s${SEED} slurm/run_cmkl.sh DistMult $SEED
    COUNT=$((COUNT + 1))
done

# ---------------------------------------------------------------
# Group 2: CMKL-TransE LP + multihop (5 jobs)
# ---------------------------------------------------------------
echo ""
echo "Group 2: CMKL-TransE LP + multihop (5 jobs)"
echo "---------------------------------------------"

for SEED in $SEEDS; do
    echo "  Submitting: CMKL-TransE seed=$SEED (retrain + multihop)"
    sbatch -J cmkl_te_s${SEED} slurm/run_cmkl.sh TransE $SEED
    COUNT=$((COUNT + 1))
done

# ---------------------------------------------------------------
# Group 3: CMKL NC (5 jobs)
# ---------------------------------------------------------------
echo ""
echo "Group 3: CMKL Node Classification (5 jobs)"
echo "--------------------------------------------"

for SEED in $SEEDS; do
    echo "  Submitting: CMKL NC seed=$SEED"
    sbatch -J cmkl_nc_s${SEED} slurm/run_nc.sh cmkl $SEED
    COUNT=$((COUNT + 1))
done

# ---------------------------------------------------------------
# Group 4: Distillation ablation (5 jobs)
# ---------------------------------------------------------------
echo ""
echo "Group 4: Distillation ablation (5 jobs, OOM fix)"
echo "-------------------------------------------------"

for SEED in $SEEDS; do
    echo "  Submitting: distillation seed=$SEED"
    sbatch -J abl_di_s${SEED} slurm/run_ablation_seed.sh distillation $SEED
    COUNT=$((COUNT + 1))
done

echo ""
echo "============================================================"
echo "Submitted $COUNT jobs total"
echo "  Group 1:  5 CMKL-DistMult LP + multihop"
echo "  Group 2:  5 CMKL-TransE LP + multihop"
echo "  Group 3:  5 CMKL NC"
echo "  Group 4:  5 Distillation ablation (OOM fix)"
echo "============================================================"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  watch -n 30 'grep -h \"\\[PROGRESS\\]\\|\\[SUCCESS\\]\\|\\[FAILED\\]\" slurm/slurm_logs/*.out | sort | tail -20'"
