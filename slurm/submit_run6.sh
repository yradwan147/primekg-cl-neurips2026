#!/bin/bash
# Run 6: Re-run all Run 5 jobs that timed out due to R-GCN scalability bug
#
# The bug: _train_epoch() called self.forward() (full-graph R-GCN) per batch,
# giving O(N/batch_size) forward passes per epoch. With 5.6M triples / 512
# batch_size = 11K R-GCN calls/epoch, each ~1.4s → ~4300 hours per task.
#
# The fix: per-epoch R-GCN training (standard from Schlichtkrull et al., 2018).
# ONE forward pass per epoch + sampled batch (50K triples) + ONE backward.
# Estimated: ~4s/epoch × 100 epochs × 10 tasks = 67 min/seed.
#
# What succeeded in Run 5 (NOT re-run here):
#   - precompute (features already exist)
#   - abl_to (text_only) — 5 seeds, all succeeded (skips R-GCN)
#
# What timed out in Run 5 (re-run here): 40 jobs
#   - 5 CMKL-DistMult LP (5 seeds)
#   - 5 CMKL-TransE LP (5 seeds)
#   - 5 CMKL NC (5 seeds)
#   - 25 CMKL ablations (5 types x 5 seeds, excluding text_only)
#
# Prerequisites:
#   Features already pre-computed in Run 5 (check below).
#
# Usage: bash slurm/submit_run6.sh

set -e
mkdir -p slurm/slurm_logs results

SEEDS="42 123 456 789 1024"
COUNT=0

# Check that features exist (from Run 5 precompute)
if [ ! -f "data/benchmark/features/edge_index.pt" ]; then
    echo "ERROR: edge_index.pt not found!"
    echo "Features should already exist from Run 5 precompute."
    echo "If not, run: sbatch -J precompute slurm/run_precompute_features.sh"
    exit 1
fi

echo "============================================================"
echo "Run 6: R-GCN Scalability Fix (re-run timed-out Run 5 jobs)"
echo "============================================================"
echo ""
echo "Feature files (from Run 5):"
ls -lh data/benchmark/features/*.pt 2>/dev/null
echo ""

# ---------------------------------------------------------------
# Group 1: CMKL-DistMult LP (5 jobs)
# ---------------------------------------------------------------
echo "Group 1: CMKL-DistMult LP (5 jobs, 5 seeds)"
echo "---------------------------------------------"

for SEED in $SEEDS; do
    echo "  Submitting: CMKL-DistMult seed=$SEED"
    sbatch -J cmkl_dm_s${SEED} slurm/run_cmkl.sh DistMult $SEED
    COUNT=$((COUNT + 1))
done

# ---------------------------------------------------------------
# Group 2: CMKL-TransE LP (5 jobs)
# ---------------------------------------------------------------
echo ""
echo "Group 2: CMKL-TransE LP (5 jobs, 5 seeds)"
echo "-------------------------------------------"

for SEED in $SEEDS; do
    echo "  Submitting: CMKL-TransE seed=$SEED"
    sbatch -J cmkl_te_s${SEED} slurm/run_cmkl.sh TransE $SEED
    COUNT=$((COUNT + 1))
done

# ---------------------------------------------------------------
# Group 3: CMKL NC (5 jobs)
# ---------------------------------------------------------------
echo ""
echo "Group 3: CMKL Node Classification (5 jobs, 5 seeds)"
echo "----------------------------------------------------"

for SEED in $SEEDS; do
    echo "  Submitting: CMKL NC seed=$SEED"
    sbatch -J cmkl_nc_s${SEED} slurm/run_nc.sh cmkl $SEED
    COUNT=$((COUNT + 1))
done

# ---------------------------------------------------------------
# Group 4: CMKL Ablations (25 jobs — text_only already done)
# ---------------------------------------------------------------
echo ""
echo "Group 4: CMKL Ablations (25 jobs, 5 types x 5 seeds)"
echo "  NOTE: text_only SKIPPED (succeeded in Run 5)"
echo "-----------------------------------------------------"

ABLATIONS="struct_only concat_fusion global_ewc random_replay distillation"

for ABL in $ABLATIONS; do
    for SEED in $SEEDS; do
        case $ABL in
            struct_only)    PREFIX="so" ;;
            concat_fusion)  PREFIX="cf" ;;
            global_ewc)     PREFIX="ge" ;;
            random_replay)  PREFIX="rr" ;;
            distillation)   PREFIX="di" ;;
            *)              PREFIX="ab" ;;
        esac
        echo "  Submitting: $ABL seed=$SEED"
        sbatch -J abl_${PREFIX}_s${SEED} slurm/run_ablation_seed.sh $ABL $SEED
        COUNT=$((COUNT + 1))
    done
done

echo ""
echo "============================================================"
echo "Submitted $COUNT jobs total"
echo "  Group 1:  5 CMKL-DistMult LP (5 seeds)"
echo "  Group 2:  5 CMKL-TransE LP (5 seeds)"
echo "  Group 3:  5 CMKL NC (5 seeds)"
echo "  Group 4: 25 CMKL ablations (5 types x 5 seeds)"
echo "  SKIPPED:  5 text_only (already succeeded in Run 5)"
echo "============================================================"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  watch -n 30 'grep -h \"\\[PROGRESS\\]\\|\\[SUCCESS\\]\\|\\[FAILED\\]\" slurm/slurm_logs/*.out | sort | tail -40'"
echo ""
echo "Check for failures:"
echo "  grep -l \"\\[FAILED\\]\" slurm/slurm_logs/*_*.out"
