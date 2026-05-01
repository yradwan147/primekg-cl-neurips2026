#!/bin/bash
# Run 5: Fixed CMKL pipeline with actual multimodal features
#
# Previous CMKL runs were INVALID — model ran as flat embedding
# (no R-GCN message passing, no text, no mol features).
# This run uses pre-computed edge_index, text embeddings, and mol features.
#
# Prerequisites:
#   Features must be pre-computed first. Run:
#     sbatch -J precompute slurm/run_precompute_features.sh
#   Wait for it to complete, then run this script.
#
# Total: 45 jobs
#   - 5 CMKL-DistMult LP (5 seeds, 30h each)
#   - 5 CMKL-TransE LP (5 seeds, 30h each)
#   - 5 CMKL NC (5 seeds, 8h each)
#   - 30 CMKL ablations (6 types x 5 seeds, 30h each)
#
# Usage: bash slurm/submit_run5.sh

set -e
mkdir -p slurm/slurm_logs results

SEEDS="42 123 456 789 1024"
COUNT=0

# Check that features exist
if [ ! -f "data/benchmark/features/edge_index.pt" ]; then
    echo "ERROR: edge_index.pt not found!"
    echo "Run: sbatch -J precompute slurm/run_precompute_features.sh"
    echo "Wait for completion, then re-run this script."
    exit 1
fi

echo "============================================================"
echo "Run 5: Fixed CMKL Pipeline (multimodal features enabled)"
echo "============================================================"
echo ""
echo "Feature files found:"
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
# Group 4: CMKL Ablations (30 jobs)
# ---------------------------------------------------------------
echo ""
echo "Group 4: CMKL Ablations (30 jobs, 6 types x 5 seeds)"
echo "-----------------------------------------------------"

ABLATIONS="struct_only text_only concat_fusion global_ewc random_replay distillation"

for ABL in $ABLATIONS; do
    for SEED in $SEEDS; do
        case $ABL in
            struct_only)    PREFIX="so" ;;
            text_only)      PREFIX="to" ;;
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
echo "  Group 1: 5 CMKL-DistMult LP (5 seeds)"
echo "  Group 2: 5 CMKL-TransE LP (5 seeds)"
echo "  Group 3: 5 CMKL NC (5 seeds)"
echo "  Group 4: 30 CMKL ablations (6 types x 5 seeds)"
echo "============================================================"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  watch -n 30 'grep -h \"\\[PROGRESS\\]\\|\\[SUCCESS\\]\\|\\[FAILED\\]\" slurm/slurm_logs/*.out | sort | tail -40'"
echo ""
echo "Check for failures:"
echo "  grep -l \"\\[FAILED\\]\" slurm/slurm_logs/*_*.out"
