#!/bin/bash
# Run 7: Re-run ALL CMKL experiments with multimodal feature fixes
#
# What changed (all bugs make Run 6 results invalid):
#   1. Gated cross-modal fusion (old cross-attention was a no-op with seq_len=1)
#   2. R-GCN skip connections (c-RGCN style residual)
#   3. Text embeddings for drugs + diseases + proteins (~52K nodes, was 0)
#   4. 1024-bit Morgan fingerprints via RDKit (was 3 scalars)
#   5. Fusion residual bypass removed (was h_fused + h_struct)
#
# Step 1: Re-run precompute features (generates text_embeddings.pt + 1024-dim mol)
# Step 2: Re-run ALL CMKL + ablation jobs (46 total)
#
# Run 6 text_only was from Run 5 and also needs re-run (fusion changed).
# Run 6 distillation failed with OOM (fixed) — re-run included.
#
# Prerequisites:
#   pip install rdkit PyTDC  (in conda env on IBEX)
#
# Usage: bash slurm/submit_run7.sh

set -e
mkdir -p slurm/slurm_logs results

SEEDS="42 123 456 789 1024"
COUNT=0

echo "============================================================"
echo "Run 7: Multimodal Feature Fixes (ALL CMKL re-run)"
echo "============================================================"
echo ""

# ---------------------------------------------------------------
# Step 0: Re-compute features (text + Morgan FPs)
# ---------------------------------------------------------------
echo "Step 0: Re-computing multimodal features"
echo "-----------------------------------------"
echo "  This generates:"
echo "    - text_embeddings.pt (~52K nodes: drugs + diseases + proteins)"
echo "    - mol_features.pt (1024-bit Morgan fingerprints via RDKit)"
echo "    - node_has_text.pt, node_has_mol.pt, mol_dim.txt"
echo "  Estimated time: ~30 min (BiomedBERT encoding + TDC download)"
echo ""

PRECOMPUTE_JOB=$(sbatch --parsable -J precompute slurm/run_precompute_features.sh)
echo "  Submitted precompute job: $PRECOMPUTE_JOB"
COUNT=$((COUNT + 1))

echo ""
echo "All training jobs depend on precompute completing first."
echo ""

# ---------------------------------------------------------------
# Group 1: CMKL-DistMult LP (5 jobs)
# ---------------------------------------------------------------
echo "Group 1: CMKL-DistMult LP (5 jobs, 5 seeds)"
echo "---------------------------------------------"

for SEED in $SEEDS; do
    echo "  Submitting: CMKL-DistMult seed=$SEED (depends on $PRECOMPUTE_JOB)"
    sbatch --dependency=afterok:$PRECOMPUTE_JOB -J r7_dm_s${SEED} slurm/run_cmkl.sh DistMult $SEED
    COUNT=$((COUNT + 1))
done

# ---------------------------------------------------------------
# Group 2: CMKL-TransE LP (5 jobs)
# ---------------------------------------------------------------
echo ""
echo "Group 2: CMKL-TransE LP (5 jobs, 5 seeds)"
echo "-------------------------------------------"

for SEED in $SEEDS; do
    echo "  Submitting: CMKL-TransE seed=$SEED (depends on $PRECOMPUTE_JOB)"
    sbatch --dependency=afterok:$PRECOMPUTE_JOB -J r7_te_s${SEED} slurm/run_cmkl.sh TransE $SEED
    COUNT=$((COUNT + 1))
done

# ---------------------------------------------------------------
# Group 3: CMKL NC (5 jobs)
# ---------------------------------------------------------------
echo ""
echo "Group 3: CMKL Node Classification (5 jobs, 5 seeds)"
echo "----------------------------------------------------"

for SEED in $SEEDS; do
    echo "  Submitting: CMKL NC seed=$SEED (depends on $PRECOMPUTE_JOB)"
    sbatch --dependency=afterok:$PRECOMPUTE_JOB -J r7_nc_s${SEED} slurm/run_nc.sh cmkl $SEED
    COUNT=$((COUNT + 1))
done

# ---------------------------------------------------------------
# Group 4: ALL ablations (30 jobs — including text_only this time)
# ---------------------------------------------------------------
echo ""
echo "Group 4: ALL CMKL Ablations (30 jobs, 6 types x 5 seeds)"
echo "  NOTE: text_only RE-RUN (fusion changed from Run 5)"
echo "----------------------------------------------------------"

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
        echo "  Submitting: $ABL seed=$SEED (depends on $PRECOMPUTE_JOB)"
        sbatch --dependency=afterok:$PRECOMPUTE_JOB -J r7_${PREFIX}_s${SEED} slurm/run_ablation_seed.sh $ABL $SEED
        COUNT=$((COUNT + 1))
    done
done

echo ""
echo "============================================================"
echo "Submitted $COUNT jobs total"
echo "  Step 0:  1 precompute (features re-generation)"
echo "  Group 1:  5 CMKL-DistMult LP (5 seeds)"
echo "  Group 2:  5 CMKL-TransE LP (5 seeds)"
echo "  Group 3:  5 CMKL NC (5 seeds)"
echo "  Group 4: 30 CMKL ablations (6 types x 5 seeds)"
echo ""
echo "  All training jobs depend on precompute ($PRECOMPUTE_JOB)"
echo "============================================================"
echo ""
echo "Before submitting, make sure:"
echo "  1. git pull  (to get the code fixes)"
echo "  2. pip install rdkit-pypi PyTDC  (or: conda install -c conda-forge rdkit && pip install PyTDC)"
echo "  3. Pre-download TDC data on login node (compute nodes may lack internet):"
echo "     python -c \"from tdc.resource import PrimeKG; PrimeKG(path='data/tdc_primekg').get_features('drug')\""
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  watch -n 30 'grep -h \"\\[PROGRESS\\]\\|\\[SUCCESS\\]\\|\\[FAILED\\]\" slurm/slurm_logs/r7_*.out | sort | tail -40'"
echo ""
echo "Check precompute first:"
echo "  cat slurm/slurm_logs/precompute_${PRECOMPUTE_JOB}.out"
echo "  ls -lh data/benchmark/features/*.pt"
echo ""
echo "Check for failures:"
echo "  grep -l \"\\[FAILED\\]\" slurm/slurm_logs/r7_*.out 2>/dev/null"
