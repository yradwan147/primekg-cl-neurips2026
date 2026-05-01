#!/bin/bash
# Run 8: Re-run ALL CMKL experiments with Bug A + Bug B fixes
#
# Bug A fix: SMILES from PubChem cache (data/smiles_cache.json) instead of TDC
#   - TDC had no node_name/smiles columns -> 0 Morgan FPs in Run 7
#   - PubChem cache provides ~5-7K drug SMILES -> proper 1024-bit Morgan FPs
#
# Bug B fix: Conditional residual in fusion.py
#   - Nodes WITHOUT text/mol now get h_struct directly (bypass MLP bottleneck)
#   - Nodes WITH text/mol still go through gated fusion MLP
#   - This restores structural signal for the ~64% of nodes without modalities
#
# Prerequisites:
#   1. Run locally: python scripts/fetch_smiles.py
#   2. Verify: data/smiles_cache.json exists with >5K entries
#   3. git add data/smiles_cache.json && git commit && git push
#   4. On IBEX: git pull
#   5. On IBEX: pip install rdkit-pypi (if not already)
#
# Usage: bash slurm/submit_run8.sh

set -e
mkdir -p slurm/slurm_logs results

SEEDS="42 123 456 789 1024"
COUNT=0

echo "============================================================"
echo "Run 8: Bug A (SMILES) + Bug B (Fusion Residual) Fixes"
echo "============================================================"
echo ""

# ---------------------------------------------------------------
# Verify SMILES cache exists
# ---------------------------------------------------------------
if [ ! -f data/smiles_cache.json ]; then
    echo "ERROR: data/smiles_cache.json not found!"
    echo "Run 'python scripts/fetch_smiles.py' first (needs internet)."
    exit 1
fi
CACHE_SIZE=$(python3 -c "import json; print(len(json.load(open('data/smiles_cache.json'))))" 2>/dev/null || echo "0")
echo "SMILES cache: $CACHE_SIZE entries"
if [ "$CACHE_SIZE" -lt 1000 ]; then
    echo "WARNING: cache has <1000 entries. Expected >5000."
    echo "Run 'python scripts/fetch_smiles.py' to fetch more SMILES."
fi
echo ""

# ---------------------------------------------------------------
# Step 0: Re-compute features (text + Morgan FPs from cache)
# ---------------------------------------------------------------
echo "Step 0: Re-computing multimodal features"
echo "-----------------------------------------"
echo "  Morgan FPs now from PubChem SMILES cache (not TDC)"
echo "  Expected: >5000 drugs with 1024-bit Morgan FPs"
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
    sbatch --dependency=afterok:$PRECOMPUTE_JOB -J r8_dm_s${SEED} slurm/run_cmkl.sh DistMult $SEED
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
    sbatch --dependency=afterok:$PRECOMPUTE_JOB -J r8_te_s${SEED} slurm/run_cmkl.sh TransE $SEED
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
    sbatch --dependency=afterok:$PRECOMPUTE_JOB -J r8_nc_s${SEED} slurm/run_nc.sh cmkl $SEED
    COUNT=$((COUNT + 1))
done

# ---------------------------------------------------------------
# Group 4: ALL ablations (30 jobs — 6 types x 5 seeds)
# ---------------------------------------------------------------
echo ""
echo "Group 4: ALL CMKL Ablations (30 jobs, 6 types x 5 seeds)"
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
        sbatch --dependency=afterok:$PRECOMPUTE_JOB -J r8_${PREFIX}_s${SEED} slurm/run_ablation_seed.sh $ABL $SEED
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
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  watch -n 30 'grep -h \"\\[PROGRESS\\]\\|\\[SUCCESS\\]\\|\\[FAILED\\]\" slurm/slurm_logs/r8_*.out | sort | tail -40'"
echo ""
echo "Check precompute first:"
echo "  cat slurm/slurm_logs/precompute_${PRECOMPUTE_JOB}.out"
echo "  ls -lh data/benchmark/features/*.pt"
echo "  # Verify: 'SMILES from cache: N drugs' with N > 5000"
echo ""
echo "Check for failures:"
echo "  grep -l \"\\[FAILED\\]\" slurm/slurm_logs/r8_*.out 2>/dev/null"
