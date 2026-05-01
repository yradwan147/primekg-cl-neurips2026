#!/bin/bash
# Run 10: Score-Level Fusion + OGM-GE + Contrastive Alignment
#
# Ablative experiment design: separate runs to isolate each component's effect.
# Motivation: "Greedy modality" problem — struct dominates text/mol in
# embedding-level fusion. Score-level fusion decouples modality training.
#
# Reuses Run 8/9 features (BiomedBERT + Morgan FPs) — no precompute needed.
#
# Usage: bash slurm/submit_run10.sh

set -e
mkdir -p slurm/slurm_logs results

SEEDS="42 123 456 789 1024"
COUNT=0

echo "============================================================"
echo "Run 10: Score-Level Fusion (34 jobs total)"
echo "============================================================"
echo ""
echo "No precompute needed — reusing Run 8/9 features."
echo ""

# Args: DECODER SEED ALPHA_TEXT ALPHA_MOL USE_OGM CONTRASTIVE_W SUFFIX_TAG

# ---------------------------------------------------------------
# Group 1: SF-only DistMult (5 seeds) — Score fusion baseline
# ---------------------------------------------------------------
echo "Group 1: SF-only DistMult LP (5 seeds) — baseline"
echo "---------------------------------------------------"
for SEED in $SEEDS; do
    echo "  Submitting: SF DistMult seed=$SEED"
    sbatch -J r10_sf_dm_s${SEED} slurm/run_cmkl_sf.sh DistMult $SEED 0.5 0.3 0 0.0 base
    COUNT=$((COUNT + 1))
done

# ---------------------------------------------------------------
# Group 2: SF + OGM-GE DistMult (5 seeds) — gradient modulation
# ---------------------------------------------------------------
echo ""
echo "Group 2: SF + OGM-GE DistMult LP (5 seeds)"
echo "---------------------------------------------"
for SEED in $SEEDS; do
    echo "  Submitting: SF+OGM DistMult seed=$SEED"
    sbatch -J r10_sfogm_dm_s${SEED} slurm/run_cmkl_sf.sh DistMult $SEED 0.5 0.3 1 0.0 ogm
    COUNT=$((COUNT + 1))
done

# ---------------------------------------------------------------
# Group 3: SF + Contrastive DistMult (5 seeds) — alignment
# ---------------------------------------------------------------
echo ""
echo "Group 3: SF + Contrastive DistMult LP (5 seeds)"
echo "-------------------------------------------------"
for SEED in $SEEDS; do
    echo "  Submitting: SF+CL DistMult seed=$SEED"
    sbatch -J r10_sfcl_dm_s${SEED} slurm/run_cmkl_sf.sh DistMult $SEED 0.5 0.3 0 0.1 cl
    COUNT=$((COUNT + 1))
done

# ---------------------------------------------------------------
# Group 4: SF + OGM + Contrastive DistMult (5 seeds) — full system
# ---------------------------------------------------------------
echo ""
echo "Group 4: SF + OGM + Contrastive DistMult LP (5 seeds) — full"
echo "--------------------------------------------------------------"
for SEED in $SEEDS; do
    echo "  Submitting: SF+OGM+CL DistMult seed=$SEED"
    sbatch -J r10_sffull_dm_s${SEED} slurm/run_cmkl_sf.sh DistMult $SEED 0.5 0.3 1 0.1 full
    COUNT=$((COUNT + 1))
done

# ---------------------------------------------------------------
# Group 5: SF-only TransE (5 seeds) — score fusion on TransE
# ---------------------------------------------------------------
echo ""
echo "Group 5: SF-only TransE LP (5 seeds)"
echo "--------------------------------------"
for SEED in $SEEDS; do
    echo "  Submitting: SF TransE seed=$SEED"
    sbatch -J r10_sf_te_s${SEED} slurm/run_cmkl_sf.sh TransE $SEED 0.5 0.3 0 0.0 base
    COUNT=$((COUNT + 1))
done

# ---------------------------------------------------------------
# Group 6: SF + OGM + Contrastive TransE (5 seeds) — full on TransE
# ---------------------------------------------------------------
echo ""
echo "Group 6: SF + OGM + Contrastive TransE LP (5 seeds)"
echo "-----------------------------------------------------"
for SEED in $SEEDS; do
    echo "  Submitting: SF+OGM+CL TransE seed=$SEED"
    sbatch -J r10_sffull_te_s${SEED} slurm/run_cmkl_sf.sh TransE $SEED 0.5 0.3 1 0.1 full
    COUNT=$((COUNT + 1))
done

# ---------------------------------------------------------------
# Group 7: Alpha sweep (DistMult, seed=42) — skip alpha=0.5 (same as Group 1)
# ---------------------------------------------------------------
echo ""
echo "Group 7: Alpha text sweep (DistMult, seed=42, 3 values + Group 1 covers 0.5)"
echo "------------------------------------------------------------------------------"
for ALPHA in 0.3 1.0 2.0; do
    echo "  Submitting: SF DistMult seed=42 alpha_text=$ALPHA"
    sbatch -J r10_a${ALPHA}_s42 slurm/run_cmkl_sf.sh DistMult 42 $ALPHA 0.3 0 0.0 a${ALPHA}
    COUNT=$((COUNT + 1))
done

# ---------------------------------------------------------------
# Group 8: MoE-DistMult re-run with EWC bug fix (5 seeds) — fair comparison
# The EWC lambda compounding bug affected ALL prior runs (8-9).
# Re-run MoE to get a fair baseline with corrected EWC.
# ---------------------------------------------------------------
echo ""
echo "Group 8: MoE-DistMult LP re-run with EWC fix (5 seeds)"
echo "--------------------------------------------------------"
for SEED in $SEEDS; do
    echo "  Submitting: MoE-DistMult (EWC fixed) seed=$SEED"
    sbatch -J r10_moe_dm_s${SEED} slurm/run_cmkl.sh DistMult $SEED moe
    COUNT=$((COUNT + 1))
done

# ---------------------------------------------------------------
# Group 9: MoE NC re-run with EWC fix (5 seeds)
# ---------------------------------------------------------------
echo ""
echo "Group 9: MoE NC re-run with EWC fix (5 seeds)"
echo "------------------------------------------------"
for SEED in $SEEDS; do
    echo "  Submitting: MoE NC (EWC fixed) seed=$SEED"
    sbatch -J r10_moe_nc_s${SEED} slurm/run_nc.sh cmkl $SEED moe
    COUNT=$((COUNT + 1))
done

echo ""
echo "============================================================"
echo "Submitted $COUNT jobs total"
echo "  Group 1:  5  SF-only DistMult (baseline)"
echo "  Group 2:  5  SF + OGM-GE DistMult"
echo "  Group 3:  5  SF + Contrastive DistMult"
echo "  Group 4:  5  SF + OGM + Contrastive DistMult (full)"
echo "  Group 5:  5  SF-only TransE"
echo "  Group 6:  5  SF + OGM + Contrastive TransE (full)"
echo "  Group 7:  3  Alpha sweep (DistMult, seed=42; 0.5 covered by Group 1)"
echo "  Group 8:  5  MoE-DistMult re-run (EWC bug fix)"
echo "  Group 9:  5  MoE NC re-run (EWC bug fix)"
echo "============================================================"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  watch -n 30 'grep -h \"\\[PROGRESS\\]\\|\\[SUCCESS\\]\\|\\[FAILED\\]\" slurm/slurm_logs/r10_*.out | sort | tail -30'"
echo ""
echo "Check per-modality losses:"
echo "  grep \"losses:\" slurm/slurm_logs/r10_sf_dm_s42_*.out | tail -10"
echo ""
echo "Check OGM weights:"
echo "  grep \"OGM:\" slurm/slurm_logs/r10_sfogm_dm_s42_*.out | tail -10"
echo ""
echo "Check for failures:"
echo "  grep -l \"\\[FAILED\\]\" slurm/slurm_logs/r10_*.out 2>/dev/null"
