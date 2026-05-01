#!/bin/bash
# Run 4: Multi-hop evaluation + CMKL ablations + decoder-controlled baseline
#
# Purpose:
#   1. Multi-hop evaluation for all KGE baselines (paper gap: graph structure advantage)
#   2. CMKL with TransE decoder (addresses DistMult vs TransE confound)
#   3. Core CMKL ablation studies (paper B: component contribution analysis)
#
# Total: 39 jobs
#   - 4 KGE baselines with multihop (1 seed each, seed 42)
#   - 5 CMKL-TransE decoder ablation (5 seeds)
#   - 30 CMKL ablations (6 types x 5 seeds)
#
# Usage: bash slurm/submit_run4.sh

set -e
mkdir -p slurm/slurm_logs results

SEEDS="42 123 456 789 1024"
COUNT=0

echo "============================================================"
echo "Run 4: Multi-hop + Ablations"
echo "============================================================"

# ---------------------------------------------------------------
# Group 1: KGE Baselines with Multi-hop Evaluation (4 jobs)
# ---------------------------------------------------------------
# Retrains model + runs multi-hop scoring with fixed make_pykeen_score_fn
# (bypasses PyKEEN score_hrt which caused crashes)
# Only seed 42 — multihop measures structural understanding, not variance.

echo ""
echo "Group 1: KGE Baselines + Multi-hop (4 jobs, seed=42)"
echo "-----------------------------------------------------"

for BASELINE in naive_sequential joint_training ewc experience_replay; do
    echo "  Submitting: $BASELINE multihop seed=42"
    sbatch -J mh_${BASELINE:0:2}_s42 slurm/run_baseline_multihop.sh $BASELINE TransE 42
    COUNT=$((COUNT + 1))
done

# ---------------------------------------------------------------
# Group 2: CMKL with TransE decoder (5 jobs)
# ---------------------------------------------------------------
# Decoder-controlled ablation: CMKL uses DistMult in main results.
# Running with TransE isolates the multimodal architecture advantage
# from the decoder choice advantage.

echo ""
echo "Group 2: CMKL-TransE decoder ablation (5 jobs, 5 seeds)"
echo "--------------------------------------------------------"

for SEED in $SEEDS; do
    echo "  Submitting: CMKL-TransE seed=$SEED"
    sbatch -J cmkl_te_s${SEED} slurm/run_cmkl.sh TransE $SEED
    COUNT=$((COUNT + 1))
done

# ---------------------------------------------------------------
# Group 3: Core CMKL Ablations (30 jobs)
# ---------------------------------------------------------------
# 6 ablation types x 5 seeds = 30 jobs
# These answer: which CMKL components matter?
#   - struct_only: value of multimodal features
#   - text_only: structural vs textual contribution
#   - concat_fusion: cross-attention vs concatenation
#   - global_ewc: modality-aware vs global EWC
#   - random_replay: K-means vs random buffer selection
#   - distillation: knowledge distillation contribution

echo ""
echo "Group 3: CMKL Ablations (30 jobs, 6 types x 5 seeds)"
echo "-----------------------------------------------------"

ABLATIONS="struct_only text_only concat_fusion global_ewc random_replay distillation"

for ABL in $ABLATIONS; do
    for SEED in $SEEDS; do
        # Short job name prefix for each ablation
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
echo "  Group 1: 4 KGE baselines + multihop (seed 42)"
echo "  Group 2: 5 CMKL-TransE (5 seeds)"
echo "  Group 3: 30 CMKL ablations (6 types x 5 seeds)"
echo "============================================================"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  watch -n 30 'grep -h \"\\[PROGRESS\\]\\|\\[SUCCESS\\]\\|\\[FAILED\\]\" slurm/slurm_logs/*.out | sort | tail -40'"
echo ""
echo "Check for failures:"
echo "  grep -l \"\\[FAILED\\]\" slurm/slurm_logs/*_*.out"
