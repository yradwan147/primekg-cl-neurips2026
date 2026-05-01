#!/bin/bash
# Run 13: Paper B ablations — MA-EWC vs Uniform EWC, lambda sensitivity, buffer size
# Total: 45 jobs (9 configs × 5 seeds)
# Script: slurm/run_cmkl_ablation.sh SEED OUTPUT_DIR SUFFIX [extra args...]

mkdir -p slurm/slurm_logs results_run13

SEEDS=(42 123 456 789 1024)

echo "=== Run 13: Paper B Ablations ==="

# ===== Group 1: Uniform EWC (all λ=10) — 5 jobs =====
# Compare against MA-EWC (λ_s=10, λ_t=5, λ_m=1, λ_f=5) already in results_run12
echo "--- Group 1: Uniform EWC ---"
for SEED in "${SEEDS[@]}"; do
    sbatch -J uniform_ewc_s${SEED} slurm/run_cmkl_ablation.sh $SEED results_run13 _uniform_ewc_seed${SEED} \
        --lambda-struct 10.0 --lambda-text 10.0 --lambda-mol 10.0 --lambda-fusion 10.0
done

# ===== Group 2: Lambda Sensitivity — 20 jobs =====
# Scale all MA-EWC lambdas by factor, keeping ratios constant
echo "--- Group 2: Lambda Sensitivity ---"
for FACTOR_LABEL in "0.1x:1.0:0.5:0.1:0.5" "0.5x:5.0:2.5:0.5:2.5" "2x:20.0:10.0:2.0:10.0" "5x:50.0:25.0:5.0:25.0"; do
    IFS=: read -r LABEL LS LT LM LF <<< "$FACTOR_LABEL"
    for SEED in "${SEEDS[@]}"; do
        sbatch -J lambda_${LABEL}_s${SEED} slurm/run_cmkl_ablation.sh $SEED results_run13 _lambda${LABEL}_seed${SEED} \
            --lambda-struct $LS --lambda-text $LT --lambda-mol $LM --lambda-fusion $LF
    done
done

# ===== Group 3: Buffer Size — 15 jobs =====
echo "--- Group 3: Buffer Size ---"
for BSIZE in 500 2000 5000; do
    for SEED in "${SEEDS[@]}"; do
        sbatch -J buffer_${BSIZE}_s${SEED} slurm/run_cmkl_ablation.sh $SEED results_run13 _buffer${BSIZE}_seed${SEED} \
            --replay-buffer-size $BSIZE
    done
done

# ===== Group 4: No EWC (λ=0) — 5 jobs =====
echo "--- Group 4: No EWC (λ=0) ---"
for SEED in "${SEEDS[@]}"; do
    sbatch -J no_ewc_s${SEED} slurm/run_cmkl_ablation.sh $SEED results_run13 _no_ewc_seed${SEED} \
        --lambda-struct 0.0 --lambda-text 0.0 --lambda-mol 0.0 --lambda-fusion 0.0
done

echo ""
echo "Done! Submitted 45 jobs."
echo "Monitor: squeue -u \$USER"
echo ""
echo "Groups:"
echo "  1. Uniform EWC (5 jobs) — compare MA-EWC vs uniform λ=10"
echo "  2. Lambda sensitivity (20 jobs) — 0.1x, 0.5x, 2x, 5x scaling"
echo "  3. Buffer size (15 jobs) — 500, 2000, 5000"
echo "  4. No EWC (5 jobs) — λ=0 baseline"
