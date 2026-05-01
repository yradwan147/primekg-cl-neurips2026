#!/bin/bash
# Run 14: Missing experiments for Paper A revision
# Group 1: NC for SI/Distillation/MIR (15 jobs)
# Group 2: KGQA zero-shot (no retrieval) baseline (5 jobs)
# Group 3: KGQA retrieval-only (no LLM) baseline (5 jobs)
# Total: 25 jobs

mkdir -p slurm/slurm_logs results

SEEDS=(42 123 456 789 1024)

echo "=== Run 14: Missing Experiments ==="

# ===== Group 1: NC for new baselines (15 jobs) =====
echo "--- Group 1: NC for SI, Distillation, MIR ---"
for METHOD in si distillation mir_replay; do
    for SEED in "${SEEDS[@]}"; do
        sbatch -J nc_${METHOD}_s${SEED} slurm/run_nc.sh $METHOD $SEED
    done
done

# ===== Group 2: KGQA zero-shot (LLM without retrieval) (5 jobs) =====
echo "--- Group 2: KGQA zero-shot (Qwen2.5-7B, no retrieval) ---"
for SEED in "${SEEDS[@]}"; do
    sbatch -J kgqa_zeroshot_s${SEED} slurm/run_rag.sh $SEED --no-retrieval --output-suffix _zeroshot_seed${SEED}
done

# ===== Group 3: KGQA retrieval-only (no LLM) (5 jobs) =====
echo "--- Group 3: KGQA retrieval-only (no LLM) ---"
for SEED in "${SEEDS[@]}"; do
    sbatch -J kgqa_retrieval_s${SEED} slurm/run_rag.sh $SEED --no-llm --output-suffix _retrieval_only_seed${SEED}
done

echo ""
echo "Done! Submitted 25 jobs."
echo "Monitor: squeue -u \$USER"
echo ""
echo "Groups:"
echo "  1. NC for SI/Distillation/MIR (15 jobs)"
echo "  2. KGQA zero-shot - LLM only, no retrieval (5 jobs)"
echo "  3. KGQA retrieval-only - no LLM (5 jobs)"
