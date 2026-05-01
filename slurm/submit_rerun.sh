#!/bin/bash
# Submit ONLY the failed/missing jobs from Run 1
# Run 1 failures:
#   - ALL naive_seq/ewc/replay (segfault during eval) — fixed with max_test_triples=50K
#   - joint_training seeds 123, 456 (segfault) — same fix
#   - ALL RAG (24h time limit) — fixed with 48h + Qwen LLM
#
# Usage: bash slurm/submit_rerun.sh

set -e
mkdir -p slurm/slurm_logs

# Ensure LKGE is cloned and patched
if [ ! -d "external/LKGE" ]; then
    echo "Cloning LKGE..."
    git clone https://github.com/nju-websoft/LKGE.git external/LKGE
fi
cp patches/LKGE_src/config.py external/LKGE/src/config.py
cp patches/LKGE_src/parse_args.py external/LKGE/src/parse_args.py

SEEDS="42 123 456 789 1024"
COUNT=0

echo "Submitting rerun jobs (failed from Run 1)..."
echo "================================================"

for SEED in $SEEDS; do
    # KGE baselines — ALL seeds need rerun
    sbatch -J ns_s${SEED} slurm/run_baseline.sh naive_sequential TransE $SEED
    sbatch -J ewc_s${SEED} slurm/run_baseline.sh ewc TransE $SEED
    sbatch -J er_s${SEED} slurm/run_baseline.sh experience_replay TransE $SEED
    COUNT=$((COUNT + 3))
done

# Joint training — only seeds 123, 456 failed
for SEED in 123 456; do
    sbatch -J jt_s${SEED} slurm/run_baseline.sh joint_training TransE $SEED
    COUNT=$((COUNT + 1))
done

# RAG — ALL seeds need rerun (now with Qwen LLM)
for SEED in $SEEDS; do
    sbatch -J rag_s${SEED} slurm/run_rag.sh $SEED
    COUNT=$((COUNT + 1))
done

echo ""
echo "Submitted $COUNT rerun jobs"
echo "  - 15 KGE baselines (naive_seq + ewc + replay, all 5 seeds)"
echo "  - 2 joint_training (seeds 123, 456)"
echo "  - 5 RAG (all seeds, with Qwen2.5-7B-Instruct LLM)"
echo ""
echo "Monitor: watch -n 30 'grep -h \"\\[PROGRESS\\]\\|\\[SUCCESS\\]\\|\\[FAILED\\]\" slurm/slurm_logs/*.out | sort | tail -30'"
