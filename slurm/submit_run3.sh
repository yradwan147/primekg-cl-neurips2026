#!/bin/bash
# Run 3: Rerun ALL KGE baseline LP jobs with custom evaluation
# (bypasses PyKEEN's RankBasedEvaluator to fix segfault)
#
# evaluate_link_prediction() was rewritten — ALL KGE baselines
# must rerun for consistency, including Joint Training's 4
# previously-successful seeds.
#
# Total: 20 jobs (4 methods x 5 seeds)
# Usage: bash slurm/submit_run3.sh

set -e
mkdir -p slurm/slurm_logs

SEEDS="42 123 456 789 1024"
COUNT=0

echo "Submitting Run 3 jobs (custom eval, no PyKEEN evaluator)..."
echo "============================================================"

for SEED in $SEEDS; do
    sbatch -J ns_s${SEED} slurm/run_baseline.sh naive_sequential TransE $SEED
    sbatch -J ewc_s${SEED} slurm/run_baseline.sh ewc TransE $SEED
    sbatch -J er_s${SEED} slurm/run_baseline.sh experience_replay TransE $SEED
    sbatch -J jt_s${SEED} slurm/run_baseline.sh joint_training TransE $SEED
    COUNT=$((COUNT + 4))
done

echo ""
echo "Submitted $COUNT jobs"
echo "  - 5 naive_sequential (all seeds)"
echo "  - 5 ewc (all seeds)"
echo "  - 5 experience_replay (all seeds)"
echo "  - 5 joint_training (all seeds — rerun for eval consistency)"
echo ""
echo "Monitor: watch -n 30 'grep -h \"\\[PROGRESS\\]\\|\\[SUCCESS\\]\\|\\[FAILED\\]\" slurm/slurm_logs/*.out | sort | tail -30'"
