#!/bin/bash
# Submit all experiments — 1 seed per job
# Usage: bash slurm/submit_all.sh
#
# Each job has a meaningful name: {method_abbrev}_s{seed}
# Monitor: watch -n 30 'grep -h "\[PROGRESS\]\|\[SUCCESS\]\|\[FAILED\]" slurm/slurm_logs/*.out | sort | tail -30'

set -e
mkdir -p slurm/slurm_logs

# Ensure LKGE is cloned and patched
if [ ! -d "external/LKGE" ]; then
    echo "Cloning LKGE..."
    git clone https://github.com/nju-websoft/LKGE.git external/LKGE
fi
echo "Applying LKGE patches..."
cp patches/LKGE_src/config.py external/LKGE/src/config.py
cp patches/LKGE_src/parse_args.py external/LKGE/src/parse_args.py

SEEDS="42 123 456 789 1024"
COUNT=0

echo "Submitting all experiments (1 seed per job)..."
echo "================================================"

for SEED in $SEEDS; do
    # KGE baselines (30h, 48G RAM)
    sbatch -J ns_s${SEED} slurm/run_baseline.sh naive_sequential TransE $SEED
    sbatch -J jt_s${SEED} slurm/run_baseline.sh joint_training TransE $SEED
    sbatch -J ewc_s${SEED} slurm/run_baseline.sh ewc TransE $SEED
    sbatch -J er_s${SEED} slurm/run_baseline.sh experience_replay TransE $SEED

    # CMKL (30h, 48G RAM)
    sbatch -J cmkl_s${SEED} slurm/run_cmkl.sh DistMult $SEED

    # LKGE (48h, 128G RAM, skip task_0_base)
    sbatch -J lkge_s${SEED} slurm/run_lkge.sh TransE $SEED

    # RAG (24h, 128G RAM, no GPU, retrieval-only)
    sbatch -J rag_s${SEED} slurm/run_rag.sh $SEED

    # Node Classification (30h, 32G RAM)
    sbatch -J nc_ns_s${SEED} slurm/run_nc.sh naive_sequential $SEED
    sbatch -J nc_ewc_s${SEED} slurm/run_nc.sh ewc $SEED
    sbatch -J nc_er_s${SEED} slurm/run_nc.sh experience_replay $SEED
    sbatch -J nc_jt_s${SEED} slurm/run_nc.sh joint_training $SEED
    sbatch -J nc_cmkl_s${SEED} slurm/run_nc.sh cmkl $SEED

    COUNT=$((COUNT + 12))
done

echo ""
echo "Submitted $COUNT jobs (5 seeds x 12 methods)"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  watch -n 30 'grep -h \"\\[PROGRESS\\]\\|\\[SUCCESS\\]\\|\\[FAILED\\]\" slurm/slurm_logs/*.out | sort | tail -30'"
echo ""
echo "After completion, merge results:"
echo "  python scripts/merge_seed_results.py --input-dir results"
