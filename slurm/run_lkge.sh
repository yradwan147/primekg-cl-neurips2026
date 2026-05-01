#!/bin/bash --login
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=v100
#SBATCH --partition=batch
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=128G
#SBATCH -o slurm/slurm_logs/%x_%J.out

# Run LKGE with a single seed on IBEX
# Usage:
#   sbatch -J lkge_s42 slurm/run_lkge.sh TransE 42

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate mcgl

mkdir -p results slurm/slurm_logs

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True

MODEL=${1:-TransE}
SEED=${2:?Must provide seed}

echo "Job ID: $SLURM_JOB_ID"
echo "LKGE Model: $MODEL, Seed: $SEED"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

# Clone LKGE if not already present
if [ ! -d "external/LKGE" ]; then
    echo "Cloning LKGE repository..."
    mkdir -p external
    git clone https://github.com/nju-websoft/LKGE.git external/LKGE
    pip install prettytable quadprog
fi

# Remove stale LKGE format data from previous runs
rm -rf results/lkge_format

# emb_dim=50 to fit LKGE's full-graph GCN in V100 32GB
# config.py no longer overrides CLI args (patched)
# --skip-base-task: task_0_base (5.67M edges) is too large for LKGE's GCN
python scripts/run_lkge.py \
    --model $MODEL \
    --num-epochs 100 \
    --emb-dim 50 \
    --skip-base-task \
    --seeds $SEED \
    --output-dir results \
    --output-suffix _seed${SEED}

echo "End: $(date)"
