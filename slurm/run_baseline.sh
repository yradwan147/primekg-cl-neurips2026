#!/bin/bash --login
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=v100
#SBATCH --partition=batch
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=48G
#SBATCH -o slurm/slurm_logs/%x_%J.out

# Run a single baseline with a single seed on IBEX
# Usage:
#   sbatch -J ns_s42 slurm/run_baseline.sh naive_sequential TransE 42
#   sbatch -J ewc_s123 slurm/run_baseline.sh ewc TransE 123

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate mcgl

mkdir -p results slurm/slurm_logs

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

BASELINE=${1:?Usage: sbatch run_baseline.sh <baseline> <model> <seed>}
MODEL=${2:-TransE}
SEED=${3:?Must provide seed}

echo "Job ID: $SLURM_JOB_ID"
echo "Baseline: $BASELINE, Model: $MODEL, Seed: $SEED"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python scripts/run_baselines.py \
    --baseline $BASELINE \
    --model $MODEL \
    --embedding-dim 256 \
    --num-epochs 100 \
    --batch-size 512 \
    --device cuda \
    --seeds $SEED \
    --output-dir results \
    --output-suffix _seed${SEED}

echo "End: $(date)"
