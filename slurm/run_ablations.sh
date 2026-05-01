#!/bin/bash --login
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=v100
#SBATCH --partition=batch
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=32G
#SBATCH -J mcgl_abl
#SBATCH -o slurm/slurm_logs/mcgl_abl_%J.out

# Run a SINGLE ablation study on IBEX
# Usage:
#   sbatch slurm/run_ablations.sh struct_only
#   sbatch slurm/run_ablations.sh buffer_size_sweep
#
# DO NOT use "all" — submit each ablation as a separate job via submit_all.sh

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate mcgl

mkdir -p results slurm/slurm_logs

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

ABLATION=${1:?Usage: sbatch run_ablations.sh <ablation_name>}

echo "Job ID: $SLURM_JOB_ID"
echo "Ablation: $ABLATION"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python scripts/run_ablations.py \
    --ablation $ABLATION \
    --embedding-dim 256 \
    --num-epochs 100 \
    --batch-size 512 \
    --device cuda \
    --seeds 42 123 456 789 1024 \
    --output-dir results

echo "End: $(date)"
