#!/bin/bash --login
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=v100
#SBATCH --partition=batch
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=48G
#SBATCH -o slurm/slurm_logs/%x_%J.out

# Run node classification with a single method and seed on IBEX
# Usage:
#   sbatch -J nc_ns_s42 slurm/run_nc.sh naive_sequential 42
#   sbatch -J nc_ewc_s123 slurm/run_nc.sh ewc 123
#   sbatch -J nc_moe_s42 slurm/run_nc.sh cmkl 42 moe

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate mcgl

mkdir -p results slurm/slurm_logs

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

METHOD=${1:?Usage: sbatch run_nc.sh <method> <seed> [fusion]}
SEED=${2:?Must provide seed}
FUSION=${3:-cross_attention}

echo "Job ID: $SLURM_JOB_ID"
echo "NC Method: $METHOD, Seed: $SEED, Fusion: $FUSION"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python scripts/run_nc.py \
    --method $METHOD \
    --fusion $FUSION \
    --embedding-dim 256 \
    --num-epochs 100 \
    --batch-size 512 \
    --device cuda \
    --seeds $SEED \
    --output-dir results \
    --output-suffix _seed${SEED}

echo "End: $(date)"
