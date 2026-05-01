#!/bin/bash --login
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=v100
#SBATCH --partition=batch
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=48G
#SBATCH -o slurm/slurm_logs/%x_%J.out

# Run a SINGLE ablation with a SINGLE seed on IBEX
# Usage:
#   sbatch -J abl_so_s42 slurm/run_ablation_seed.sh struct_only 42

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate mcgl

mkdir -p results slurm/slurm_logs

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

ABLATION=${1:?Usage: sbatch run_ablation_seed.sh <ablation_name> <seed>}
SEED=${2:?Must provide seed}

echo "[STARTED] Ablation: $ABLATION, Seed: $SEED"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python scripts/run_ablations.py \
    --ablation $ABLATION \
    --embedding-dim 256 \
    --num-epochs 100 \
    --batch-size 512 \
    --device cuda \
    --seeds $SEED \
    --output-dir results \
    --output-suffix _seed${SEED}

# Note: samples_per_epoch=50000 is set in get_base_config()

STATUS=$?
if [ $STATUS -eq 0 ]; then
    echo "[SUCCESS] Ablation: $ABLATION, Seed: $SEED"
else
    echo "[FAILED] Ablation: $ABLATION, Seed: $SEED (exit=$STATUS)"
fi
echo "End: $(date)"
