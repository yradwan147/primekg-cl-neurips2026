#!/bin/bash --login
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=v100
#SBATCH --partition=batch
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=48G
#SBATCH -o slurm/slurm_logs/%x_%J.out

# Retrain a baseline and emit per-stratum MRR (persistent / removed / added)
# at the final task. Output JSON contains `results[].stratified` dict.
#
# Usage:
#   sbatch -J strat_ns_d_s42 slurm/run_baseline_stratified.sh naive_sequential DistMult 42

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate mcgl

mkdir -p results_stratified slurm/slurm_logs

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

BASELINE=${1:?Usage: sbatch run_baseline_stratified.sh <baseline> <model> <seed>}
MODEL=${2:-DistMult}
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
    --output-dir results_stratified \
    --output-suffix _seed${SEED} \
    --eval-stratified

echo "End: $(date)"
