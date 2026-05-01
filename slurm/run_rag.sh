#!/bin/bash --login
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --gpus-per-node=1
#SBATCH --constraint=v100
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH -o slurm/slurm_logs/%x_%J.out

# Run RAG agent with Qwen2.5-7B-Instruct LLM
# Usage:
#   sbatch -J rag_s42 slurm/run_rag.sh 42

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate mcgl

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

mkdir -p results slurm/slurm_logs

SEED=${1:?Must provide seed}
shift 1  # remaining args passed to run_rag.py
EXTRA_ARGS="$@"

echo "Job ID: $SLURM_JOB_ID"
echo "RAG Seed: $SEED"
echo "Extra args: $EXTRA_ARGS"
echo "Start: $(date)"

python scripts/run_rag.py \
    --llm Qwen/Qwen2.5-7B-Instruct \
    --questions-per-task 200 \
    --seeds $SEED \
    --output-dir results \
    --output-suffix _seed${SEED} \
    $EXTRA_ARGS

echo "End: $(date)"
