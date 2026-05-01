#!/bin/bash --login
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=v100
#SBATCH --partition=batch
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=350G
#SBATCH -o slurm/slurm_logs/%x_%J.out

# Run IncDE on PrimeKG-CL benchmark
# Usage:
#   sbatch -J incde_s42 slurm/run_incde.sh 42

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate mcgl

mkdir -p slurm/slurm_logs

SEED=${1:?Must provide seed}

echo "Job ID: $SLURM_JOB_ID"
echo "IncDE on PrimeKG_CL, Seed: $SEED"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

cd ./external/IncDE

# IncDE uses TransE only, emb_dim=200 by default
# Reduced to 100 for memory on 123K entities
# 10 snapshots (all tasks including base)
python main.py \
    -dataset PrimeKG_CL \
    -snapshot_num 10 \
    -emb_dim 50 \
    -batch_size 512 \
    -epoch_num 100 \
    -learning_rate 1e-4 \
    -margin 8.0 \
    -neg_ratio 10 \
    -patience 10 \
    -num_old_triples 20000 \
    -random_seed "$((SEED))" \
    -gpu 0 \
    -using_embedding_distill True \
    -use_multi_layers False \
    -use_two_stage True \
    -using_mask_weight False \
    2>&1 | tee ./results/incde_TransE_seed${SEED}.log

echo "End: $(date)"
echo "[SUCCESS] IncDE seed=$SEED complete"
