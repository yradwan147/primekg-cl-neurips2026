# MCGL SLURM Jobs Guide

## Overview

The MCGL experiment suite consists of **20 independent SLURM jobs** submitted via `submit_all.sh`. They cover the full experiment matrix: **8 experiment types** across **3 evaluation tasks** (Link Prediction, KGQA, Node Classification).

**Cluster:** KAUST IBEX
**Submit all:** `bash slurm/submit_all.sh`
**Monitor:** `squeue -u $USER`
**Logs:** `slurm/slurm_logs/`
**Results:** `results/*.json`

---

## Common SBATCH Configuration

All jobs share these defaults unless noted otherwise:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--time` | `24:00:00` | 24 hours max wall time |
| `--nodes` | `1` | Single node |
| `--gpus-per-node` | `1` | 1 GPU per node |
| `--constraint` | `v100` | NVIDIA V100 (32GB VRAM) |
| `--partition` | `batch` | Standard batch queue |
| `--cpus-per-gpu` | `2` | 2 CPU cores per GPU |
| `--mem` | `32G` | 32GB system RAM |

**Environment setup** (all scripts):
```bash
source ~/miniconda3/bin/activate
conda activate mcgl
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

The `PYTORCH_CUDA_ALLOC_CONF` setting reduces CUDA memory fragmentation, which is critical for PyKEEN's all-entity scoring with 129K+ entities.

---

## Script Inventory

| Script | Job Name | Purpose |
|--------|----------|---------|
| `base_job.sh` | `mcgl` | Template — not submitted directly |
| `run_baseline.sh` | `mcgl_bl` | KGE baselines for Link Prediction |
| `run_cmkl.sh` | `mcgl_cmkl` | CMKL model for Link Prediction |
| `run_ablations.sh` | `mcgl_abl` | CMKL ablation studies |
| `run_lkge.sh` | `mcgl_lkge` | LKGE external baseline |
| `run_rag.sh` | `mcgl_rag` | RAG agent for KGQA |
| `run_nc.sh` | `mcgl_nc` | Node Classification experiments |
| `submit_all.sh` | — | Submits all 20 jobs |

---

## Job Details

### 1. Link Prediction: KGE Baselines (4 jobs)

**Script:** `slurm/run_baseline.sh`
**Submission:**
```bash
sbatch slurm/run_baseline.sh naive_sequential TransE
sbatch slurm/run_baseline.sh joint_training TransE
sbatch slurm/run_baseline.sh ewc TransE
sbatch slurm/run_baseline.sh experience_replay TransE
```

**Arguments:**
- `$1` (required): Baseline name — `naive_sequential`, `joint_training`, `ewc`, `experience_replay`
- `$2` (optional, default `TransE`): KGE decoder model

**Python command:**
```bash
python scripts/run_baselines.py \
    --baseline $BASELINE --model $MODEL \
    --embedding-dim 256 --num-epochs 100 --batch-size 512 \
    --device cuda --seeds 42 123 456 789 1024 --output-dir results
```

**Hyperparameters:**
- Embedding dimension: 256
- Epochs: 100 per task
- Batch size: 512
- 5 seeds: [42, 123, 456, 789, 1024]

**Output files:**
- `results/naive_sequential_TransE.json`
- `results/joint_training_TransE.json`
- `results/ewc_TransE.json`
- `results/experience_replay_TransE.json`

**What each baseline does:**
- **Naive Sequential:** Trains on each task sequentially, no forgetting mitigation. Lower bound.
- **Joint Training:** Trains on all tasks simultaneously. Upper bound (not continual).
- **EWC (Elastic Weight Consolidation):** Regularizes important parameters via Fisher information matrix. Uniform lambda across all parameters.
- **Experience Replay:** Stores random exemplars from past tasks and replays them during training.

**Metrics computed:** MRR, Hits@1, Hits@3, Hits@10, AP, AF, BWT, FWT, REM

---

### 2. Link Prediction: LKGE (1 job)

**Script:** `slurm/run_lkge.sh`
**Submission:**
```bash
sbatch slurm/run_lkge.sh TransE
```

**Arguments:**
- `$1` (optional, default `TransE`): KGE model for LKGE

**Special behavior:** Automatically clones the LKGE repository into `external/LKGE` if not present, and installs its requirements.

**Python command:**
```bash
python scripts/run_lkge.py \
    --model $MODEL --num-epochs 100 \
    --seeds 42 123 456 789 1024 --output-dir results
```

**What it does:** Runs the external LKGE (Lifelong Knowledge Graph Embedding) framework as a subprocess. Converts our benchmark data to LKGE's expected format, executes LKGE training, parses the log output using regex patterns, and computes CL metrics.

**Output:** `results/lkge_TransE.json`

---

### 3. Link Prediction: CMKL (1 job)

**Script:** `slurm/run_cmkl.sh`
**Submission:**
```bash
sbatch slurm/run_cmkl.sh DistMult
```

**Arguments:**
- `$1` (optional, default `DistMult`): Decoder — `TransE`, `DistMult`, or `Bilinear`

**Python command:**
```bash
python scripts/run_cmkl.py \
    --decoder $DECODER --embedding-dim 256 --num-epochs 100 --batch-size 512 \
    --device cuda --seeds 42 123 456 789 1024 --output-dir results
```

**What it does:** Trains the full CMKL model (our proposed method) with:
- 3 modality-specific encoders (R-GCN, BiomedBERT, Morgan FP MLP)
- Cross-modal attention fusion
- Modality-aware EWC (per-modality Fisher matrices with per-modality lambdas)
- Multimodal memory replay (K-means diverse selection)
- DistMult decoder (default, best in smoke tests)

**Output:** `results/cmkl_DistMult.json`

---

### 4. Link Prediction: Ablation Studies (8 jobs)

**Script:** `slurm/run_ablations.sh`
**Submission:**
```bash
sbatch slurm/run_ablations.sh struct_only
sbatch slurm/run_ablations.sh text_only
sbatch slurm/run_ablations.sh concat_fusion
sbatch slurm/run_ablations.sh global_ewc
sbatch slurm/run_ablations.sh random_replay
sbatch slurm/run_ablations.sh buffer_size_sweep
sbatch slurm/run_ablations.sh lambda_sweep
sbatch slurm/run_ablations.sh distillation
```

**Arguments:**
- `$1` (required): Ablation name (one of the 8 above)

**Python command:**
```bash
python scripts/run_ablations.py \
    --ablation $ABLATION --embedding-dim 256 --num-epochs 100 --batch-size 512 \
    --device cuda --seeds 42 123 456 789 1024 --output-dir results
```

**Ablation descriptions:**

| Ablation | What changes | Tests hypothesis |
|----------|-------------|-----------------|
| `struct_only` | Only structural encoder (R-GCN), no text/molecular | Multimodal features help |
| `text_only` | Only textual encoder (BiomedBERT), no struct/molecular | Structural info matters |
| `concat_fusion` | Replace cross-attention with simple concatenation | Cross-attention > concat |
| `global_ewc` | Use single global lambda instead of per-modality lambdas | Per-modality EWC > global |
| `random_replay` | Random exemplar selection instead of K-means diverse | Diverse replay > random |
| `buffer_size_sweep` | Tests buffer sizes [50, 100, 200, 500, 1000] | Optimal buffer size |
| `lambda_sweep` | Tests EWC lambdas [0.01, 0.1, 1.0, 10.0, 100.0] | Optimal regularization strength |
| `distillation` | Adds knowledge distillation (T=2.0, alpha=0.5) | Distillation helps |

**Output files:**
- `results/ablation_struct_only.json`
- `results/ablation_text_only.json`
- `results/ablation_concat_fusion.json`
- `results/ablation_global_ewc.json`
- `results/ablation_random_replay.json`
- `results/ablation_buffer_size_sweep.json`
- `results/ablation_lambda_sweep.json`
- `results/ablation_distillation.json`

---

### 5. KGQA: RAG Agent (1 job)

**Script:** `slurm/run_rag.sh`
**Submission:**
```bash
sbatch slurm/run_rag.sh
```

**Arguments:** None (no positional args)

**Resource overrides:**
- `--mem=64G` (double the default — LLM needs more RAM)
- `--cpus-per-gpu=4` (more CPU for text processing)

**Python command:**
```bash
python scripts/run_rag.py \
    --questions-per-task 200 \
    --seeds 42 123 456 789 1024 --output-dir results
```

**What it does:** Evaluates a Retrieval-Augmented Generation agent on continual biomedical KGQA:
1. For each CL task, indexes the KG snapshot into ChromaDB (PubMedBERT embeddings)
2. Generates 200 QA questions per task from test triples using relation-type templates
3. For each question: retrieves top-K relevant triples, generates answer via LLM (or retrieval-only fallback)
4. Evaluates: Exact Match, Token F1, Accuracy
5. Builds a CL results matrix tracking performance across all tasks seen so far

**Output:** `results/rag_agent.json`

---

### 6. Node Classification (5 jobs)

**Script:** `slurm/run_nc.sh`
**Submission:**
```bash
sbatch slurm/run_nc.sh naive_sequential
sbatch slurm/run_nc.sh joint_training
sbatch slurm/run_nc.sh ewc
sbatch slurm/run_nc.sh experience_replay
sbatch slurm/run_nc.sh cmkl
```

**Arguments:**
- `$1` (required): Method — `naive_sequential`, `joint_training`, `ewc`, `experience_replay`, `cmkl`

**Python command:**
```bash
python scripts/run_nc.py \
    --method $METHOD --embedding-dim 256 --num-epochs 100 --batch-size 512 \
    --device cuda --seeds 42 123 456 789 1024 --output-dir results
```

**What it does:**
- For KGE baselines (4 methods): Trains KGE continually on link prediction, extracts frozen entity embeddings, trains a 2-layer MLP classifier on PrimeKG's 10 node types
- For CMKL: Uses the fused multimodal embeddings + integrated NC classifier head

**Node types classified:** anatomy, biological_process, cellular_component, disease, drug, effect/phenotype, gene/protein, molecular_function, pathway (10 total from PrimeKG)

**Output files:**
- `results/nc_naive_sequential.json`
- `results/nc_joint_training.json`
- `results/nc_ewc.json`
- `results/nc_experience_replay.json`
- `results/nc_cmkl.json`

**Metrics:** Accuracy, Macro-F1, Weighted-F1

---

## Job Summary Table

| # | Job | Script | Args | Task | Output |
|---|-----|--------|------|------|--------|
| 1 | Naive Sequential LP | `run_baseline.sh` | `naive_sequential TransE` | Link Prediction | `naive_sequential_TransE.json` |
| 2 | Joint Training LP | `run_baseline.sh` | `joint_training TransE` | Link Prediction | `joint_training_TransE.json` |
| 3 | EWC LP | `run_baseline.sh` | `ewc TransE` | Link Prediction | `ewc_TransE.json` |
| 4 | Experience Replay LP | `run_baseline.sh` | `experience_replay TransE` | Link Prediction | `experience_replay_TransE.json` |
| 5 | LKGE LP | `run_lkge.sh` | `TransE` | Link Prediction | `lkge_TransE.json` |
| 6 | CMKL LP | `run_cmkl.sh` | `DistMult` | Link Prediction | `cmkl_DistMult.json` |
| 7 | Ablation: struct_only | `run_ablations.sh` | `struct_only` | Link Prediction | `ablation_struct_only.json` |
| 8 | Ablation: text_only | `run_ablations.sh` | `text_only` | Link Prediction | `ablation_text_only.json` |
| 9 | Ablation: concat_fusion | `run_ablations.sh` | `concat_fusion` | Link Prediction | `ablation_concat_fusion.json` |
| 10 | Ablation: global_ewc | `run_ablations.sh` | `global_ewc` | Link Prediction | `ablation_global_ewc.json` |
| 11 | Ablation: random_replay | `run_ablations.sh` | `random_replay` | Link Prediction | `ablation_random_replay.json` |
| 12 | Ablation: buffer_size_sweep | `run_ablations.sh` | `buffer_size_sweep` | Link Prediction | `ablation_buffer_size_sweep.json` |
| 13 | Ablation: lambda_sweep | `run_ablations.sh` | `lambda_sweep` | Link Prediction | `ablation_lambda_sweep.json` |
| 14 | Ablation: distillation | `run_ablations.sh` | `distillation` | Link Prediction | `ablation_distillation.json` |
| 15 | RAG Agent KGQA | `run_rag.sh` | (none) | KGQA | `rag_agent.json` |
| 16 | Naive Sequential NC | `run_nc.sh` | `naive_sequential` | Node Classification | `nc_naive_sequential.json` |
| 17 | Joint Training NC | `run_nc.sh` | `joint_training` | Node Classification | `nc_joint_training.json` |
| 18 | EWC NC | `run_nc.sh` | `ewc` | Node Classification | `nc_ewc.json` |
| 19 | Experience Replay NC | `run_nc.sh` | `experience_replay` | Node Classification | `nc_experience_replay.json` |
| 20 | CMKL NC | `run_nc.sh` | `cmkl` | Node Classification | `nc_cmkl.json` |

---

## Resource Estimates

| Task Group | Jobs | GPU Hours (max) | Total V100-Hours |
|------------|------|----------------|-----------------|
| LP Baselines | 4 | 24 each | 96 |
| LP LKGE | 1 | 24 | 24 |
| LP CMKL | 1 | 24 | 24 |
| LP Ablations | 8 | 24 each | 192 |
| KGQA RAG | 1 | 24 | 24 |
| NC All | 5 | 24 each | 120 |
| **Total** | **20** | | **480 V100-hours** |

Most jobs should complete well within 24 hours. The sweep ablations (buffer_size_sweep, lambda_sweep) test multiple hyperparameter values and may take longer.

---

## Troubleshooting

**OOM (Out of Memory):**
- All scripts request V100 GPUs (32GB VRAM) via `--gpus-per-node=1` + `--constraint=v100`
- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` is set in every script to reduce CUDA memory fragmentation
- If still OOM, reduce `--batch-size` from 512 to 256

**Job stuck in queue:**
- V100s are in high demand. Check `squeue -u $USER` for status
- Consider using `--constraint=a100` if available and queue is shorter

**LKGE clone fails:**
- Check internet access from compute node
- Pre-clone on login node: `git clone https://github.com/nju-websoft/LKGE.git external/LKGE`

**RAG agent LLM unavailable:**
- The script falls back to retrieval-only mode if the LLM fails to load
- Use `--no-llm` flag to explicitly disable LLM and use only retrieval
