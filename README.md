# PrimeKG-CL: A Continual Graph Learning Benchmark on Evolving Biomedical Knowledge Graphs

Anonymous code release accompanying the NeurIPS 2026 Datasets and Benchmarks Track submission **PrimeKG-CL**. This repository contains the benchmark construction pipeline, all baseline implementations, and evaluation scripts. The benchmark dataset (snapshots, task splits, multimodal features, stratified test split) is released separately as a downloadable archive.

## What is PrimeKG-CL?

PrimeKG-CL is a continual graph learning (CGL) benchmark built on a real biomedical knowledge graph with **genuine temporal evolution**, in contrast to prior CGL benchmarks that synthetically partition static generic KGs.

- **Two real temporal snapshots** of PrimeKG: $t_0$ (June 2021, 129K+ nodes, 8.1M+ edges) and $t_1$ (July 2023, reconstructed by re-querying nine freely accessible upstream databases).
- **5.83M added, 889K removed, 7.21M persistent edges** between snapshots — genuine, structured evolution rather than random splits.
- **10 entity-type-grouped continual tasks** (gene/protein, disease, drug, phenotype, anatomy, ...).
- **Three evaluation tracks**: biomedical relationship prediction (link prediction), biomedical entity classification (node classification), KGQA.
- **Multimodal node features**: BiomedBERT text embeddings, Morgan molecular fingerprints, R-GCN structural embeddings.
- **Persistent / added / removed test stratification** for separating *retention of still-true knowledge* from *correct unlearning of deprecated knowledge*.

## Repository layout

```
primekg-cl/
├── src/
│   ├── data/             # benchmark construction (snapshot building, splits, task sequence,
│   │                     # KGQA generation, node-classification labels, multimodal features)
│   ├── baselines/        # 10 baseline methods (Naive, Joint, EWC, ER, SI, Distillation,
│   │                     # MIR, LKGE wrapper, RAG agent, NC baseline) on 4 KGE decoders
│   │                     # (TransE, DistMult, ComplEx, RotatE)
│   ├── continual/        # CL components used by baselines (e.g., distillation)
│   ├── evaluation/       # filtered MRR, AP/AF/BWT/REM, stratified per-stratum evaluation
│   └── utils/
├── scripts/              # entry-point scripts: build_benchmark.py, run_baselines.py,
│                         # run_lkge.py, run_nc.py, run_rag.py, build_stratified_table.py, ...
├── slurm/                # SLURM job scripts (cluster-agnostic; edit modules/conda env)
├── configs/              # per-experiment YAML configs
├── requirements.txt
└── environment.yml
```

## Quick start

```bash
# 1. Set up the environment
conda env create -f environment.yml
conda activate mcgl

# 2. Download the benchmark dataset (released alongside this repo)
#    Place the unpacked archive at ./data/benchmark/ so that the directory looks like:
#       data/benchmark/snapshots/{kg_t0.csv, kg_t1.csv}
#       data/benchmark/tasks/task_{0..9}_*/...
#       data/benchmark/features/{text_embeddings.pt, mol_features.pt, edge_index.pt, ...}
#       data/benchmark/test_stratification.json

# 3. Run a baseline (example: EWC on DistMult, 5 seeds)
python scripts/run_baselines.py \
    --method ewc --decoder DistMult \
    --seeds 42 123 456 789 1024 \
    --data_root data/benchmark \
    --output results/

# 4. Aggregate results and produce the main results table
python scripts/merge_seed_results.py
python scripts/generate_tables.py
```

## Reproducing the paper

The main paper reports a 6 (CL methods) × 4 (KGE decoders) matrix. The full sweep with 5 seeds per cell takes approximately 36 hours on an NVIDIA V100 (32 GB) per (method, decoder) pair. The `slurm/` directory contains job scripts for running on a cluster.

| Result | Script |
|---|---|
| Main results matrix (Table 3) | `slurm/submit_run15_decoders.sh` (cell-by-cell) |
| Stratified evaluation (Table 5) | `slurm/run_baseline_stratified.sh` + `scripts/build_stratified_table.py` |
| Node classification (Table 4) | `slurm/run_nc.sh` |
| KGQA (per-task token F1) | `slurm/run_rag.sh` |
| Per-task learning matrices (Fig 4 supp) | `scripts/generate_learning_matrix_figures.py` |

## Reconstructing the $t_1$ snapshot

The $t_1$ snapshot is built by re-querying nine upstream databases (Bgee, CTD, GO, Gene2GO, HPO, HPOA, MONDO, Uberon, HGNC). The pipeline in `scripts/build_real_t1.py` is reproducible end-to-end; it documents the access date and source URL for each upstream resource. Seven additional databases (DrugBank, UMLS, DrugCentral, SIDER, DisGeNET) require restrictive licensing and are carried forward from $t_0$ unchanged.

## License

Code is released under the MIT license. See `LICENSE`.
The benchmark dataset (released separately) inherits PrimeKG's license; CC BY 4.0 for the derived components.

## External dependencies

- **PrimeKG** is the upstream knowledge graph (Chandak et al., 2023; not redistributed; downloaded from Harvard Dataverse).
- **PyKEEN** is the KGE library used for baselines.
- **LKGE** is wrapped as an external dependency (cloned at runtime by `scripts/run_lkge.py`).
- **IncDE** was attempted as a baseline but did not scale to our 5.67 M-triple base task; we document our configurations in `slurm/run_incde.sh` for transparency.

## Anonymous submission notice

This is an anonymous code release for double-blind peer review. Author identities, institutional affiliations, and external links to author repositories have been redacted. The code released here is functionally complete and reproduces all numbers reported in the paper.
