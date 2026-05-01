"""Run node classification experiments on the temporal benchmark.

For KGE baselines: trains KGE model continually, extracts frozen embeddings,
trains MLP classifier per task, evaluates on all tasks.
For CMKL: uses fused embeddings from the CMKL model + MLP classifier.

Usage:
    # Quick local test
    python scripts/run_nc.py --method naive_sequential --quick

    # Full run for a specific method
    python scripts/run_nc.py --method cmkl --seeds 42 123 456 789 1024

    # All methods
    python scripts/run_nc.py --method all --seeds 42 123 456 789 1024
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

SEEDS = [42, 123, 456, 789, 1024]
METHODS = [
    "naive_sequential",
    "joint_training",
    "ewc",
    "experience_replay",
    "si",
    "distillation",
    "mir_replay",
    "cmkl",
]


def run_nc_kge_baseline(
    method: str,
    task_seq: dict,
    task_names: list[str],
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
    nc_tasks: dict,
    model_name: str,
    embedding_dim: int,
    num_epochs: int,
    batch_size: int,
    device: str,
    seed: int,
) -> dict:
    """Run NC evaluation for a KGE baseline.

    Trains KGE continually, extracts embeddings, trains MLP classifier.

    Args:
        method: Baseline name.
        task_seq: CL task sequence.
        task_names: Task names.
        entity_to_id: Entity mapping.
        relation_to_id: Relation mapping.
        nc_tasks: NC datasets per task.
        model_name: PyKEEN model name.
        embedding_dim: Embedding dimension.
        num_epochs: Training epochs.
        batch_size: Batch size.
        device: Device.
        seed: Random seed.

    Returns:
        Dict with results_matrix and metrics.
    """
    from src.baselines._base import (
        make_triples_factory,
        create_model, train_epoch, get_device,
        _generate_negatives, _margin_loss,
    )
    from src.baselines.nc_baseline import NCBaseline
    from src.baselines.ewc import EWC_KGE
    from src.baselines.experience_replay import ExperienceReplayKGE
    from src.evaluation.metrics import evaluate_continual_learning

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device(device)

    # Create initial model from first task
    first_task = task_seq[task_names[0]]
    tf = make_triples_factory(first_task["train"], entity_to_id, relation_to_id)
    model = create_model(model_name, tf, embedding_dim=embedding_dim, random_seed=seed)
    model = model.to(device)

    num_tasks = len(task_names)
    nc_task_names = [t for t in task_names if t in nc_tasks]
    if not nc_task_names:
        logger.warning("No NC tasks available")
        return {"error": "no_nc_tasks"}

    n_nc = len(nc_task_names)
    results_matrix = np.zeros((n_nc, n_nc))

    nc_baseline = NCBaseline(
        embedding_dim=embedding_dim,
        num_classes=10,
        lr=0.01,
        num_epochs=100,
    )

    nc_idx = 0

    # Method-specific state
    ewc = EWC_KGE(model, lambda_ewc=10.0) if method == "ewc" else None
    replay = ExperienceReplayKGE(
        buffer_size_per_task=500, selection_strategy="relation_balanced"
    ) if method == "experience_replay" else None

    for task_idx, task_name in enumerate(task_names):
        logger.info(f"=== Task {task_idx + 1}/{num_tasks}: {task_name} ({method}) ===")
        task_data = task_seq[task_name]
        train_data = task_data["train"]
        tf_train = make_triples_factory(train_data, entity_to_id, relation_to_id)

        # Method-specific training data preparation
        if method == "joint_training":
            # Joint: accumulate all triples
            all_triples = np.vstack([
                task_seq[t]["train"] for t in task_names[:task_idx + 1]
            ])
            tf_train = make_triples_factory(all_triples, entity_to_id, relation_to_id)
        elif method == "experience_replay" and replay is not None:
            # Mix current task with replay buffer
            replay_triples = replay.get_replay_triples()
            if replay_triples is not None:
                n_replay = min(
                    int(len(train_data) * 0.3), len(replay_triples)
                )
                replay_idx = np.random.choice(
                    len(replay_triples), n_replay, replace=True
                )
                combined = np.concatenate(
                    [train_data, replay_triples[replay_idx]], axis=0
                )
                tf_train = make_triples_factory(combined, entity_to_id, relation_to_id)
                logger.info(f"  Replay: {len(train_data)} current + {n_replay} replay")

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        if method == "ewc" and ewc is not None:
            # EWC: custom training loop with penalty
            model.train()
            mapped = tf_train.mapped_triples.to(device)
            for epoch in range(num_epochs):
                perm = torch.randperm(mapped.shape[0], device=device)
                shuffled = mapped[perm]
                for start in range(0, shuffled.shape[0], batch_size):
                    batch = shuffled[start:start + batch_size]
                    neg = _generate_negatives(batch, model.num_entities, device)
                    pos_scores = model.score_hrt(batch)
                    neg_scores = model.score_hrt(neg)
                    base_loss = _margin_loss(pos_scores, neg_scores)
                    total_loss = base_loss + ewc.ewc_loss()
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
        else:
            # Naive sequential, joint, experience_replay: standard training
            for epoch in range(num_epochs):
                train_epoch(model, tf_train, optimizer, device=device, batch_size=batch_size)

        # Post-task updates for CL methods
        if method == "ewc" and ewc is not None:
            ewc.compute_fisher(tf_train, device=device, n_samples=1000)
            logger.info("  Fisher computed")
        elif method == "experience_replay" and replay is not None:
            replay.add_task(train_data, task_idx)

        # If this task has NC data, evaluate on all NC tasks seen so far
        if task_name in nc_tasks:
            # Extract embeddings
            embeddings = NCBaseline.extract_pykeen_embeddings(model, entity_to_id)

            for eval_nc_idx, eval_name in enumerate(nc_task_names[:nc_idx + 1]):
                nc_data = nc_tasks[eval_name]
                node_embs = embeddings[nc_data["node_ids"]]
                metrics = nc_baseline.train_and_evaluate(
                    node_embs, nc_data["labels"],
                    nc_data["train_mask"], nc_data["val_mask"], nc_data["test_mask"],
                    device=device,
                )
                results_matrix[nc_idx, eval_nc_idx] = metrics["macro_f1"]
                logger.info(f"  NC eval {eval_name}: macro_f1={metrics['macro_f1']:.4f}")

            nc_idx += 1

    cl_metrics = evaluate_continual_learning(results_matrix, nc_task_names)
    return {
        "method": method,
        "results_matrix": results_matrix.tolist(),
        "task_names": nc_task_names,
        "cl_metrics": cl_metrics,
        "seed": seed,
    }


def run_nc_cmkl(
    task_seq: dict,
    task_names: list[str],
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
    nc_tasks: dict,
    embedding_dim: int,
    num_epochs: int,
    batch_size: int,
    device: str,
    seed: int,
    fusion_type: str = "cross_attention",
) -> dict:
    """Run NC evaluation using CMKL fused embeddings.

    Args:
        task_seq: CL task sequence.
        task_names: Task names.
        entity_to_id: Entity mapping.
        relation_to_id: Relation mapping.
        nc_tasks: NC datasets per task.
        embedding_dim: Embedding dimension.
        num_epochs: Training epochs.
        batch_size: Batch size.
        device: Device.
        seed: Random seed.

    Returns:
        Dict with results_matrix and metrics.
    """
    from src.models.cmkl import CMKL
    from src.baselines.nc_baseline import NCBaseline
    from src.evaluation.metrics import evaluate_continual_learning
    from src.utils.feature_loader import load_features

    features = load_features()

    num_entities = max(len(entity_to_id), features.get("num_entities", 0))
    num_relations = max(len(relation_to_id), features.get("num_relations", 0))

    config = {
        "num_entities": num_entities,
        "num_relations": num_relations,
        "embedding_dim": embedding_dim,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "samples_per_epoch": 50000,
        "decoder_type": "DistMult",
        "fusion_type": fusion_type,
        "mol_input_dim": features.get("mol_input_dim", 1024),
    }

    model = CMKL(config)
    # Train CMKL continually with multimodal features
    model.train_continually(
        task_sequence=task_seq,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        device=device,
        seed=seed,
        edge_index=features.get("edge_index"),
        edge_type=features.get("edge_type"),
        text_embeddings=features.get("text_embeddings"),
        mol_fingerprints=features.get("mol_fingerprints"),
        node_has_text=features.get("node_has_text"),
        node_has_mol=features.get("node_has_mol"),
    )

    # Extract fused embeddings with features (move to model's device)
    dev = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        fused = model.forward(
            edge_index=features.get("edge_index").to(dev) if features.get("edge_index") is not None else None,
            edge_type=features.get("edge_type").to(dev) if features.get("edge_type") is not None else None,
            text_embeddings=features.get("text_embeddings").to(dev) if features.get("text_embeddings") is not None else None,
            mol_fingerprints=features.get("mol_fingerprints").to(dev) if features.get("mol_fingerprints") is not None else None,
            node_has_text=features.get("node_has_text").to(dev) if features.get("node_has_text") is not None else None,
            node_has_mol=features.get("node_has_mol").to(dev) if features.get("node_has_mol") is not None else None,
        ).cpu().numpy()

    nc_task_names = [t for t in task_names if t in nc_tasks]
    n_nc = len(nc_task_names)
    results_matrix = np.zeros((n_nc, n_nc))

    nc_baseline = NCBaseline(
        embedding_dim=embedding_dim,
        num_classes=10,
        lr=0.01,
        num_epochs=100,
    )

    # Evaluate NC on each task
    for nc_idx, task_name in enumerate(nc_task_names):
        nc_data = nc_tasks[task_name]
        node_embs = fused[nc_data["node_ids"]]

        for eval_idx in range(nc_idx + 1):
            eval_name = nc_task_names[eval_idx]
            eval_data = nc_tasks[eval_name]
            eval_embs = fused[eval_data["node_ids"]]
            metrics = nc_baseline.train_and_evaluate(
                eval_embs, eval_data["labels"],
                eval_data["train_mask"], eval_data["val_mask"], eval_data["test_mask"],
                device=device,
            )
            results_matrix[nc_idx, eval_idx] = metrics["macro_f1"]
            logger.info(f"  NC eval {eval_name}: macro_f1={metrics['macro_f1']:.4f}")

    cl_metrics = evaluate_continual_learning(results_matrix, nc_task_names)
    return {
        "method": "cmkl",
        "results_matrix": results_matrix.tolist(),
        "task_names": nc_task_names,
        "cl_metrics": cl_metrics,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run node classification experiments")
    parser.add_argument(
        "--method", choices=METHODS + ["all"], required=True,
        help="Method to run",
    )
    parser.add_argument("--tasks-dir", default="data/benchmark/tasks")
    parser.add_argument("--task-names", nargs="+", default=None)
    parser.add_argument("--kg-csv", default="data/benchmark/snapshots/kg_t0.csv",
                        help="Path to KG CSV for node type extraction")
    parser.add_argument("--model", default="TransE",
                        choices=["TransE", "DistMult", "ComplEx", "RotatE"])
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--fusion", default="cross_attention",
                        choices=["cross_attention", "concatenation", "moe"],
                        help="Fusion type for CMKL method")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--output-suffix", default="",
                        help="Suffix for output filename (e.g. _seed42)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: dim=64, epochs=3")
    args = parser.parse_args()

    if args.quick:
        args.embedding_dim = 64
        args.num_epochs = 3
        args.batch_size = 256
        logger.info("Quick mode: dim=64, epochs=3")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.baselines._base import load_task_sequence
    from src.data.node_classification import build_nc_dataset, load_node_types

    # Load tasks
    task_seq, entity_to_id, relation_to_id = load_task_sequence(
        args.tasks_dir, args.task_names
    )
    task_names = list(task_seq.keys())

    # Load node types and build NC dataset
    node_types = load_node_types(kg_csv_path=args.kg_csv)
    if not node_types:
        logger.warning("No node types found. NC evaluation will be limited.")

    nc_tasks = build_nc_dataset(task_seq, entity_to_id, node_types)
    logger.info(f"NC tasks: {list(nc_tasks.keys())}")

    methods = METHODS if args.method == "all" else [args.method]

    for method in methods:
        logger.info(f"\n{'='*60}")
        logger.info(f"NC Method: {method}")
        logger.info(f"{'='*60}")
        print(f"[STARTED] method=nc_{method} seeds={args.seeds} "
              f"tasks={len(task_names)} epochs={args.num_epochs}")

        all_seed_results = []
        method_start = time.time()

        for seed in args.seeds:
            logger.info(f"  Seed {seed}")
            start = time.time()

            if method == "cmkl":
                result = run_nc_cmkl(
                    task_seq, task_names, entity_to_id, relation_to_id,
                    nc_tasks, args.embedding_dim, args.num_epochs,
                    args.batch_size, args.device, seed,
                    fusion_type=args.fusion,
                )
            else:
                result = run_nc_kge_baseline(
                    method, task_seq, task_names, entity_to_id, relation_to_id,
                    nc_tasks, args.model, args.embedding_dim, args.num_epochs,
                    args.batch_size, args.device, seed,
                )

            elapsed = time.time() - start
            result["training_time_s"] = elapsed

            if "cl_metrics" in result:
                cl = result["cl_metrics"]
                logger.info(f"  AP={cl['Average Performance (AP)']:.4f}, "
                            f"AF={cl['Average Forgetting (AF)']:.4f}")

            all_seed_results.append(result)

            if "cl_metrics" in result:
                seed_elapsed = time.time() - method_start
                cl = result["cl_metrics"]
                print(f"[PROGRESS] method=nc_{method} seed={seed} "
                      f"AP={cl['Average Performance (AP)']:.4f} "
                      f"elapsed={seed_elapsed:.0f}s")

        # Save results
        result_path = output_dir / f"nc_{method}{args.output_suffix}.json"
        with open(result_path, "w") as f:
            json.dump({
                "method": method,
                "task": "node_classification",
                "task_names": list(nc_tasks.keys()),
                "seeds": args.seeds,
                "results": all_seed_results,
            }, f, indent=2, default=str)
        logger.info(f"NC results saved to {result_path}")


if __name__ == "__main__":
    try:
        main()
        print("[SUCCESS] run_nc completed")
    except Exception as e:
        print(f"[FAILED] run_nc error={str(e)[:200]}")
        raise
