"""Run baseline experiments on the temporal benchmark.

Supports 4 KGE baselines: naive_sequential, joint_training, ewc,
experience_replay. Runs with configurable seeds and logs results.

Usage:
    # Run a single baseline
    python scripts/run_baselines.py --baseline naive_sequential --tasks-dir data/benchmark/tasks

    # Run with specific tasks (skip the huge base task for local testing)
    python scripts/run_baselines.py --baseline naive_sequential \
        --task-names task_1_disease_related task_3_phenotype_related

    # Run all baselines
    python scripts/run_baselines.py --baseline all

    # Quick local test (small embedding, few epochs)
    python scripts/run_baselines.py --baseline naive_sequential --quick
"""

from __future__ import annotations

import argparse
import json
import logging
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def log_memory(label: str) -> None:
    """Log current RSS memory usage."""
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS returns bytes, Linux returns KB
    if sys.platform == "darwin":
        rss_mb = rss_mb / (1024 * 1024)
    else:
        rss_mb = rss_mb / 1024
    logger.info(f"[MEMORY] {label}: {rss_mb:.0f} MB RSS")


SEEDS = [42, 123, 456, 789, 1024]
KGE_BASELINES = [
    "naive_sequential", "joint_training", "ewc", "experience_replay",
    "si", "distillation", "mir_replay",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline experiments")
    parser.add_argument(
        "--baseline",
        choices=KGE_BASELINES + ["all"],
        required=True,
        help="Which baseline to run",
    )
    parser.add_argument(
        "--tasks-dir",
        default="data/benchmark/tasks",
        help="Path to benchmark tasks directory",
    )
    parser.add_argument(
        "--task-names",
        nargs="+",
        default=None,
        help="Specific task names to use (default: all tasks in directory)",
    )
    parser.add_argument("--model", default="TransE", help="KGE model type")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42],
        help="Random seeds (default: [42]; use --seeds 42 123 456 789 1024 for full)",
    )
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--output-suffix", default="",
                        help="Suffix for output filename (e.g. _seed42)")

    # EWC-specific
    parser.add_argument("--lambda-ewc", type=float, default=10.0)
    parser.add_argument("--fisher-samples", type=int, default=1000)

    # Replay-specific
    parser.add_argument("--buffer-size", type=int, default=500)
    parser.add_argument("--selection-strategy", default="relation_balanced")
    parser.add_argument("--replay-ratio", type=float, default=0.3)

    # SI-specific
    parser.add_argument("--lambda-si", type=float, default=1.0)
    parser.add_argument("--si-damping", type=float, default=0.1)

    # Distillation-specific
    parser.add_argument("--lambda-distill", type=float, default=5.0)

    # MIR-specific
    parser.add_argument("--mir-candidates", type=int, default=200)
    parser.add_argument("--mir-select", type=int, default=50)

    # Multi-hop evaluation
    parser.add_argument(
        "--eval-multihop", action="store_true",
        help="Run multi-hop path evaluation after training",
    )
    parser.add_argument(
        "--eval-stratified", action="store_true",
        help="After final training task, eval model on persistent / removed / added strata",
    )

    # Quick mode for local testing
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: small embedding (64), few epochs (10)",
    )

    args = parser.parse_args()

    if args.quick:
        args.embedding_dim = 64
        args.num_epochs = 10
        logger.info("Quick mode: embedding_dim=64, num_epochs=10")

    baselines = KGE_BASELINES if args.baseline == "all" else [args.baseline]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.baselines._base import load_task_sequence
    from src.evaluation.metrics import evaluate_continual_learning

    log_memory("before loading tasks")

    # Load tasks once (returns int arrays + mappings)
    task_seq, entity_to_id, relation_to_id = load_task_sequence(
        args.tasks_dir, args.task_names
    )
    task_names = list(task_seq.keys())
    logger.info(f"Loaded {len(task_names)} tasks: {task_names}")

    # Log per-task sizes
    for name, data in task_seq.items():
        total = sum(len(v) for v in data.values())
        logger.info(f"  {name}: {total:,} triples "
                    f"(train={len(data['train']):,})")
    log_memory("after loading tasks")

    total_start = time.time()

    for baseline_name in baselines:
        print(f"\n{'=' * 60}")
        print(f"Baseline: {baseline_name}")
        print(f"Model: {args.model}, dim={args.embedding_dim}, "
              f"epochs={args.num_epochs}, lr={args.lr}")
        print(f"Seeds: {args.seeds}")
        print(f"{'=' * 60}")
        print(f"[STARTED] method={baseline_name} seeds={args.seeds} "
              f"tasks={len(task_names)} epochs={args.num_epochs}")

        all_seed_results = []

        for seed in args.seeds:
            logger.info(f"\n--- Seed {seed} ---")
            start = time.time()

            log_memory(f"before {baseline_name} seed={seed}")
            R, trained_model = _run_baseline(
                baseline_name, task_seq,
                entity_to_id, relation_to_id,
                args, seed,
            )
            log_memory(f"after {baseline_name} seed={seed}")

            elapsed = time.time() - start
            logger.info(f"Seed {seed} completed in {elapsed:.1f}s")

            # Compute CL metrics
            cl_metrics = evaluate_continual_learning(R, task_names)
            cl_metrics["seed"] = seed
            cl_metrics["results_matrix"] = R.tolist()
            all_seed_results.append(cl_metrics)

            # Optional stratified eval on persistent / removed / added test triples
            if args.eval_stratified and trained_model is not None:
                try:
                    from src.evaluation.stratified import stratified_eval
                    logger.info("Running stratified eval (persistent / removed / added)...")
                    strat = stratified_eval(
                        trained_model, task_seq, entity_to_id, relation_to_id,
                        device=args.device, batch_size=args.batch_size,
                    )
                    cl_metrics["stratified"] = strat
                    for sname, m in strat.items():
                        logger.info(
                            f"  [stratum {sname}] MRR={m.get('MRR', float('nan')):.4f} "
                            f"H@10={m.get('Hits@10', float('nan')):.4f} n={m.get('n', 0)}"
                        )
                except Exception as exc:
                    logger.error(f"stratified eval failed: {exc}")
                    cl_metrics["stratified_error"] = str(exc)

            # Print summary
            for name, val in cl_metrics.items():
                if isinstance(val, float):
                    logger.info(f"  {name}: {val:.4f}")

            seed_elapsed = time.time() - total_start
            print(f"[PROGRESS] method={baseline_name} seed={seed} "
                  f"AP={cl_metrics['Average Performance (AP)']:.4f} "
                  f"AF={cl_metrics['Average Forgetting (AF)']:.4f} "
                  f"elapsed={seed_elapsed:.0f}s")

            # Save after each seed so partial results survive failures
            result_path = output_dir / f"{baseline_name}_{args.model}{args.output_suffix}.json"
            with open(result_path, "w") as f:
                json.dump({
                    "baseline": baseline_name,
                    "model": args.model,
                    "embedding_dim": args.embedding_dim,
                    "num_epochs": args.num_epochs,
                    "lr": args.lr,
                    "task_names": task_names,
                    "seeds": args.seeds,
                    "results": all_seed_results,
                }, f, indent=2)
            logger.info(f"Intermediate save: {result_path} ({len(all_seed_results)}/{len(args.seeds)} seeds)")

        # Multi-hop evaluation (if requested)
        multihop_results = None
        if args.eval_multihop:
            from src.evaluation.multihop import (
                extract_all_path_types,
                evaluate_multihop,
                make_pykeen_score_fn,
            )

            logger.info("Running multi-hop path evaluation...")
            all_train = np.concatenate(
                [data["train"] for data in task_seq.values()], axis=0,
            )
            all_paths = extract_all_path_types(
                all_train, relation_to_id, max_paths_per_type=5000,
            )
            multihop_results = {}
            for desc, paths in all_paths.items():
                if not paths:
                    continue
                multihop_results[desc] = {"num_paths": len(paths)}
                logger.info(f"  {desc}: {len(paths):,} paths extracted")

            # Score with the last trained model (if available)
            if trained_model is not None:
                score_fn = make_pykeen_score_fn(
                    trained_model, len(entity_to_id), device=args.device,
                )
                for desc, paths in all_paths.items():
                    if not paths:
                        continue
                    mh_metrics = evaluate_multihop(
                        score_fn, paths, len(entity_to_id),
                    )
                    multihop_results[desc].update(mh_metrics)
                    logger.info(f"  {desc}: MRR={mh_metrics['multihop_MRR']:.4f}, "
                                f"H@10={mh_metrics['multihop_Hits@10']:.4f}")

        # Save results
        result_path = output_dir / f"{baseline_name}_{args.model}{args.output_suffix}.json"
        output_data = {
            "baseline": baseline_name,
            "model": args.model,
            "embedding_dim": args.embedding_dim,
            "num_epochs": args.num_epochs,
            "lr": args.lr,
            "task_names": task_names,
            "seeds": args.seeds,
            "results": all_seed_results,
        }
        if multihop_results:
            output_data["multihop_paths"] = multihop_results
        with open(result_path, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {result_path}")

        # Print aggregate summary
        if len(all_seed_results) > 1:
            from src.evaluation.statistical import summarize_results
            summary = summarize_results(all_seed_results)
            print(f"\n--- {baseline_name} Summary ({len(args.seeds)} seeds) ---")
            for name, val in summary.items():
                if name not in ("seed", "results_matrix"):
                    print(f"  {name}: {val}")


def _run_baseline(
    name: str,
    task_seq: dict,
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
    args: argparse.Namespace,
    seed: int,
) -> tuple:
    """Run a single baseline with a single seed.

    Returns:
        Tuple of (results_matrix, trained_model) where trained_model is the
        PyKEEN model for multi-hop evaluation.
    """
    if name == "naive_sequential":
        from src.baselines.naive_sequential import NaiveSequentialTrainer
        trainer = NaiveSequentialTrainer(
            model_name=args.model,
            embedding_dim=args.embedding_dim,
            num_epochs=args.num_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=args.device,
            seed=seed,
        )
        R = trainer.train(task_seq, entity_to_id, relation_to_id)
        return R, trainer.model

    elif name == "joint_training":
        from src.baselines.joint_training import JointTrainer
        trainer = JointTrainer(
            model_name=args.model,
            embedding_dim=args.embedding_dim,
            num_epochs=args.num_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=args.device,
            seed=seed,
        )
        result = trainer.train(task_seq, entity_to_id, relation_to_id)
        return result["results_matrix"], getattr(trainer, 'model', None)

    elif name == "ewc":
        from src.baselines.ewc import EWCTrainer
        trainer = EWCTrainer(
            model_name=args.model,
            embedding_dim=args.embedding_dim,
            num_epochs=args.num_epochs,
            lr=args.lr,
            lambda_ewc=args.lambda_ewc,
            batch_size=args.batch_size,
            fisher_samples=args.fisher_samples,
            device=args.device,
            seed=seed,
        )
        R = trainer.train(task_seq, entity_to_id, relation_to_id)
        return R, getattr(trainer, 'model', None)

    elif name == "experience_replay":
        from src.baselines.experience_replay import ReplayTrainer
        trainer = ReplayTrainer(
            model_name=args.model,
            embedding_dim=args.embedding_dim,
            num_epochs=args.num_epochs,
            lr=args.lr,
            buffer_size_per_task=args.buffer_size,
            selection_strategy=args.selection_strategy,
            replay_ratio=args.replay_ratio,
            batch_size=args.batch_size,
            device=args.device,
            seed=seed,
        )
        R = trainer.train(task_seq, entity_to_id, relation_to_id)
        return R, getattr(trainer, 'model', None)

    elif name == "si":
        from src.baselines.synaptic_intelligence import SITrainer
        trainer = SITrainer(
            model_name=args.model,
            embedding_dim=args.embedding_dim,
            num_epochs=args.num_epochs,
            lr=args.lr,
            lambda_si=args.lambda_si,
            damping=args.si_damping,
            batch_size=args.batch_size,
            device=args.device,
            seed=seed,
        )
        R = trainer.train(task_seq, entity_to_id, relation_to_id)
        return R, getattr(trainer, 'model', None)

    elif name == "distillation":
        from src.baselines.knowledge_distillation import DistillationTrainer
        trainer = DistillationTrainer(
            model_name=args.model,
            embedding_dim=args.embedding_dim,
            num_epochs=args.num_epochs,
            lr=args.lr,
            lambda_distill=args.lambda_distill,
            batch_size=args.batch_size,
            device=args.device,
            seed=seed,
        )
        R = trainer.train(task_seq, entity_to_id, relation_to_id)
        return R, getattr(trainer, 'model', None)

    elif name == "mir_replay":
        from src.baselines.mir_replay import MIRReplayTrainer
        trainer = MIRReplayTrainer(
            model_name=args.model,
            embedding_dim=args.embedding_dim,
            num_epochs=args.num_epochs,
            lr=args.lr,
            buffer_size=args.buffer_size,
            mir_candidates=args.mir_candidates,
            mir_select=args.mir_select,
            batch_size=args.batch_size,
            device=args.device,
            seed=seed,
        )
        R = trainer.train(task_seq, entity_to_id, relation_to_id)
        return R, getattr(trainer, 'model', None)

    else:
        raise ValueError(f"Unknown baseline: {name}")


if __name__ == "__main__":
    try:
        main()
        print("[SUCCESS] run_baselines completed")
    except Exception as e:
        print(f"[FAILED] run_baselines error={str(e)[:200]}")
        raise
