"""Run LKGE baseline on the temporal benchmark.

Converts our benchmark to LKGE format, runs LKGE as a subprocess,
parses results, and computes CL metrics.

Usage:
    # Quick mode: only test format conversion (no LKGE run)
    python scripts/run_lkge.py --quick

    # Full run (requires LKGE repo cloned to external/LKGE)
    python scripts/run_lkge.py --seeds 42 123 456 789 1024

    # Custom LKGE directory
    python scripts/run_lkge.py --lkge-dir /path/to/LKGE --model TransE
"""

from __future__ import annotations

import argparse
import json
import logging
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

SEEDS = [42, 123, 456, 789, 1024]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LKGE baseline")
    parser.add_argument("--tasks-dir", default="data/benchmark/tasks")
    parser.add_argument("--task-names", nargs="+", default=None)
    parser.add_argument("--model", default="TransE",
                        choices=["TransE", "DistMult", "ComplEx", "RotatE"])
    parser.add_argument("--lkge-dir", default="external/LKGE",
                        help="Path to cloned LKGE repository")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--emb-dim", type=int, default=50,
                        help="Embedding dimension (default 50 to avoid OOM on large KGs)")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--skip-base-task", action="store_true",
                        help="Skip task_0_base (5.67M triples) to avoid GCN OOM")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--output-suffix", default="",
                        help="Suffix for output filename (e.g. _seed42)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: only test format conversion, no LKGE run")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.baselines._base import load_task_sequence
    from src.baselines.lkge import LKGEWrapper
    from src.evaluation.metrics import evaluate_continual_learning

    # Load tasks
    task_seq, entity_to_id, relation_to_id = load_task_sequence(
        args.tasks_dir, args.task_names
    )

    # Optionally skip task_0_base (too large for LKGE's GCN)
    if args.skip_base_task:
        from collections import OrderedDict
        task_seq = OrderedDict(
            (k, v) for k, v in task_seq.items() if k != "task_0_base"
        )
        if not task_seq:
            logger.error("No tasks remaining after skipping task_0_base")
            return
        logger.info(f"Skipped task_0_base, {len(task_seq)} tasks remaining")

    task_names = list(task_seq.keys())

    logger.info(f"Tasks: {task_names}")
    logger.info(f"Entities: {len(entity_to_id):,}, Relations: {len(relation_to_id)}")

    # Convert to LKGE format
    wrapper = LKGEWrapper(lkge_dir=args.lkge_dir)
    lkge_data_dir = str(output_dir / "lkge_format")
    wrapper.convert_to_lkge_format(task_seq, lkge_data_dir)

    # Verify format conversion
    lkge_path = Path(lkge_data_dir)
    assert (lkge_path / "entity2id.txt").exists(), "entity2id.txt not created"
    assert (lkge_path / "relation2id.txt").exists(), "relation2id.txt not created"
    n_snapshots = len([d for d in lkge_path.iterdir()
                       if d.is_dir() and d.name.isdigit()])
    logger.info(f"LKGE format: {n_snapshots} snapshots in {lkge_data_dir}")

    if args.quick:
        logger.info("Quick mode: format conversion successful, skipping LKGE run")
        # Create a dummy result for testing
        result = {
            "method": "lkge",
            "model": args.model,
            "task_names": task_names,
            "num_snapshots": n_snapshots,
            "lkge_data_dir": lkge_data_dir,
            "status": "format_conversion_only",
            "lkge_command": wrapper.get_run_command(
                lkge_data_dir, model=args.model, num_epochs=args.num_epochs),
        }
        result_path = output_dir / f"lkge_{args.model}_quick.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Quick result saved to {result_path}")
        return

    # Full run: execute LKGE for each seed
    all_seed_results = []
    print(f"[STARTED] method=lkge model={args.model} seeds={args.seeds} "
          f"emb_dim={args.emb_dim} snapshots={n_snapshots}")

    for seed in args.seeds:
        logger.info(f"\n{'='*60}")
        logger.info(f"LKGE Seed {seed}")
        logger.info(f"{'='*60}")
        start = time.time()

        seed_output = str(output_dir / f"lkge_output_seed{seed}")
        results = wrapper.run_and_parse(
            dataset_dir=lkge_data_dir,
            output_dir=seed_output,
            model=args.model,
            num_epochs=args.num_epochs,
            seed=seed,
            emb_dim=args.emb_dim,
            batch_size=args.batch_size,
        )
        elapsed = time.time() - start

        if "error" in results:
            logger.error(f"LKGE failed for seed {seed}: {results['error']}")
            continue

        results["seed"] = seed
        results["training_time_s"] = elapsed

        # Compute CL metrics if we have a results matrix
        if results.get("results_matrix"):
            R = np.array(results["results_matrix"])
            cl_metrics = evaluate_continual_learning(R, task_names)
            results.update(cl_metrics)
            logger.info(f"  AP={cl_metrics['Average Performance (AP)']:.4f}, "
                        f"AF={cl_metrics['Average Forgetting (AF)']:.4f}")

        all_seed_results.append(results)

        if "Average Performance (AP)" in results:
            print(f"[PROGRESS] method=lkge seed={seed} "
                  f"AP={results['Average Performance (AP)']:.4f} "
                  f"elapsed={elapsed:.0f}s")
        else:
            print(f"[PROGRESS] method=lkge seed={seed} elapsed={elapsed:.0f}s")

        # Save after each seed so partial results survive failures
        result_path = output_dir / f"lkge_{args.model}{args.output_suffix}.json"
        with open(result_path, "w") as f:
            json.dump({
                "method": "lkge",
                "model": args.model,
                "task_names": task_names,
                "seeds": args.seeds,
                "results": all_seed_results,
            }, f, indent=2, default=str)
        logger.info(f"Results saved to {result_path} ({len(all_seed_results)}/{len(args.seeds)} seeds)")


if __name__ == "__main__":
    try:
        main()
        print("[SUCCESS] run_lkge completed")
    except Exception as e:
        print(f"[FAILED] run_lkge error={str(e)[:200]}")
        raise
