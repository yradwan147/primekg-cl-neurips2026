"""Full benchmark construction pipeline.

Runs the complete benchmark construction:
1. Load PrimeKG snapshots (t0, t1)
2. Compute temporal diffs
3. Create task sequences
4. Extract multimodal features
5. Create train/val/test splits
6. Save benchmark to disk

Usage:
    python scripts/build_benchmark.py --data-dir data/benchmark
    python scripts/build_benchmark.py --data-dir data/benchmark --simulate-t1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build temporal benchmark")
    parser.add_argument(
        "--data-dir",
        default="data/benchmark",
        help="Base directory for benchmark data",
    )
    parser.add_argument(
        "--strategy",
        choices=["entity_type", "relation_type", "temporal"],
        default="entity_type",
        help="Task sequence strategy (default: entity_type)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Test set ratio",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--simulate-t1",
        action="store_true",
        help="Create a simulated t1 from t0 (for development)",
    )
    parser.add_argument(
        "--min-triples",
        type=int,
        default=100,
        help="Minimum triples per task (smaller tasks get merged)",
    )
    args = parser.parse_args()

    base_dir = Path(args.data_dir)
    snapshots_dir = base_dir / "snapshots"
    diffs_dir = base_dir / "diffs"
    tasks_dir = base_dir / "tasks"
    features_dir = base_dir / "features"

    for d in [diffs_dir, tasks_dir, features_dir]:
        d.mkdir(parents=True, exist_ok=True)

    t0_path = snapshots_dir / "kg_t0.csv"
    t1_path = snapshots_dir / "kg_t1.csv"
    t1_sim_path = snapshots_dir / "kg_t1_simulated.csv"

    start_time = time.time()

    # Step 1: Ensure snapshots exist
    print("=" * 60)
    print("Step 1: Loading snapshots")
    print("=" * 60)

    if not t0_path.exists():
        print(f"ERROR: t0 snapshot not found at {t0_path}")
        print("Run: python scripts/download_primekg.py --method tdc")
        sys.exit(1)

    if args.simulate_t1 or (not t1_path.exists() and not t1_sim_path.exists()):
        print("Creating simulated t1 from t0 (development mode)...")
        from src.data.temporal_diff import create_simulated_t1

        create_simulated_t1(
            str(t0_path), str(t1_sim_path),
            add_fraction=0.05, remove_fraction=0.01, seed=args.seed,
        )
        t1_path = t1_sim_path
    elif t1_sim_path.exists() and not t1_path.exists():
        t1_path = t1_sim_path
        print(f"Using simulated t1: {t1_path}")

    import pandas as pd

    print(f"Loading t0: {t0_path}")
    kg_t0 = pd.read_csv(str(t0_path), low_memory=False)
    print(f"  t0: {len(kg_t0):,} triples")

    print(f"Loading t1: {t1_path}")
    kg_t1 = pd.read_csv(str(t1_path), low_memory=False)
    print(f"  t1: {len(kg_t1):,} triples")

    # Step 2: Compute temporal diffs
    print()
    print("=" * 60)
    print("Step 2: Computing temporal diffs")
    print("=" * 60)

    from src.data.temporal_diff import compute_kg_diff, save_diff_report

    stats, added_df, removed_df, emerged = compute_kg_diff(
        str(t0_path), str(t1_path)
    )
    save_diff_report(stats, str(diffs_dir / "diff_t0_t1.json"))

    print(f"  Added:      {stats['diff']['added_triples']:,}")
    print(f"  Removed:    {stats['diff']['removed_triples']:,}")
    print(f"  Persistent: {stats['diff']['persistent_triples']:,}")
    print(f"  Emerged entities:  {stats['diff']['emerged_entities']:,}")
    print(f"  New relation types: {stats['diff']['emerged_relations']}")

    # Step 3: Create task sequences
    print()
    print("=" * 60)
    print(f"Step 3: Creating task sequences (strategy={args.strategy})")
    print("=" * 60)

    from src.data.task_sequence import create_task_sequence, validate_task_sequence

    tasks = create_task_sequence(
        kg_t0, kg_t1, strategy=args.strategy, include_base_task=True
    )
    tasks = validate_task_sequence(tasks, min_triples=args.min_triples)

    print(f"  Total tasks: {len(tasks)}")
    for name, df in tasks.items():
        print(f"    {name}: {len(df):,} triples")

    # Step 4: Extract multimodal features
    print()
    print("=" * 60)
    print("Step 4: Extracting multimodal features")
    print("=" * 60)

    from src.data.features import extract_multimodal_features

    drug_features_path = snapshots_dir / "drug_features_t0.csv"
    drug_feat, disease_feat = extract_multimodal_features(
        kg_path=str(t0_path),
        drug_features_path=str(drug_features_path),
        disease_features_path=None,
        output_dir=str(features_dir),
    )

    print(f"  Drug features: {len(drug_feat)} drugs, {drug_feat['has_text'].sum()} with text")
    print(f"  Disease features: {len(disease_feat)} diseases, {disease_feat['has_text'].sum()} with text")

    # Step 5: Create splits
    print()
    print("=" * 60)
    print("Step 5: Creating train/val/test splits")
    print("=" * 60)

    from src.data.splits import create_splits_per_task, save_splits, verify_no_leakage

    splits = create_splits_per_task(
        tasks, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed
    )

    # Verify no leakage
    verify_no_leakage(splits)
    print("  No data leakage detected!")

    # Save splits
    save_splits(splits, str(tasks_dir))

    for name, s in splits.items():
        print(f"    {name}: train={s['n_train']:,}, val={s['n_val']:,}, test={s['n_test']:,}")

    # Step 6: Save benchmark statistics
    print()
    print("=" * 60)
    print("Step 6: Saving benchmark statistics")
    print("=" * 60)

    elapsed = time.time() - start_time

    benchmark_stats = {
        "benchmark_version": "1.0-simulated" if "simulated" in str(t1_path) else "1.0",
        "strategy": args.strategy,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "min_triples": args.min_triples,
        "snapshots": {
            "t0": {"path": str(t0_path), "num_triples": len(kg_t0)},
            "t1": {"path": str(t1_path), "num_triples": len(kg_t1)},
        },
        "diff": stats["diff"],
        "tasks": {
            name: {
                "num_triples": len(df),
                "splits": {
                    "train": splits[name]["n_train"],
                    "val": splits[name]["n_val"],
                    "test": splits[name]["n_test"],
                } if name in splits else None,
            }
            for name, df in tasks.items()
        },
        "features": {
            "num_drugs": len(drug_feat),
            "drugs_with_text": int(drug_feat["has_text"].sum()),
            "num_diseases": len(disease_feat),
            "diseases_with_text": int(disease_feat["has_text"].sum()),
            "total_nodes": len(pd.read_csv(str(features_dir / "node_index_map.csv"))),
        },
        "build_time_seconds": round(elapsed, 1),
    }

    stats_path = base_dir / "statistics.json"
    with open(stats_path, "w") as f:
        json.dump(benchmark_stats, f, indent=2)
    print(f"  Saved statistics to {stats_path}")

    # Create benchmark README
    readme_path = base_dir / "README.md"
    _write_benchmark_readme(readme_path, benchmark_stats)
    print(f"  Saved README to {readme_path}")

    print()
    print("=" * 60)
    print(f"Benchmark construction complete! ({elapsed:.1f}s)")
    print(f"Output directory: {base_dir}")
    print("=" * 60)


def _write_benchmark_readme(path: Path, stats: dict) -> None:
    """Write a README.md summarizing the benchmark."""
    lines = [
        "# MCGL Temporal Benchmark",
        "",
        f"**Version:** {stats['benchmark_version']}",
        f"**Strategy:** {stats['strategy']}",
        f"**Seed:** {stats['seed']}",
        "",
        "## Snapshots",
        f"- t0: {stats['snapshots']['t0']['num_triples']:,} triples",
        f"- t1: {stats['snapshots']['t1']['num_triples']:,} triples",
        "",
        "## Temporal Diff (t0 -> t1)",
        f"- Added: {stats['diff']['added_triples']:,}",
        f"- Removed: {stats['diff']['removed_triples']:,}",
        f"- Persistent: {stats['diff']['persistent_triples']:,}",
        "",
        "## Tasks",
    ]

    for name, info in stats["tasks"].items():
        s = info.get("splits")
        if s:
            lines.append(
                f"- **{name}**: {info['num_triples']:,} triples "
                f"(train={s['train']:,}, val={s['val']:,}, test={s['test']:,})"
            )
        else:
            lines.append(f"- **{name}**: {info['num_triples']:,} triples")

    lines.extend([
        "",
        "## Features",
        f"- Total nodes: {stats['features']['total_nodes']:,}",
        f"- Drugs: {stats['features']['num_drugs']:,} ({stats['features']['drugs_with_text']} with text)",
        f"- Diseases: {stats['features']['num_diseases']:,} ({stats['features']['diseases_with_text']} with text)",
        "",
        "## Directory Structure",
        "```",
        "benchmark/",
        "├── snapshots/     # Raw KG CSV files",
        "├── diffs/         # Temporal diff JSON reports",
        "├── tasks/         # Train/val/test splits per task",
        "├── features/      # Node features and index maps",
        "├── statistics.json",
        "└── README.md",
        "```",
    ])

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
