"""Convert PrimeKG-CL benchmark to IncDE format.

IncDE expects per-snapshot directories with:
  - train.txt, valid.txt, test.txt  (entity_str \\t relation_str \\t entity_str)
  - entity2id.txt                   (entity_str \\t id)
  - relation2id.txt                 (relation_str \\t id)
  - train_id.txt                    (head_id \\t rel_id \\t tail_id)

Our benchmark uses numeric IDs already (head_id \\t relation_name \\t tail_id).
We treat each task as a snapshot in IncDE's terminology.

IncDE's snapshots are CUMULATIVE (snapshot i includes all data from 0..i).
Our tasks are SEQUENTIAL (each task has its own train/valid/test).
We need to accumulate training data across tasks.

Usage:
    python scripts/convert_to_incde.py --tasks-dir data/benchmark/tasks --output-dir external/IncDE/data/PrimeKG_CL
"""

import argparse
import os
from pathlib import Path


def load_triples(filepath):
    """Load tab-separated triples from file."""
    triples = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                triples.append(tuple(parts))
    return triples


def main():
    parser = argparse.ArgumentParser(description="Convert PrimeKG-CL to IncDE format")
    parser.add_argument("--tasks-dir", default="data/benchmark/tasks",
                        help="Path to benchmark tasks directory")
    parser.add_argument("--output-dir", default="external/IncDE/data/PrimeKG_CL",
                        help="Output directory for IncDE format data")
    parser.add_argument("--skip-base", action="store_true",
                        help="Skip task_0_base (5.67M triples, may be too large for IncDE)")
    args = parser.parse_args()

    tasks_dir = Path(args.tasks_dir)
    output_dir = Path(args.output_dir)

    # Get ordered task list
    task_dirs = sorted([d for d in tasks_dir.iterdir() if d.is_dir()])
    task_names = [d.name for d in task_dirs]
    print(f"Found {len(task_names)} tasks: {task_names}")

    if args.skip_base:
        task_dirs = [d for d in task_dirs if d.name != "task_0_base"]
        task_names = [d.name for d in task_dirs]
        print(f"Skipping task_0_base, {len(task_names)} tasks remaining")

    # Collect all entities and relations across all tasks
    all_entities = set()
    all_relations = set()

    task_data = []
    for task_dir in task_dirs:
        train = load_triples(task_dir / "train.txt")
        valid = load_triples(task_dir / "valid.txt")
        test = load_triples(task_dir / "test.txt")

        for h, r, t in train + valid + test:
            all_entities.add(h)
            all_entities.add(t)
            all_relations.add(r)

        task_data.append({
            "name": task_dir.name,
            "train": train,
            "valid": valid,
            "test": test,
        })
        print(f"  {task_dir.name}: {len(train)} train, {len(valid)} valid, {len(test)} test")

    # Create global entity2id and relation2id mappings
    entity2id = {e: i for i, e in enumerate(sorted(all_entities))}
    relation2id = {r: i for i, r in enumerate(sorted(all_relations))}
    print(f"\nTotal entities: {len(entity2id)}, Total relations: {len(relation2id)}")

    # Generate per-snapshot directories (cumulative training data)
    cumulative_train = []

    for snapshot_id, data in enumerate(task_data):
        snap_dir = output_dir / str(snapshot_id)
        snap_dir.mkdir(parents=True, exist_ok=True)

        # Accumulate training data
        cumulative_train.extend(data["train"])

        # Write train.txt (cumulative)
        with open(snap_dir / "train.txt", "w") as f:
            for h, r, t in cumulative_train:
                f.write(f"{h}\t{r}\t{t}\n")

        # Write valid.txt (current task only)
        with open(snap_dir / "valid.txt", "w") as f:
            for h, r, t in data["valid"]:
                f.write(f"{h}\t{r}\t{t}\n")

        # Write test.txt (current task only)
        with open(snap_dir / "test.txt", "w") as f:
            for h, r, t in data["test"]:
                f.write(f"{h}\t{r}\t{t}\n")

        # Write entity2id.txt (global, same for all snapshots)
        with open(snap_dir / "entity2id.txt", "w") as f:
            for ent, eid in sorted(entity2id.items(), key=lambda x: x[1]):
                f.write(f"{ent}\t{eid}\n")

        # Write relation2id.txt (global, same for all snapshots)
        with open(snap_dir / "relation2id.txt", "w") as f:
            for rel, rid in sorted(relation2id.items(), key=lambda x: x[1]):
                f.write(f"{rel}\t{rid}\n")

        # Write train_id.txt (cumulative, numeric IDs)
        with open(snap_dir / "train_id.txt", "w") as f:
            for h, r, t in cumulative_train:
                f.write(f"{entity2id[h]}\t{relation2id[r]}\t{entity2id[t]}\n")

        print(f"  Snapshot {snapshot_id} ({data['name']}): "
              f"{len(cumulative_train)} cumulative train, "
              f"{len(data['valid'])} valid, {len(data['test'])} test")

    print(f"\nIncDE data written to {output_dir}")
    print(f"  Snapshots: {len(task_data)}")
    print(f"  Run: cd external/IncDE && python data_preprocess.py")
    print(f"  Then: python main.py -dataset PrimeKG_CL -snapshot_num {len(task_data)}")


if __name__ == "__main__":
    main()
