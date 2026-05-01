"""Preprocess PrimeKG-CL data for IncDE.

Generates centrality files needed by IncDE. Uses degree centrality instead of
betweenness centrality for scalability (PrimeKG has millions of edges where
NetworkX betweenness is infeasible).

Also generates the multi-layer ordering file based on edge degree sum.

Usage:
    python scripts/preprocess_incde_primekgcl.py --data-dir external/IncDE/data/PrimeKG_CL --num-snapshots 10
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path


def compute_degree_centrality(triples):
    """Compute degree centrality for all nodes (scalable to millions of edges)."""
    degree = defaultdict(int)
    for h, r, t in triples:
        degree[h] += 1
        degree[t] += 1
    max_deg = max(degree.values()) if degree else 1
    return {node: deg / max_deg for node, deg in degree.items()}


def compute_edge_importance(triples, node_degree):
    """Compute edge importance as sum of endpoint degrees (proxy for betweenness)."""
    edge_importance = {}
    for h, r, t in triples:
        key = (h, t)
        edge_importance[key] = node_degree.get(h, 0) + node_degree.get(t, 0)
    return edge_importance


def load_id_triples(filepath):
    """Load numeric ID triples."""
    triples = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                triples.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return triples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="external/IncDE/data/PrimeKG_CL")
    parser.add_argument("--num-snapshots", type=int, default=10)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    for snap_id in range(args.num_snapshots):
        snap_dir = data_dir / str(snap_id)
        if not snap_dir.exists():
            print(f"Snapshot {snap_id} not found, stopping")
            break

        train_id_path = snap_dir / "train_id.txt"
        if not train_id_path.exists():
            print(f"  Snapshot {snap_id}: no train_id.txt, skipping")
            continue

        triples = load_id_triples(train_id_path)
        print(f"  Snapshot {snap_id}: {len(triples)} triples")

        # Degree centrality (scalable)
        node_degree = compute_degree_centrality(triples)

        # Write train_nodes_degree.txt
        with open(snap_dir / "train_nodes_degree.txt", "w") as f:
            for node, deg in sorted(node_degree.items()):
                f.write(f"{node}\t{deg:.6f}\n")

        # Use degree as proxy for betweenness (exact betweenness infeasible at this scale)
        with open(snap_dir / "train_nodes_betweenness.txt", "w") as f:
            for node, deg in sorted(node_degree.items()):
                f.write(f"{node}\t{deg:.6f}\n")

        # Edge importance (sum of endpoint degrees as betweenness proxy)
        edge_imp = compute_edge_importance(triples, node_degree)
        with open(snap_dir / "train_edges_betweenness.txt", "w") as f:
            for (h, t), imp in edge_imp.items():
                f.write(f"{h}\t{t}\t{imp:.6f}\n")

        # Multi-layer ordering: sort triples by edge importance (high → low)
        triple_importance = []
        for h, r, t in triples:
            imp = node_degree.get(h, 0) + node_degree.get(t, 0)
            triple_importance.append((h, r, t, imp))
        triple_importance.sort(key=lambda x: -x[3])

        with open(snap_dir / "train_sorted_by_edges_betweenness.txt", "w") as f:
            for h, r, t, imp in triple_importance:
                f.write(f"{h}\t{r}\t{t}\n")

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
