"""Build a real PrimeKG t1 snapshot from source databases.

Downloads enabled databases, processes them using PrimeKG-compatible logic,
and assembles a partial or full KG. Supports selective database inclusion
so you can build with whatever data you have access to.

Drug edges are carried from t0 when DrugBank is unavailable.

Usage:
    # Build with default config (all free databases + DisGeNET)
    python scripts/build_real_t1.py

    # Build with custom config
    python scripts/build_real_t1.py --config configs/t1_sources.yaml

    # Skip download (data already present)
    python scripts/build_real_t1.py --skip-download

    # Only download, don't build
    python scripts/build_real_t1.py --download-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build real PrimeKG t1 from source databases")
    parser.add_argument("--config", default="configs/t1_sources.yaml",
                        help="Path to database configuration YAML")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading (use existing data)")
    parser.add_argument("--download-only", action="store_true",
                        help="Only download data, don't build KG")
    parser.add_argument("--output", default=None,
                        help="Override output path for kg_t1.csv")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    with open(str(config_path)) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config.get("output_dir", "data/t1_sources"))
    snapshot_output = Path(args.output or config.get("snapshot_output", "data/benchmark/snapshots/kg_t1.csv"))
    carry_from = config.get("carry_drug_edges_from")

    # Load .env file for API keys
    env_path = Path(".env")
    if env_path.exists():
        with open(str(env_path)) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, val = line.split("=", 1)
                    import os
                    os.environ.setdefault(key.strip(), val.strip())

    # Print configuration summary
    databases = config.get("databases", {})
    enabled = [name for name, db in databases.items() if db.get("enabled")]
    disabled = [name for name, db in databases.items() if not db.get("enabled")]

    print("=" * 60)
    print("PrimeKG t1 Builder")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Data dir: {output_dir}")
    print(f"Output: {snapshot_output}")
    print(f"Carry drug edges from: {carry_from or 'none'}")
    print(f"\nEnabled databases ({len(enabled)}):")
    for name in enabled:
        desc = databases[name].get("description", "")
        print(f"  + {name}: {desc}")
    print(f"\nDisabled databases ({len(disabled)}):")
    for name in disabled:
        note = databases[name].get("note", "")
        print(f"  - {name}: {note}")
    print()

    start_time = time.time()

    # Step 1: Download
    if not args.skip_download:
        print("=" * 60)
        print("Step 1: Downloading source databases")
        print("=" * 60)
        from src.data.kg_builder import download_sources
        download_sources(config, output_dir)
    else:
        print("Skipping download (--skip-download)")

    if args.download_only:
        elapsed = time.time() - start_time
        print(f"\nDownload complete! ({elapsed:.1f}s)")
        return

    # Step 2: Build KG
    print()
    print("=" * 60)
    print("Step 2: Processing databases and building KG")
    print("=" * 60)
    from src.data.kg_builder import build_kg
    kg = build_kg(output_dir, config, carry_from=carry_from)

    if len(kg) == 0:
        logger.error("KG is empty! Check logs above for errors.")
        sys.exit(1)

    # Step 3: Save
    print()
    print("=" * 60)
    print("Step 3: Saving KG snapshot")
    print("=" * 60)
    snapshot_output.parent.mkdir(parents=True, exist_ok=True)
    kg.to_csv(str(snapshot_output), index=False)
    size_mb = snapshot_output.stat().st_size / 1e6
    print(f"Saved to {snapshot_output} ({size_mb:.1f} MB)")

    # Step 4: Summary statistics
    print()
    print("=" * 60)
    print("KG Statistics")
    print("=" * 60)
    print(f"Total edges: {len(kg):,}")
    print(f"Unique nodes: {max(kg['x_index'].max(), kg['y_index'].max()) + 1:,}")
    print(f"Relation types: {kg['relation'].nunique()}")
    print(f"Node types: {len(set(kg['x_type'].unique()) | set(kg['y_type'].unique()))}")

    print("\nEdge counts by relation:")
    rel_counts = kg["relation"].value_counts()
    for rel, count in rel_counts.items():
        print(f"  {rel}: {count:,}")

    print("\nNode counts by type:")
    import pandas as pd
    node_types = pd.concat([
        kg[["x_id", "x_type"]].rename(columns={"x_id": "id", "x_type": "type"}),
        kg[["y_id", "y_type"]].rename(columns={"y_id": "id", "y_type": "type"}),
    ]).drop_duplicates()
    for ntype, count in node_types["type"].value_counts().items():
        print(f"  {ntype}: {count:,}")

    # Save build info
    elapsed = time.time() - start_time
    build_info = {
        "build_time_seconds": round(elapsed, 1),
        "enabled_databases": enabled,
        "disabled_databases": disabled,
        "carried_drug_edges_from": str(carry_from) if carry_from else None,
        "total_edges": len(kg),
        "relation_types": kg["relation"].nunique(),
        "edge_counts": {str(k): int(v) for k, v in rel_counts.items()},
    }
    info_path = snapshot_output.parent / "t1_build_info.json"
    with open(str(info_path), "w") as f:
        json.dump(build_info, f, indent=2)
    print(f"\nBuild info saved to {info_path}")

    print()
    print("=" * 60)
    print(f"Build complete! ({elapsed:.1f}s)")
    print(f"Next: python scripts/build_benchmark.py  (uses real t1)")
    print("=" * 60)


if __name__ == "__main__":
    main()
