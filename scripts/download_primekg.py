"""Download PrimeKG snapshots for benchmark construction.

Downloads PrimeKG t0 (June 2021) and prepares for t1 rebuild.
Run this script first before building the benchmark.

Usage:
    python scripts/download_primekg.py --method tdc --save-dir data/benchmark/snapshots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PrimeKG snapshots")
    parser.add_argument(
        "--method",
        choices=["tdc", "dataverse"],
        default="tdc",
        help="Download method (default: tdc)",
    )
    parser.add_argument(
        "--save-dir",
        default="data/benchmark/snapshots",
        help="Directory to save downloaded data",
    )
    parser.add_argument(
        "--snapshot",
        choices=["t0", "t1", "all"],
        default="t0",
        help="Which snapshot to download",
    )
    args = parser.parse_args()

    from src.data.download import download_primekg_t0, verify_primekg

    print(f"Downloading PrimeKG {args.snapshot} via {args.method}...")
    kg = download_primekg_t0(save_dir=args.save_dir, method=args.method)

    stats = verify_primekg(kg, expected_snapshot=args.snapshot)
    print(f"Verification: {stats}")


if __name__ == "__main__":
    main()
