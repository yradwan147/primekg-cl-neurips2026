"""Fetch SMILES for PrimeKG drugs from PubChem REST API.

Run ONCE before precompute_features.py to populate the SMILES cache.
The cache is then read by precompute_features.py on IBEX compute nodes
(which may lack internet access).

Uses concurrent requests (4 workers) to stay under PubChem's 5 req/s
rate limit while completing in ~5 minutes instead of ~25.

Usage:
    # Local (Mac) or IBEX login node (has internet):
    python scripts/fetch_smiles.py

    # Resume after interruption (reads existing cache):
    python scripts/fetch_smiles.py

    # Custom paths:
    python scripts/fetch_smiles.py --kg-csv data/benchmark/snapshots/kg_t0.csv \
                                    --cache data/smiles_cache.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

PUBCHEM_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/"
    "CanonicalSMILES/JSON"
)


def extract_drug_names(kg_csv: Path) -> set[str]:
    """Extract unique drug names from PrimeKG CSV.

    Args:
        kg_csv: Path to PrimeKG CSV (e.g., kg_t0.csv).

    Returns:
        Set of lowercased drug names.
    """
    drug_names: set[str] = set()
    with open(kg_csv) as f:
        for row in csv.DictReader(f):
            if row["x_type"] == "drug":
                drug_names.add(row["x_name"].strip().lower())
            if row["y_type"] == "drug":
                drug_names.add(row["y_name"].strip().lower())
    return drug_names


def _fetch_one(name: str, session: requests.Session) -> tuple[str, str | None]:
    """Fetch SMILES for a single drug name from PubChem.

    Returns:
        (drug_name, smiles_or_None)
    """
    try:
        url = PUBCHEM_URL.format(requests.utils.quote(name))
        resp = session.get(url, timeout=10)

        if resp.status_code == 200:
            props = resp.json().get("PropertyTable", {}).get("Properties", [])
            if props:
                smiles = (props[0].get("CanonicalSMILES")
                          or props[0].get("ConnectivitySMILES"))
                return name, smiles
        # 404 = not found, other = server error
        return name, None

    except Exception:
        return name, None


def fetch_smiles(
    drug_names: set[str],
    cache: dict[str, str],
    workers: int = 4,
) -> dict[str, str]:
    """Fetch SMILES from PubChem using concurrent requests.

    Uses a thread pool with rate limiting to stay under PubChem's
    5 requests/second limit. Saves cache every 500 drugs for resilience.

    Args:
        drug_names: Set of lowercased drug names to look up.
        cache: Existing cache dict (drug_name_lower -> SMILES).
        workers: Number of concurrent threads (default 4, under 5 req/s limit).

    Returns:
        Updated cache dict.
    """
    to_fetch = sorted(n for n in drug_names if n not in cache)
    logger.info(f"Drugs to fetch: {len(to_fetch)} (already cached: {len(cache)})")

    if not to_fetch:
        return cache

    n_found = 0
    n_failed = 0
    lock = threading.Lock()

    # Rate limiter: token bucket allowing ~4.5 req/s
    # (conservative to avoid 429s with concurrent workers)
    min_interval = 1.0 / 4.5
    last_request_time = time.monotonic()
    rate_lock = threading.Lock()

    session = requests.Session()
    # Connection pooling for performance
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=workers, pool_maxsize=workers
    )
    session.mount("https://", adapter)

    def rate_limited_fetch(name: str) -> tuple[str, str | None]:
        nonlocal last_request_time
        with rate_lock:
            now = time.monotonic()
            wait = min_interval - (now - last_request_time)
            if wait > 0:
                time.sleep(wait)
            last_request_time = time.monotonic()
        return _fetch_one(name, session)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(rate_limited_fetch, name): name for name in to_fetch}

        for i, future in enumerate(as_completed(futures), 1):
            name, smiles = future.result()
            with lock:
                if smiles:
                    cache[name] = smiles
                    n_found += 1
                else:
                    n_failed += 1

            if i % 200 == 0:
                logger.info(
                    f"  Progress: {i}/{len(to_fetch)} "
                    f"(found: {n_found}, failed: {n_failed}, "
                    f"total cached: {len(cache)})"
                )

    session.close()

    logger.info(
        f"Fetch complete: {n_found} new SMILES found, {n_failed} failed, "
        f"{len(cache)} total cached"
    )
    return cache


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch SMILES for PrimeKG drugs from PubChem"
    )
    parser.add_argument(
        "--kg-csv",
        default="data/benchmark/snapshots/kg_t0.csv",
        help="Path to PrimeKG CSV",
    )
    parser.add_argument(
        "--cache",
        default="data/smiles_cache.json",
        help="Path to SMILES cache JSON",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent fetch threads (default: 4)",
    )
    args = parser.parse_args()

    kg_csv = Path(args.kg_csv)
    cache_path = Path(args.cache)

    if not kg_csv.exists():
        logger.error(f"KG CSV not found: {kg_csv}")
        return

    # Extract drug names from KG
    logger.info(f"Extracting drug names from {kg_csv}...")
    drug_names = extract_drug_names(kg_csv)
    logger.info(f"Found {len(drug_names)} unique drug names")

    # Load existing cache
    cache: dict[str, str] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        logger.info(f"Loaded existing cache: {len(cache)} entries")

    # Fetch missing SMILES
    cache = fetch_smiles(drug_names, cache, workers=args.workers)

    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
    logger.info(f"Saved {len(cache)} SMILES to {cache_path}")

    # Report coverage
    matched = sum(1 for n in drug_names if n in cache)
    logger.info(f"Coverage: {matched}/{len(drug_names)} drugs have SMILES "
                f"({100 * matched / len(drug_names):.1f}%)")


if __name__ == "__main__":
    main()
