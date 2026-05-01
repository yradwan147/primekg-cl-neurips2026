"""Baseline 5: LKGE (Lifelong Knowledge Graph Embedding) wrapper.

Wraps the external LKGE framework (https://github.com/nju-websoft/LKGE)
for use with our PrimeKG temporal benchmark. Handles data format conversion
between our benchmark format and LKGE's expected directory structure,
subprocess execution, and result parsing.

Usage:
    from src.baselines.lkge import LKGEWrapper
    wrapper = LKGEWrapper(lkge_dir='external/LKGE')
    wrapper.convert_to_lkge_format(task_sequence, 'data/lkge_format')
    results = wrapper.run_and_parse(dataset_dir, output_dir)
"""

from __future__ import annotations

import logging
import re
import subprocess
from collections import OrderedDict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class LKGEWrapper:
    """Wrapper for the LKGE external framework.

    Args:
        lkge_dir: Path to cloned LKGE repository.
        gpu_id: GPU device ID for LKGE.
    """

    def __init__(
        self,
        lkge_dir: str = "external/LKGE",
        gpu_id: int = 0,
    ) -> None:
        self.lkge_dir = Path(lkge_dir)
        self.gpu_id = gpu_id

    def convert_to_lkge_format(
        self,
        task_sequence: OrderedDict[str, dict[str, np.ndarray]],
        output_dir: str,
    ) -> str:
        """Convert task sequence to LKGE's expected directory format.

        Creates snapshot_0/, snapshot_1/, etc. with train.txt, valid.txt,
        test.txt in each. Format: head_entity<tab>relation<tab>tail_entity.

        Also creates entity2id.txt and relation2id.txt at the top level.

        Args:
            task_sequence: Our benchmark task sequence.
            output_dir: Directory to write LKGE-formatted data.

        Returns:
            Path to the converted dataset directory.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Collect all entities and relations
        entities = set()
        relations = set()
        for data in task_sequence.values():
            for split in data.values():
                if len(split) > 0:
                    entities.update(split[:, 0])
                    entities.update(split[:, 2])
                    relations.update(split[:, 1])

        entity_list = sorted(entities)
        relation_list = sorted(relations)

        # Write entity2id.txt
        with open(out / "entity2id.txt", "w") as f:
            f.write(f"{len(entity_list)}\n")
            for i, e in enumerate(entity_list):
                f.write(f"{e}\t{i}\n")

        # Write relation2id.txt
        with open(out / "relation2id.txt", "w") as f:
            f.write(f"{len(relation_list)}\n")
            for i, r in enumerate(relation_list):
                f.write(f"{r}\t{i}\n")

        # Write each snapshot — LKGE expects dirs named 0/, 1/, 2/, ...
        for idx, (name, data) in enumerate(task_sequence.items()):
            snap_dir = out / str(idx)
            snap_dir.mkdir(exist_ok=True)

            for split_name, split_data in data.items():
                fname = {"train": "train.txt", "val": "valid.txt", "test": "test.txt"}
                if split_name in fname and len(split_data) > 0:
                    with open(snap_dir / fname[split_name], "w") as f:
                        for h, r, t in split_data:
                            f.write(f"{h}\t{r}\t{t}\n")

            logger.info(f"  Wrote {idx}/ ({name}): "
                       f"{len(data.get('train', [])):,} train triples")

        logger.info(f"LKGE dataset written to {out} "
                    f"({len(task_sequence)} snapshots, "
                    f"{len(entity_list)} entities, "
                    f"{len(relation_list)} relations)")
        return str(out)

    def get_run_command(
        self,
        dataset_dir: str,
        lifelong_name: str = "LKGE",
        model: str = "TransE",
        num_epochs: int = 100,
        snapshot_num: int | None = None,
        seed: int = 42,
        emb_dim: int = 200,
        batch_size: int = 2048,
    ) -> str:
        """Generate the command to run LKGE.

        Args:
            dataset_dir: Path to LKGE-formatted dataset.
            lifelong_name: LKGE method variant.
            model: KGE model to use.
            num_epochs: Training epochs.
            snapshot_num: Number of snapshots. Auto-detected if None.
            seed: Random seed.
            emb_dim: Embedding dimension (default 200; reduce for large KGs).
            batch_size: Training batch size.

        Returns:
            Command string to execute.
        """
        # LKGE expects: -data_path <parent>/ -dataset <folder_name>
        # It constructs paths as: data_path + dataset + '/'
        abs_dataset_dir = Path(dataset_dir).resolve()
        data_path = str(abs_dataset_dir.parent) + "/"
        dataset_name = abs_dataset_dir.name
        # Checkpoint and log paths must also be absolute to avoid
        # writing into the LKGE repo directory
        save_path = str(abs_dataset_dir.parent / "lkge_checkpoints") + "/"
        log_path = str(abs_dataset_dir.parent / "lkge_logs") + "/"
        # Auto-detect snapshot count from dataset directory
        if snapshot_num is None:
            # Dirs are named 0/, 1/, 2/, ... — count numeric directories
            snapshot_num = len([
                d for d in abs_dataset_dir.iterdir()
                if d.is_dir() and d.name.isdigit()
            ])
        cmd = (
            f"python main.py "
            f"-data_path {data_path} "
            f"-dataset {dataset_name} "
            f"-save_path {save_path} "
            f"-log_path {log_path} "
            f"-gpu {self.gpu_id} "
            f"-snapshot_num {snapshot_num} "
            f"-lifelong_name {lifelong_name} "
            f"-embedding_model {model} "
            f"-epoch_num {num_epochs} "
            f"-emb_dim {emb_dim} "
            f"-batch_size {batch_size} "
            f"-seed {seed}"
        )
        return cmd

    def run_and_parse(
        self,
        dataset_dir: str,
        output_dir: str,
        lifelong_name: str = "LKGE",
        model: str = "TransE",
        num_epochs: int = 100,
        timeout: int = 86400,
        seed: int = 42,
        emb_dim: int = 200,
        batch_size: int = 2048,
    ) -> dict:
        """Run LKGE as a subprocess and parse the output.

        Args:
            dataset_dir: Path to LKGE-formatted dataset.
            output_dir: Directory for LKGE outputs.
            lifelong_name: LKGE method variant.
            model: KGE model to use.
            num_epochs: Training epochs.
            timeout: Max runtime in seconds (default 24h).
            seed: Random seed.
            emb_dim: Embedding dimension.
            batch_size: Training batch size.

        Returns:
            Dict with parsed metrics per snapshot.
        """
        cmd = self.get_run_command(
            dataset_dir, lifelong_name, model, num_epochs,
            seed=seed, emb_dim=emb_dim, batch_size=batch_size,
        )
        logger.info(f"Running LKGE: {cmd}")

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        log_file = out_path / "lkge_run.log"

        # Resolve lkge_dir to absolute so cwd works correctly
        abs_lkge_dir = str(self.lkge_dir.resolve())
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=timeout, cwd=abs_lkge_dir,
            )
            output = result.stdout + "\n" + result.stderr

            with open(log_file, "w") as f:
                f.write(output)

            logger.info(f"LKGE completed (returncode={result.returncode})")

            if result.returncode != 0:
                logger.warning(f"LKGE non-zero exit: {result.returncode}")
                logger.warning(f"stderr (last 2000 chars): {result.stderr[-2000:]}")

        except subprocess.TimeoutExpired:
            logger.error(f"LKGE timed out after {timeout}s")
            return {"error": "timeout"}
        except FileNotFoundError:
            logger.error(f"LKGE not found at {self.lkge_dir}")
            return {"error": "lkge_not_found"}

        return self.parse_results(output_dir)

    def parse_results(self, output_dir: str) -> dict:
        """Parse LKGE output files into our metrics format.

        Extracts MRR, Hits@1, Hits@3, Hits@10 per snapshot from LKGE log
        output using regex patterns matching LKGE's standard output format.

        Args:
            output_dir: LKGE output directory.

        Returns:
            Dict with 'per_snapshot' metrics and 'results_matrix'.
        """
        results_path = Path(output_dir)
        if not results_path.exists():
            logger.warning(f"LKGE output not found: {output_dir}")
            return {}

        # Collect all text from log files
        content = ""
        log_files = (list(results_path.glob("*.log"))
                     + list(results_path.glob("*.txt"))
                     + list(results_path.glob("**/*.log")))
        for lf in log_files:
            with open(lf) as f:
                content += f.read() + "\n"

        if not content.strip():
            logger.warning("No LKGE log content found")
            return {"raw_log": "", "per_snapshot": {}}

        return self._parse_log_content(content)

    def _parse_log_content(self, content: str) -> dict:
        """Parse LKGE log text to extract per-snapshot metrics.

        LKGE outputs PrettyTable format with per-snapshot test tables:
            +-----------+------+--------+--------+--------+---------+
            | Snapshot:0| MRR  | Hits@1 | Hits@3 | Hits@5 | Hits@10 |
            +-----------+------+--------+--------+--------+---------+
            |     0     | 0.45 | 0.23   | 0.34   | 0.40   | 0.52    |
            +-----------+------+--------+--------+--------+---------+

        And a report table:
            +----------+------+-----------+--------------+--------------+---------------+
            | Snapshot | Time | Whole_MRR | Whole_Hits@1 | Whole_Hits@3 | Whole_Hits@10 |
            +----------+------+-----------+--------------+--------------+---------------+
            |    0     | 12.3 |   0.456   |    0.234     |    0.345     |    0.523      |
            +----------+------+-----------+--------------+--------------+---------------+

        And training logs:
            Snapshot:X\tEpoch:X\tLoss:X\tMRR:XX.XX\tHits@10:XX.XX\tBest:XX.XX

        Args:
            content: Raw log text.

        Returns:
            Dict with per_snapshot metrics and results_matrix.
        """
        # results_matrix[train_snap][test_snap] = MRR
        matrix_entries: dict[tuple[int, int], float] = {}
        # Per-snapshot final metrics (from report table)
        per_snapshot: dict[int, dict[str, float]] = {}

        lines = content.split("\n")

        # --- Parse per-snapshot test tables ---
        # Header pattern: "Snapshot:X" in the table header row
        header_pat = re.compile(r"Snapshot:(\d+)")
        # Data row: | test_id | mrr | hits1 | hits3 | hits5 | hits10 |
        data_row_pat = re.compile(
            r"\|\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|"
            r"\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|"
        )

        current_train_snap = None
        for line in lines:
            # Stop before Report Result / Final Result summary tables
            # whose data rows have a different column layout (Time column)
            # that would corrupt matrix_entries with non-MRR values
            if "Report Result" in line:
                break

            # Check for table header identifying the training snapshot
            hm = header_pat.search(line)
            if hm and "field_names" not in line:
                # Could be header row or just a mention — check it's in a table
                if "|" in line or "Snapshot:" in line:
                    current_train_snap = int(hm.group(1))
                continue

            # Check for data rows
            dm = data_row_pat.search(line)
            if dm and current_train_snap is not None:
                test_snap = int(dm.group(1))
                mrr = float(dm.group(2))
                # Sanity check: MRR should be in [0, 1]
                if mrr <= 1.0:
                    matrix_entries[(current_train_snap, test_snap)] = mrr

            # Reset on separator lines between tables
            if line.strip().startswith("+") and line.count("+") >= 3:
                pass  # PrettyTable border — don't reset

        # --- Parse report table (whole metrics per snapshot) ---
        report_row_pat = re.compile(
            r"\|\s*(\d+)\s*\|\s*[0-9.]+\s*\|\s*([0-9.]+)\s*\|"
            r"\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|"
        )
        in_report = False
        for line in lines:
            if "Whole_MRR" in line:
                in_report = True
                continue
            if in_report:
                rm = report_row_pat.search(line)
                if rm:
                    snap_id = int(rm.group(1))
                    per_snapshot[snap_id] = {
                        "Whole_MRR": float(rm.group(2)),
                        "Whole_Hits@1": float(rm.group(3)),
                        "Whole_Hits@3": float(rm.group(4)),
                        "Whole_Hits@10": float(rm.group(5)),
                    }
                elif line.strip().startswith("+"):
                    pass  # Table border
                elif line.strip() and not line.strip().startswith("|"):
                    in_report = False

        # --- Parse training log lines as fallback ---
        # Format: Snapshot:X\tEpoch:X\tLoss:X\tMRR:XX.XX\tHits@10:XX.XX\tBest:XX.XX
        train_log_pat = re.compile(
            r"Snapshot:(\d+)\s+Epoch:(\d+)\s+Loss:([0-9.]+)\s+"
            r"MRR:([0-9.]+)\s+Hits@10:([0-9.]+)"
        )
        training_logs = []
        for line in lines:
            tm = train_log_pat.search(line)
            if tm:
                training_logs.append({
                    "snapshot": int(tm.group(1)),
                    "epoch": int(tm.group(2)),
                    "loss": float(tm.group(3)),
                    "mrr_pct": float(tm.group(4)),
                    "hits10_pct": float(tm.group(5)),
                })

        # --- Parse forward/backward transfer ---
        transfer = {}
        ft_pat = re.compile(
            r"Forward transfer:\s*([0-9.eE+-]+)\s+"
            r"Backward transfer:\s*([0-9.eE+-]+)"
        )
        fm = ft_pat.search(content)
        if fm:
            transfer["forward_transfer"] = float(fm.group(1))
            transfer["backward_transfer"] = float(fm.group(2))

        # --- Build results matrix ---
        if matrix_entries:
            max_snap = max(
                max(k[0] for k in matrix_entries),
                max(k[1] for k in matrix_entries),
            )
            n = max_snap + 1
            results_matrix = np.zeros((n, n))
            for (train_s, test_s), mrr in matrix_entries.items():
                results_matrix[train_s, test_s] = mrr
            results_matrix = results_matrix.tolist()
        elif per_snapshot:
            # Fallback: use report table (whole metrics)
            n = max(per_snapshot.keys()) + 1
            results_matrix = np.zeros((n, n))
            for snap_id, metrics in per_snapshot.items():
                # Report table only has final row — fill last row
                results_matrix[n - 1, snap_id] = metrics.get("Whole_MRR", 0.0)
            results_matrix = results_matrix.tolist()
        else:
            results_matrix = []

        return {
            "per_snapshot": {
                str(k): v for k, v in sorted(per_snapshot.items())
            },
            "matrix_entries": {
                f"{k[0]}_{k[1]}": v for k, v in sorted(matrix_entries.items())
            },
            "results_matrix": results_matrix,
            "transfer": transfer,
            "num_training_logs": len(training_logs),
            "raw_log": content[:10000],
        }
