"""Smoke test for new CL baselines (SI, Distillation, MIR).

Runs each method on 2 tiny tasks with minimal settings to verify
the training loop, evaluation, and results matrix work end-to-end.

Usage (on IBEX or any machine with PyKEEN):
    python scripts/smoke_test_new_baselines.py
"""

from __future__ import annotations

import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_fake_task_sequence() -> tuple:
    """Create a minimal 2-task sequence with fake triples."""
    np.random.seed(42)
    n_entities = 100
    n_relations = 5

    entity_to_id = {f"e{i}": i for i in range(n_entities)}
    relation_to_id = {f"r{i}": i for i in range(n_relations)}

    def random_triples(n: int) -> np.ndarray:
        h = np.random.randint(0, n_entities, n)
        r = np.random.randint(0, n_relations, n)
        t = np.random.randint(0, n_entities, n)
        return np.stack([h, r, t], axis=1).astype(np.int64)

    task_seq = OrderedDict()
    task_seq["task_0"] = {
        "train": random_triples(200),
        "val": random_triples(20),
        "test": random_triples(50),
    }
    task_seq["task_1"] = {
        "train": random_triples(150),
        "val": random_triples(15),
        "test": random_triples(40),
    }

    return task_seq, entity_to_id, relation_to_id


def test_baseline(name: str, trainer_cls, **kwargs) -> None:
    """Run a single baseline smoke test."""
    print(f"\n{'='*50}")
    print(f"Testing: {name}")
    print(f"{'='*50}")

    task_seq, e2id, r2id = make_fake_task_sequence()

    trainer = trainer_cls(
        model_name="DistMult",
        embedding_dim=32,
        num_epochs=3,
        lr=0.01,
        batch_size=64,
        device="auto",
        seed=42,
        **kwargs,
    )

    start = time.time()
    R = trainer.train(task_seq, e2id, r2id)
    elapsed = time.time() - start

    print(f"\nResults matrix shape: {R.shape}")
    print(f"R[0,0] (task 0 after task 0): {R[0,0]:.4f}")
    print(f"R[1,0] (task 0 after task 1): {R[1,0]:.4f}")
    print(f"R[1,1] (task 1 after task 1): {R[1,1]:.4f}")
    print(f"Time: {elapsed:.1f}s")

    # Basic sanity checks
    assert R.shape == (2, 2), f"Wrong shape: {R.shape}"
    assert R[0, 0] >= 0, "Negative MRR"
    assert R[1, 1] >= 0, "Negative MRR"
    assert R[0, 1] == 0, "Upper triangle should be 0"
    assert trainer.model is not None, "Model not saved"

    print(f"PASSED: {name}")


def main() -> None:
    print("Smoke testing new CL baselines...")
    print(f"Python: {sys.version}")

    # Test SI
    from src.baselines.synaptic_intelligence import SITrainer
    test_baseline("Synaptic Intelligence", SITrainer, lambda_si=1.0, damping=0.1)

    # Test Distillation
    from src.baselines.knowledge_distillation import DistillationTrainer
    test_baseline("Knowledge Distillation", DistillationTrainer, lambda_distill=5.0)

    # Test MIR
    from src.baselines.mir_replay import MIRReplayTrainer
    test_baseline("MIR Replay", MIRReplayTrainer,
                  buffer_size=100, mir_candidates=50, mir_select=10)

    # Also test existing baselines with DistMult (new decoder)
    from src.baselines.naive_sequential import NaiveSequentialTrainer
    test_baseline("Naive Sequential (DistMult)", NaiveSequentialTrainer)

    from src.baselines.ewc import EWCTrainer
    test_baseline("EWC (DistMult)", EWCTrainer, lambda_ewc=10.0, fisher_samples=50)

    print(f"\n{'='*50}")
    print("ALL SMOKE TESTS PASSED")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
