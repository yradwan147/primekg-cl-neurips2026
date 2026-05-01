"""Maximally Interfered Retrieval (MIR) replay for continual KGE.

MIR (Aljundi et al., NeurIPS 2019) selects replay samples that would be
most negatively affected (highest loss increase) by the current parameter
update. This is a smarter replay selection than random or relation-balanced.

The key idea: after a virtual update step on the current task's batch,
select buffer samples whose loss increased the most — those are the ones
most at risk of being forgotten.

Usage:
    from src.baselines.mir_replay import MIRReplayTrainer
    trainer = MIRReplayTrainer(model_name="DistMult", buffer_size=1000)
    R = trainer.train(task_seq, entity_to_id, relation_to_id)
"""

from __future__ import annotations

import logging
from collections import OrderedDict

import numpy as np
import torch

from src.baselines._base import (
    create_model,
    evaluate_link_prediction,
    get_device,
    make_triples_factory,
    _generate_negatives,
    _margin_loss,
    _score_triples,
)

logger = logging.getLogger(__name__)


class MIRReplayTrainer:
    """MIR replay trainer for continual KGE.

    Maintains a buffer of exemplars. At each training step, performs a
    virtual update, then selects the buffer samples most interfered by
    the update for replay.

    Args:
        model_name: KGE model type.
        embedding_dim: Entity/relation embedding dimension.
        num_epochs: Training epochs per task.
        lr: Learning rate.
        buffer_size: Total buffer size across all tasks.
        mir_candidates: Number of candidate samples to evaluate for MIR selection.
        mir_select: Number of samples to actually replay per batch.
        batch_size: Training batch size.
        device: Torch device.
        seed: Random seed.
    """

    def __init__(
        self,
        model_name: str = "TransE",
        embedding_dim: int = 256,
        num_epochs: int = 100,
        lr: float = 0.001,
        buffer_size: int = 1000,
        mir_candidates: int = 200,
        mir_select: int = 50,
        batch_size: int = 256,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.num_epochs = num_epochs
        self.lr = lr
        self.buffer_size = buffer_size
        self.mir_candidates = mir_candidates
        self.mir_select = mir_select
        self.batch_size = batch_size
        self.device = get_device(device)
        self.seed = seed
        self.model = None

        # Buffer: list of numpy arrays (one per completed task)
        self._buffer_triples: list[np.ndarray] = []

    def _add_to_buffer(self, triples: np.ndarray) -> None:
        """Add exemplars from completed task to buffer using random selection."""
        n_tasks = len(self._buffer_triples) + 1
        per_task = max(1, self.buffer_size // n_tasks)

        # Subsample new task
        if len(triples) > per_task:
            idx = np.random.choice(len(triples), per_task, replace=False)
            self._buffer_triples.append(triples[idx])
        else:
            self._buffer_triples.append(triples.copy())

        # Trim older tasks if needed
        total = sum(len(b) for b in self._buffer_triples)
        while total > self.buffer_size and len(self._buffer_triples) > 1:
            # Remove from largest buffer
            sizes = [len(b) for b in self._buffer_triples]
            largest_idx = int(np.argmax(sizes))
            old = self._buffer_triples[largest_idx]
            keep = max(1, len(old) - (total - self.buffer_size))
            idx = np.random.choice(len(old), keep, replace=False)
            self._buffer_triples[largest_idx] = old[idx]
            total = sum(len(b) for b in self._buffer_triples)

    def _get_buffer(self) -> np.ndarray | None:
        """Get concatenated buffer triples."""
        if not self._buffer_triples:
            return None
        return np.concatenate(self._buffer_triples, axis=0)

    def train(
        self,
        task_sequence: OrderedDict[str, dict],
        entity_to_id: dict[str, int],
        relation_to_id: dict[str, int],
    ) -> np.ndarray:
        """Train with MIR replay.

        Returns:
            Results matrix R[i][j] = MRR on task j after training on task i.
        """
        task_names = list(task_sequence.keys())
        n_tasks = len(task_names)

        task_factories = {}
        for name, data in task_sequence.items():
            task_factories[name] = {
                split: make_triples_factory(arr, entity_to_id, relation_to_id)
                for split, arr in data.items()
                if len(arr) > 0
            }

        first_tf = task_factories[task_names[0]]["train"]
        model = create_model(
            self.model_name, first_tf,
            embedding_dim=self.embedding_dim,
            random_seed=self.seed,
        )
        model = model.to(self.device)
        self.model = model
        np.random.seed(self.seed)

        R = np.zeros((n_tasks, n_tasks))

        for i, task_name in enumerate(task_names):
            logger.info(f"=== MIR Replay: Training on task {i}: {task_name} ===")

            train_tf = task_factories[task_name]["train"]
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            buffer = self._get_buffer()

            from pykeen.models import TransE as TransEModel
            is_transe = isinstance(model, TransEModel)
            model.train()
            mapped_cpu = train_tf.mapped_triples

            for epoch in range(self.num_epochs):
                perm = torch.randperm(len(mapped_cpu))
                shuffled = mapped_cpu[perm]
                epoch_loss = 0.0
                n_batches = 0

                for start in range(0, len(shuffled), self.batch_size):
                    batch = shuffled[start:start + self.batch_size].to(self.device)
                    neg = _generate_negatives(batch, model.num_entities, self.device)

                    pos_scores = _score_triples(model, batch, is_transe)
                    neg_scores = _score_triples(model, neg, is_transe)
                    base_loss = _margin_loss(pos_scores, neg_scores)

                    # MIR replay: select most interfered samples
                    if buffer is not None and len(buffer) > 0:
                        # Virtual update: save state, do one SGD step, select, restore
                        saved_state = {
                            k: v.clone() for k, v in model.state_dict().items()
                        }

                        # Score buffer candidates BEFORE the virtual update
                        n_cand = min(self.mir_candidates, len(buffer))
                        cand_idx = np.random.choice(len(buffer), n_cand, replace=False)
                        candidates = torch.tensor(
                            buffer[cand_idx], dtype=torch.long, device=self.device,
                        )
                        with torch.no_grad():
                            cand_neg = _generate_negatives(
                                candidates, model.num_entities, self.device,
                            )
                            pos_before = _score_triples(model, candidates, is_transe)
                            neg_before = _score_triples(model, cand_neg, is_transe)
                            loss_before = torch.relu(
                                1.0 - pos_before + neg_before
                            ).squeeze()

                        # Do virtual SGD step on current batch
                        optimizer.zero_grad()
                        base_loss.backward()
                        optimizer.step()

                        # Score same candidates AFTER virtual update
                        with torch.no_grad():
                            pos_after = _score_triples(model, candidates, is_transe)
                            neg_after = _score_triples(model, cand_neg, is_transe)
                            loss_after = torch.relu(
                                1.0 - pos_after + neg_after
                            ).squeeze()

                        # Select most interfered
                        interference = loss_after - loss_before
                        n_select = min(self.mir_select, len(candidates))
                        _, top_idx = torch.topk(interference, n_select)
                        replay_batch = candidates[top_idx]

                        # Restore model to pre-update state
                        model.load_state_dict(saved_state)

                        # Recompute base loss + replay loss on restored model
                        pos_scores = _score_triples(model, batch, is_transe)
                        neg_scores = _score_triples(model, neg, is_transe)
                        base_loss = _margin_loss(pos_scores, neg_scores)

                        replay_neg = _generate_negatives(
                            replay_batch, model.num_entities, self.device,
                        )
                        replay_pos = _score_triples(model, replay_batch, is_transe)
                        replay_neg_scores = _score_triples(model, replay_neg, is_transe)
                        replay_loss = _margin_loss(replay_pos, replay_neg_scores)

                        total_loss = base_loss + replay_loss
                    else:
                        total_loss = base_loss

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    epoch_loss += base_loss.item()
                    n_batches += 1

                if epoch % max(1, self.num_epochs // 5) == 0:
                    avg_loss = epoch_loss / max(n_batches, 1)
                    logger.info(f"  Epoch {epoch}: loss={avg_loss:.4f}")

            # Add current task to buffer
            train_data = task_sequence[task_name]["train"]
            self._add_to_buffer(train_data)

            # Evaluate
            model.eval()
            all_known = torch.cat([
                torch.cat([
                    task_factories[task_names[k]][split].mapped_triples
                    for split in ("train", "val", "test")
                    if split in task_factories[task_names[k]]
                ])
                for k in range(i + 1)
            ])

            for j in range(i + 1):
                test_tf = task_factories[task_names[j]]["test"]
                metrics = evaluate_link_prediction(
                    model, test_tf,
                    device=self.device, batch_size=self.batch_size,
                    all_known_mapped_triples=all_known,
                )
                R[i, j] = metrics["MRR"]
                logger.info(f"  Task {j} ({task_names[j]}): MRR={metrics['MRR']:.4f}")

        return R
