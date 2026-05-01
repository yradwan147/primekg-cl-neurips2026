"""Knowledge Distillation baseline for continual KGE.

Maintains a frozen copy of the model from the previous task and penalizes
embedding drift via L2 distance between current and old entity embeddings.
This is the distillation component used in LKGE (Cui et al., AAAI 2023)
isolated as a standalone baseline.

Usage:
    from src.baselines.knowledge_distillation import DistillationTrainer
    trainer = DistillationTrainer(model_name="DistMult", lambda_distill=5.0)
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


class DistillationTrainer:
    """Knowledge distillation trainer for continual KGE.

    After each task, saves a frozen copy of the model. During training on
    the next task, adds an L2 penalty on entity embedding drift from the
    frozen model, preserving learned representations.

    Args:
        model_name: KGE model type.
        embedding_dim: Entity/relation embedding dimension.
        num_epochs: Training epochs per task.
        lr: Learning rate.
        lambda_distill: Distillation loss weight.
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
        lambda_distill: float = 5.0,
        batch_size: int = 256,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.num_epochs = num_epochs
        self.lr = lr
        self.lambda_distill = lambda_distill
        self.batch_size = batch_size
        self.device = get_device(device)
        self.seed = seed
        self.model = None

    def _distillation_loss(
        self, model: torch.nn.Module,
        old_state: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """L2 distance between current and saved old parameter values."""
        loss = torch.tensor(0.0, device=self.device)
        for name, param in model.named_parameters():
            if name in old_state:
                loss = loss + ((param - old_state[name]) ** 2).sum()
        return self.lambda_distill * loss

    def train(
        self,
        task_sequence: OrderedDict[str, dict],
        entity_to_id: dict[str, int],
        relation_to_id: dict[str, int],
    ) -> np.ndarray:
        """Train sequentially with knowledge distillation.

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

        old_state: dict[str, torch.Tensor] | None = None
        R = np.zeros((n_tasks, n_tasks))

        for i, task_name in enumerate(task_names):
            logger.info(f"=== Distillation: Training on task {i}: {task_name} ===")

            train_tf = task_factories[task_name]["train"]
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

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

                    # Add distillation penalty if we have saved old params
                    if old_state is not None:
                        total_loss = base_loss + self._distillation_loss(
                            model, old_state,
                        )
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

            # Save parameter snapshot for next task's distillation
            old_state = {
                name: param.data.clone().detach()
                for name, param in model.named_parameters()
            }

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
