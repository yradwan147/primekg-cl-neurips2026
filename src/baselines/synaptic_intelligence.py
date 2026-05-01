"""Synaptic Intelligence (SI) baseline for continual KGE.

SI (Zenke et al., ICML 2017) accumulates per-parameter importance online
during training, then penalizes changes to important parameters when learning
new tasks. Unlike EWC which computes Fisher after each task, SI tracks
importance continuously via the running product of gradient and parameter change.

Usage:
    from src.baselines.synaptic_intelligence import SITrainer
    trainer = SITrainer(model_name="DistMult", lambda_si=1.0)
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


class SynapticIntelligenceKGE:
    """Online importance accumulation for continual KGE.

    Tracks the contribution of each parameter to loss reduction during training,
    then uses these accumulated importances as regularization weights.

    Args:
        model: PyKEEN model to track.
        lambda_si: Regularization strength.
        damping: Small constant to prevent division by zero in importance normalization.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        lambda_si: float = 1.0,
        damping: float = 0.1,
    ) -> None:
        self.model = model
        self.lambda_si = lambda_si
        self.damping = damping

        # Accumulated importance across tasks
        self.omega: dict[str, torch.Tensor] = {}
        # Reference parameters (snapshot after each task)
        self.old_params: dict[str, torch.Tensor] = {}
        # Running importance accumulator for current task
        self.running_omega: dict[str, torch.Tensor] = {}
        # Parameter snapshot at start of current task
        self.task_start_params: dict[str, torch.Tensor] = {}

    def begin_task(self) -> None:
        """Call at the start of each task to begin tracking importance."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.task_start_params[name] = param.data.clone()
                self.running_omega[name] = torch.zeros_like(param.data)

    def update_running_omega(self) -> None:
        """Call after each optimizer step to accumulate importance.

        Records gradient * negative parameter change as importance signal.
        This captures how much each parameter contributed to loss reduction.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Importance = -gradient * (param_change)
                # Since we're doing gradient descent, param_change = -lr * grad
                # So importance ≈ lr * grad^2 (positive, measures contribution)
                self.running_omega[name] += (
                    param.grad.data ** 2
                )

    def end_task(self) -> None:
        """Call after training on a task to consolidate importance."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Compute parameter change during this task
                delta = param.data - self.task_start_params[name]
                delta_sq = delta ** 2 + self.damping

                # Normalize running importance by parameter change
                task_importance = self.running_omega.get(
                    name, torch.zeros_like(param.data)
                ) / delta_sq

                # Accumulate across tasks
                if name in self.omega:
                    self.omega[name] = self.omega[name] + task_importance
                else:
                    self.omega[name] = task_importance

                # Save reference parameters
                self.old_params[name] = param.data.clone()

    def si_loss(self) -> torch.Tensor:
        """Compute SI regularization penalty.

        Returns:
            Scalar tensor: weighted L2 penalty on parameter changes from reference.
        """
        if not self.omega:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)

        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for name, param in self.model.named_parameters():
            if name in self.omega:
                loss = loss + (
                    self.omega[name] * (param - self.old_params[name]) ** 2
                ).sum()

        return (self.lambda_si / 2.0) * loss


class SITrainer:
    """Synaptic Intelligence trainer for continual KGE.

    Args:
        model_name: KGE model type (TransE, DistMult, etc.).
        embedding_dim: Entity/relation embedding dimension.
        num_epochs: Training epochs per task.
        lr: Learning rate.
        lambda_si: SI regularization strength.
        damping: SI damping constant.
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
        lambda_si: float = 1.0,
        damping: float = 0.1,
        batch_size: int = 256,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.num_epochs = num_epochs
        self.lr = lr
        self.lambda_si = lambda_si
        self.damping = damping
        self.batch_size = batch_size
        self.device = get_device(device)
        self.seed = seed
        self.model = None

    def train(
        self,
        task_sequence: OrderedDict[str, dict],
        entity_to_id: dict[str, int],
        relation_to_id: dict[str, int],
    ) -> np.ndarray:
        """Train sequentially with SI regularization.

        Args:
            task_sequence: Ordered dict of task_name -> {train, val, test} arrays.
            entity_to_id: Entity name to integer mapping.
            relation_to_id: Relation name to integer mapping.

        Returns:
            Results matrix R[i][j] = MRR on task j after training on task i.
        """
        task_names = list(task_sequence.keys())
        n_tasks = len(task_names)

        # Build TriplesFactories
        task_factories = {}
        for name, data in task_sequence.items():
            task_factories[name] = {
                split: make_triples_factory(arr, entity_to_id, relation_to_id)
                for split, arr in data.items()
                if len(arr) > 0
            }

        # Initialize model
        first_tf = task_factories[task_names[0]]["train"]
        model = create_model(
            self.model_name, first_tf,
            embedding_dim=self.embedding_dim,
            random_seed=self.seed,
        )
        model = model.to(self.device)
        self.model = model

        # Initialize SI
        si = SynapticIntelligenceKGE(
            model, lambda_si=self.lambda_si, damping=self.damping,
        )

        R = np.zeros((n_tasks, n_tasks))

        for i, task_name in enumerate(task_names):
            logger.info(f"=== SI: Training on task {i}: {task_name} ===")

            train_tf = task_factories[task_name]["train"]
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

            # Begin tracking importance for this task
            si.begin_task()

            # Training loop with SI penalty
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

                    # Add SI penalty
                    total_loss = base_loss + si.si_loss()

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    # Track importance after each step
                    si.update_running_omega()

                    epoch_loss += base_loss.item()
                    n_batches += 1

                if epoch % max(1, self.num_epochs // 5) == 0:
                    avg_loss = epoch_loss / max(n_batches, 1)
                    logger.info(f"  Epoch {epoch}: loss={avg_loss:.4f}")

            # Consolidate importance
            si.end_task()

            # Evaluate on all tasks seen so far
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
