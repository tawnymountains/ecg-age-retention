"""Auxiliary age supervision (Section II-C, eqs. (9)-(10) of the paper).

An age head ``h_phi`` maps an encoder representation ``z`` to a predicted age
``a_hat = h_phi(z)``. For a training batch ``B`` the age loss is the batch mean
of the Huber loss with threshold ``kappa`` (eq. (9)),

    ``L_age = (1 / |B|) * sum_{i in B} Huber_kappa(a_hat_i - a_i)``,

and it is added to the original training objective of the encoder (eq. (10)),

    ``L = L_task + lambda_age * L_age``.

For objectives with two views, the age head is applied to the representation
of each view and the age losses are averaged over the views. Ages are
in years, and ``Huber_kappa(d) = 0.5 * d**2`` for ``|d| <= kappa`` and
``kappa * (|d| - kappa / 2)`` otherwise (``torch.nn.functional.huber_loss``).

The age head is used only during training. After training it is discarded and
only the encoder is kept, so computing a representation never requires age.

With the objectives of :mod:`ecg_age_retention.models`::

    encoder = ResNet1dEncoder()
    model = AgeSupervisedObjective(ContrastiveObjective(encoder), AgeHead(hidden_dim=256),
                                   lambda_age=lambda_age)
    out = model(view1, view2, age=age)
    out.total.backward()

With any other encoder and task loss::

    head = AgeHead(hidden_dim=256)
    z = encoder(x)
    loss = task_loss + age_supervision_loss([head(z).squeeze(-1)], [age], lambda_age)
"""
from __future__ import annotations

from typing import List, NamedTuple, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["AgeHead", "age_supervision_loss", "AgeSupervisedOutput", "AgeSupervisedObjective"]


class AgeHead(nn.Sequential):
    """Age-prediction head ``Linear(in_dim, hidden_dim) -> ReLU -> Dropout -> Linear(hidden_dim, 1)``.

    The output has shape ``(batch, 1)``; squeeze the last dimension to obtain
    one predicted age per ECG.

    Args:
        hidden_dim: Hidden width. Required; no default is provided.
        in_dim: Representation dimension.
        dropout: Dropout probability.
    """

    def __init__(self, *, hidden_dim: int, in_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )


def age_supervision_loss(
    predictions: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
    lambda_age: float,
    kappa: float = 5.0,
) -> torch.Tensor:
    """Auxiliary age term ``lambda_age * L_age`` of eq. (10), averaged over views.

    For each view ``v``, the per-ECG Huber losses are multiplied by
    ``lambda_age`` and averaged over the batch; the result is averaged over the
    views.

    Args:
        predictions: Predicted ages, one tensor of shape ``(B,)`` per view.
        targets: Recorded ages in years, one float tensor of shape ``(B,)`` per
            view (at least as many as ``predictions``).
        lambda_age: Multiplier ``lambda_age`` of the age loss in eq. (10).
            Required; no default is provided.
        kappa: Huber threshold in years.
    """
    if len(targets) < len(predictions):
        raise ValueError("one age target is required for every prediction")
    return sum(
        (F.huber_loss(p.float(), targets[i], delta=kappa, reduction="none") * lambda_age).mean()
        for i, p in enumerate(predictions)
    ) / len(predictions)


class AgeSupervisedOutput(NamedTuple):
    """Output of :class:`AgeSupervisedObjective`.

    Attributes:
        total: ``L_task + lambda_age * L_age``; call ``backward`` on this value.
        task_loss: Original training objective (detached).
        age_loss: ``lambda_age * L_age`` averaged over views (detached).
        age_predictions: Predicted ages, one tensor per view.
    """

    total: torch.Tensor
    task_loss: torch.Tensor
    age_loss: torch.Tensor
    age_predictions: List[torch.Tensor]


class AgeSupervisedObjective(nn.Module):
    """Attach an auxiliary age head to an encoder objective (eqs. (9)-(10)).

    ``objective`` must implement ``encode(*inputs) -> (representations, state)``
    and ``task_loss(state) -> Tensor`` (all objectives in
    :mod:`ecg_age_retention.models` do). The forward pass encodes the inputs,
    applies the age head to every representation, evaluates the original
    objective and returns ``L_task + lambda_age * L_age``.

    After training, keep only the encoder (:attr:`encoder`); the age head is
    discarded.

    Args:
        objective: Objective that owns the encoder.
        age_head: Module mapping a representation to a predicted age, for
            example :class:`AgeHead`.
        lambda_age: Multiplier of the age loss (``lambda_age`` in eq. (10)).
            Required; no default is provided.
        kappa: Huber threshold in years (``kappa`` in eq. (9)).
    """

    def __init__(self, objective: nn.Module, age_head: nn.Module, lambda_age: float, kappa: float = 5.0) -> None:
        super().__init__()
        self.objective = objective
        self.age_head = age_head
        self.lambda_age = float(lambda_age)
        self.kappa = kappa

    @property
    def encoder(self) -> nn.Module:
        """The encoder that is kept after training."""
        return self.objective.enc

    def forward(self, *inputs: torch.Tensor, age: torch.Tensor, age2: Optional[torch.Tensor] = None) -> AgeSupervisedOutput:
        """Evaluate the joint objective.

        Args:
            *inputs: Inputs of ``objective`` (for example the two views).
            age: Ages in years for the first (or only) view, shape ``(B,)``.
            age2: Ages for the second view when they differ from ``age``.
        """
        representations, state = self.objective.encode(*inputs)
        predictions = [self.age_head(z).squeeze(-1) for z in representations]
        task = self.objective.task_loss(state)
        targets = [age, age if age2 is None else age2]
        aux = age_supervision_loss(predictions, targets, self.lambda_age, self.kappa)
        total = task + aux
        return AgeSupervisedOutput(total, task.detach(), aux.detach(), predictions)
