"""Cox partial-likelihood batch loss of the survival-supervised encoder.

The single-task supervised encoder is trained with a survival objective: an
MLP risk head on the encoder representation, scored with the negative Cox
partial log-likelihood of the mini-batch (:func:`cox_ph_loss`).
:class:`CoxObjective` packages encoder, risk head and loss behind the same
``encode`` / ``task_loss`` interface as the self-supervised objectives, so that
an auxiliary age head can be attached with
:class:`ecg_age_retention.supervision.AgeSupervisedObjective`.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn

__all__ = ["cox_ph_loss", "survival_head", "CoxObjective"]


def cox_ph_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Negative Cox partial log-likelihood of a mini-batch.

    The batch is sorted by ``time`` in descending order with
    ``torch.argsort(time, descending=True)``. After sorting, the risk set of
    the sample at position ``k`` is the set of samples at positions ``0..k``, and
    the loss is

        ``-sum_k event_k * (risk_k - logsumexp(risk_0..risk_k)) / (sum(event) + eps)``.

    Tied times receive no special treatment: among samples with equal times,
    each risk set contains only the tied samples placed at or before it by the
    sort, so the result for tied times depends on that order (this is not the
    Breslow or Efron tie correction). If the batch contains no event the
    function returns ``risk.sum() * 0.0``, a zero that keeps the computation
    graph.

    Args:
        risk: Predicted log-hazard (linear risk score), shape ``(B,)``.
        time: Follow-up time, shape ``(B,)``.
        event: Event indicator (1 = event, 0 = censored), shape ``(B,)``,
            in the floating dtype of ``risk``.
        eps: Constant added to the number of events in the denominator.
    """
    if event.sum() < 1:
        return risk.sum() * 0.0
    idx = torch.argsort(time, descending=True)
    risk, event = risk[idx], event[idx]
    log_cum = torch.logcumsumexp(risk, dim=0)
    return -(((risk - log_cum) * event).sum()) / (event.sum() + eps)


def survival_head(in_dim: int = 256, hidden_dim: int = 128, dropout: float = 0.2) -> nn.Sequential:
    """Risk head ``Linear -> ReLU -> Dropout -> Linear`` (256-128-1) producing one log-hazard per ECG."""
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))


class CoxObjective(nn.Module):
    """Survival objective of the single-task supervised encoder.

    ``task_loss`` applies :func:`survival_head` to the representation and
    scores the risk with :func:`cox_ph_loss`.

    Args:
        encoder: Encoder mapping ``(batch, 12, T)`` to ``(batch, in_dim)``.
        in_dim: Representation dimension.
        hidden_dim: Hidden width of the risk head.
        dropout: Dropout probability of the risk head.
    """

    def __init__(self, encoder: nn.Module, in_dim: int = 256, hidden_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.enc = encoder
        self.head = survival_head(in_dim, hidden_dim, dropout)

    def encode(self, x: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> Tuple[List[torch.Tensor], tuple]:
        """Encode the ECGs; returns ``([z], state)``.

        Args:
            x: ECGs, shape ``(B, 12, T)``.
            time: Follow-up time, shape ``(B,)``.
            event: Event indicator, shape ``(B,)``.
        """
        z = self.enc(x)
        return [z], (z, time, event)

    def task_loss(self, state: tuple) -> torch.Tensor:
        """Cox partial-likelihood loss of the risk head output."""
        z, time, event = state
        risk = self.head(z).squeeze(-1)
        return cox_ph_loss(risk.float(), time, event)

    def forward(self, x: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Return ``(task_loss, [z])``."""
        representations, state = self.encode(x, time, event)
        return self.task_loss(state), representations
