"""Self-supervised objectives of the ECG encoders: heads and losses.

The paper uses six self-supervised encoders: SimCLR and CLOCS-style
(contrastive), BYOL and VICReg (non-contrastive), and SSL-MAE and a denoising
autoencoder (DAE) (reconstruction-based). This module provides their heads and
losses. Every objective takes already-formed inputs:

* two views of shape ``(batch, 12, n_samples)`` for :class:`ContrastiveObjective`
  (SimCLR and CLOCS-style), :class:`BYOLObjective` and :class:`VICRegObjective`;
* a masked input, the clean target and the mask for
  :class:`MaskedReconstructionObjective` (SSL-MAE);
* a noisy input and the clean target for :class:`DenoisingObjective` (DAE).

How the views, masks or noisy inputs are produced is left to the caller.

All objectives share a two-stage interface, which
:class:`ecg_age_retention.supervision.AgeSupervisedObjective` uses to attach
an auxiliary age head:

``encode(*inputs) -> (representations, state)``
    runs the encoder and returns the list of encoder outputs (one per view,
    each of shape ``(batch, 256)``) together with the state needed by the task
    loss;
``task_loss(state) -> torch.Tensor``
    evaluates the original training objective (heads and loss).

Calling an objective returns ``(task_loss, representations)``. The loss
functions cast their inputs to float32.
"""
from __future__ import annotations

import copy
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    "nt_xent_loss",
    "vicreg_loss",
    "byol_loss",
    "masked_reconstruction_loss",
    "reconstruction_mse_loss",
    "projection_head",
    "mlp_bn",
    "vicreg_expander",
    "reconstruction_decoder",
    "ContrastiveObjective",
    "VICRegObjective",
    "BYOLObjective",
    "MaskedReconstructionObjective",
    "DenoisingObjective",
]


# --------------------------------------------------------------------------
# Loss functions
# --------------------------------------------------------------------------
def nt_xent_loss(p1: torch.Tensor, p2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Normalized temperature-scaled cross-entropy (NT-Xent) loss.

    The ``2N`` projections ``cat([p1, p2])`` are L2-normalized and their pairwise
    cosine similarities are divided by ``temperature``. Self-similarities are
    set to ``-1e9``; for each projection the positive is the other view of the
    same sample (row ``i`` of ``p1`` pairs with row ``i`` of ``p2``) and the other
    ``2N - 2`` projections are negatives. Returns the mean cross-entropy over
    the ``2N`` projections.

    Args:
        p1: Projections of the first views, shape ``(N, d)``.
        p2: Projections of the second views, shape ``(N, d)``.
        temperature: Softmax temperature.
    """
    z = F.normalize(torch.cat([p1, p2]).float(), dim=1)
    sim = z @ z.T / temperature
    sim.fill_diagonal_(-1e9)
    n = len(p1)
    targets = (torch.arange(2 * n, device=z.device) + n) % (2 * n)
    return F.cross_entropy(sim, targets)


def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    invariance_weight: float = 25.0,
    variance_weight: float = 25.0,
    covariance_weight: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Variance-invariance-covariance (VICReg) loss of two batches of embeddings.

    * invariance: mean squared error between ``z1`` and ``z2``;
    * variance, per branch: ``mean(relu(1 - sqrt(var(z) + eps)))`` over
      dimensions, with the unbiased variance over the batch;
    * covariance, per branch: sum of squared off-diagonal entries of the
      covariance matrix (normalized by ``N - 1``), divided by the dimension.

    Returns ``invariance_weight * invariance + variance_weight * (v(z1) + v(z2))
    + covariance_weight * c(z1) + covariance_weight * c(z2)``. The batch size
    must be at least 2.
    """
    z1, z2 = z1.float(), z2.float()
    inv = F.mse_loss(z1, z2)

    def variance_term(z: torch.Tensor) -> torch.Tensor:
        return torch.relu(1 - torch.sqrt(z.var(0) + eps)).mean()

    def covariance_term(z: torch.Tensor) -> torch.Tensor:
        z = z - z.mean(0)
        cov = z.T @ z / (len(z) - 1)
        return (cov.square().sum() - cov.diagonal().square().sum()) / z.shape[1]

    return (
        invariance_weight * inv
        + variance_weight * (variance_term(z1) + variance_term(z2))
        + covariance_weight * covariance_term(z1)
        + covariance_weight * covariance_term(z2)
    )


def byol_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """BYOL regression loss ``mean(2 - 2 * cos(p, z))`` over the batch.

    Args:
        p: Online predictions, shape ``(N, d)``.
        z: Target projections of the other view, shape ``(N, d)``.
    """
    return (2 - 2 * (F.normalize(p.float(), dim=1) * F.normalize(z.float(), dim=1)).sum(1)).mean()


def masked_reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    n_leads: int = 12,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Squared reconstruction error averaged over the masked samples (SSL-MAE).

    Returns ``sum((reconstruction - target)**2 * mask) / (sum(mask) * n_leads + eps)``.

    Args:
        reconstruction: Reconstructed signal, shape ``(N, 12, T)``.
        target: Clean signal, shape ``(N, 12, T)``.
        mask: 1 at masked time samples and 0 elsewhere, shape ``(N, 1, T)``;
            the same time samples are masked in all leads.
        n_leads: Number of leads the mask is broadcast over.
        eps: Constant added to the denominator.
    """
    error = (reconstruction - target).square()
    return (error * mask).sum() / (mask.sum() * n_leads + eps)


def reconstruction_mse_loss(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared reconstruction error over all leads and samples (DAE)."""
    return F.mse_loss(reconstruction, target)


# --------------------------------------------------------------------------
# Heads
# --------------------------------------------------------------------------
def projection_head(in_dim: int = 256, hidden_dim: int = 256, out_dim: int = 128) -> nn.Sequential:
    """Projection head of SimCLR and CLOCS-style: ``Linear -> ReLU -> Linear`` (256-256-128)."""
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))


def mlp_bn(in_dim: int = 256, hidden_dim: int = 1024, out_dim: int = 256) -> nn.Sequential:
    """``Linear -> BatchNorm1d -> ReLU -> Linear`` (256-1024-256), used as the BYOL projector and predictor."""
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))


def vicreg_expander(in_dim: int = 256, dim: int = 2048) -> nn.Sequential:
    """VICReg expander: three linear layers of width 2048, with BatchNorm and ReLU after the first two."""
    return nn.Sequential(
        nn.Linear(in_dim, dim),
        nn.BatchNorm1d(dim),
        nn.ReLU(),
        nn.Linear(dim, dim),
        nn.BatchNorm1d(dim),
        nn.ReLU(),
        nn.Linear(dim, dim),
    )


def reconstruction_decoder(in_channels: int = 256, out_channels: int = 12) -> nn.Sequential:
    """Decoder of SSL-MAE and DAE: four ``ConvTranspose1d(kernel_size=4, stride=4)`` layers.

    Channels 256 -> 128 -> 64 -> 32 -> 12 with ReLU between layers. It maps the
    encoder feature map (before pooling) back to a 12-lead signal; each layer
    multiplies the length by 4.
    """
    return nn.Sequential(
        nn.ConvTranspose1d(in_channels, 128, 4, 4),
        nn.ReLU(),
        nn.ConvTranspose1d(128, 64, 4, 4),
        nn.ReLU(),
        nn.ConvTranspose1d(64, 32, 4, 4),
        nn.ReLU(),
        nn.ConvTranspose1d(32, out_channels, 4, 4),
    )


# --------------------------------------------------------------------------
# Two-view objectives
# --------------------------------------------------------------------------
class _TwoViewObjective(nn.Module):
    """Encoder plus head for objectives that compare two views."""

    def __init__(self, encoder: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.enc = encoder
        self.head = head

    def encode(self, view1: torch.Tensor, view2: torch.Tensor) -> Tuple[List[torch.Tensor], tuple]:
        """Encode both views; returns ``([z1, z2], state)``."""
        z1 = self.enc(view1)
        z2 = self.enc(view2)
        return [z1, z2], (view1, view2, z1, z2)

    def task_loss(self, state: tuple) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Return ``(task_loss, [z1, z2])``."""
        representations, state = self.encode(view1, view2)
        return self.task_loss(state), representations


class ContrastiveObjective(_TwoViewObjective):
    """NT-Xent objective used by the SimCLR and CLOCS-style encoders.

    Both views are encoded, projected by :func:`projection_head` and compared
    with :func:`nt_xent_loss`.

    Args:
        encoder: Encoder mapping ``(batch, 12, T)`` to ``(batch, in_dim)``.
        temperature: NT-Xent temperature.
        in_dim: Representation dimension.
    """

    def __init__(self, encoder: nn.Module, temperature: float = 0.2, in_dim: int = 256) -> None:
        super().__init__(encoder, projection_head(in_dim))
        self.temperature = temperature

    def task_loss(self, state: tuple) -> torch.Tensor:
        """NT-Xent loss of the projected representations."""
        _, _, z1, z2 = state
        p1 = self.head(z1)
        p2 = self.head(z2)
        return nt_xent_loss(p1, p2, self.temperature)


class VICRegObjective(_TwoViewObjective):
    """VICReg objective: :func:`vicreg_expander` followed by :func:`vicreg_loss`.

    Args:
        encoder: Encoder mapping ``(batch, 12, T)`` to ``(batch, in_dim)``.
        in_dim: Representation dimension.
    """

    def __init__(self, encoder: nn.Module, in_dim: int = 256) -> None:
        super().__init__(encoder, vicreg_expander(in_dim))

    def task_loss(self, state: tuple) -> torch.Tensor:
        """VICReg loss of the expanded representations."""
        _, _, z1, z2 = state
        p1 = self.head(z1)
        p2 = self.head(z2)
        return vicreg_loss(p1, p2)


class BYOLObjective(_TwoViewObjective):
    """BYOL objective with an exponential-moving-average target network.

    The online branch is encoder -> projector (:func:`mlp_bn`) -> predictor
    (:func:`mlp_bn`). The target branch is a copy of the encoder and projector
    made at construction; it receives no gradients and is updated only by
    :meth:`update_target`. The loss is the symmetric
    ``byol_loss(q1, t2) + byol_loss(q2, t1)``, where ``q`` are online predictions
    and ``t`` are target projections computed without gradient. The target
    branch is evaluated in the same train/eval mode as the module.

    Args:
        encoder: Encoder mapping ``(batch, 12, T)`` to ``(batch, in_dim)``.
        momentum: Target update coefficient ``m`` in :meth:`update_target`.
        in_dim: Representation dimension.
        hidden_dim: Hidden width of the projector and predictor.
        out_dim: Output width of the projector and predictor.
    """

    def __init__(
        self,
        encoder: nn.Module,
        momentum: float = 0.996,
        in_dim: int = 256,
        hidden_dim: int = 1024,
        out_dim: int = 256,
    ) -> None:
        super().__init__(encoder, mlp_bn(in_dim, hidden_dim, out_dim))
        self.predictor = mlp_bn(out_dim, hidden_dim, out_dim)
        self.target_enc = copy.deepcopy(self.enc)
        self.target_head = copy.deepcopy(self.head)
        self.target_enc.requires_grad_(False)
        self.target_head.requires_grad_(False)
        self.momentum = momentum

    def task_loss(self, state: tuple) -> torch.Tensor:
        """Symmetric BYOL loss between online predictions and target projections."""
        view1, view2, z1, z2 = state
        p1 = self.head(z1)
        p2 = self.head(z2)
        p1 = self.predictor(p1)
        p2 = self.predictor(p2)
        with torch.no_grad():
            t1 = self.target_head(self.target_enc(view1))
            t2 = self.target_head(self.target_enc(view2))
        return byol_loss(p1, t2) + byol_loss(p2, t1)

    @torch.no_grad()
    def update_target(self) -> None:
        """Update the target branch from the online branch.

        For encoder and projector parameters, ``target <- m * target + (1 - m) * online``;
        buffers (BatchNorm statistics) are copied from the online branch. Call
        after every parameter update of the online branch.
        """
        m = self.momentum
        for online, target in ((self.enc, self.target_enc), (self.head, self.target_head)):
            for p, q in zip(online.parameters(), target.parameters()):
                q.mul_(m).add_(p, alpha=1 - m)
            for p, q in zip(online.buffers(), target.buffers()):
                q.copy_(p)


# --------------------------------------------------------------------------
# Reconstruction objectives
# --------------------------------------------------------------------------
class _ReconstructionObjective(nn.Module):
    """Encoder plus :func:`reconstruction_decoder` applied to the encoder feature map.

    The encoder must expose ``stem``, ``blocks`` and ``pool`` as
    :class:`~ecg_age_retention.models.resnet1d.ResNet1dEncoder` does. The
    decoder output is linearly interpolated to the length of the target. The
    representation returned by :meth:`encode` (and therefore the input of an
    auxiliary age head) is the pooled representation of the corrupted input.
    """

    def __init__(self, encoder: nn.Module, in_channels: int = 256, out_channels: int = 12) -> None:
        super().__init__()
        self.enc = encoder
        self.head = reconstruction_decoder(in_channels, out_channels)

    def _features(self, corrupted: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.enc.blocks(self.enc.stem(corrupted))
        z = self.enc.pool(feature_map).squeeze(-1)
        return feature_map, z

    def _reconstruct(self, feature_map: torch.Tensor, length: int) -> torch.Tensor:
        reconstruction = self.head(feature_map)
        return F.interpolate(reconstruction, size=length, mode="linear", align_corners=False).float()


class MaskedReconstructionObjective(_ReconstructionObjective):
    """SSL-MAE objective: reconstruct the clean signal from a temporally masked input.

    The loss (:func:`masked_reconstruction_loss`) is evaluated on the masked
    time samples only.
    """

    def encode(
        self, masked_input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[List[torch.Tensor], tuple]:
        """Encode the masked input; returns ``([z], state)``.

        Args:
            masked_input: Input with masked samples set to zero, shape ``(N, 12, T)``.
            target: Clean signal, shape ``(N, 12, T)``.
            mask: 1 at masked time samples, 0 elsewhere, shape ``(N, 1, T)``.
        """
        feature_map, z = self._features(masked_input)
        return [z], (feature_map, target, mask)

    def task_loss(self, state: tuple) -> torch.Tensor:
        """Masked reconstruction loss."""
        feature_map, target, mask = state
        reconstruction = self._reconstruct(feature_map, target.shape[-1])
        return masked_reconstruction_loss(reconstruction, target, mask)

    def forward(
        self, masked_input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Return ``(task_loss, [z])``."""
        representations, state = self.encode(masked_input, target, mask)
        return self.task_loss(state), representations


class DenoisingObjective(_ReconstructionObjective):
    """DAE objective: reconstruct the clean signal from a noise-corrupted input.

    The loss (:func:`reconstruction_mse_loss`) is the mean squared error over
    all leads and samples.
    """

    def encode(self, noisy_input: torch.Tensor, target: torch.Tensor) -> Tuple[List[torch.Tensor], tuple]:
        """Encode the noisy input; returns ``([z], state)``.

        Args:
            noisy_input: Corrupted signal, shape ``(N, 12, T)``.
            target: Clean signal, shape ``(N, 12, T)``.
        """
        feature_map, z = self._features(noisy_input)
        return [z], (feature_map, target)

    def task_loss(self, state: tuple) -> torch.Tensor:
        """Mean squared reconstruction error."""
        feature_map, target = state
        reconstruction = self._reconstruct(feature_map, target.shape[-1])
        return reconstruction_mse_loss(reconstruction, target)

    def forward(self, noisy_input: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Return ``(task_loss, [z])``."""
        representations, state = self.encode(noisy_input, target)
        return self.task_loss(state), representations
