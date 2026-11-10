"""One-dimensional residual network encoder for 12-lead ECG.

This is the encoder backbone used in the paper: a convolutional stem, four
residual blocks and global average pooling, mapping
an ECG of shape ``(batch, 12, n_samples)`` to a 256-dimensional
representation of shape ``(batch, 256)``.

Design note: the kernel size (16), the down-sampling factor per block (4), the
dropout rate (0.2) and the channel sequence 64-128-196-256 follow the residual
network of Ribeiro et al., "Automatic diagnosis of the 12-lead ECG using a deep
neural network", Nat. Commun. 11, 1760 (2020). The implementation is
independent of that work's code and differs from it in the block layout
(post-activation residual blocks whose shortcut always applies max-pooling and
a 1x1 convolution) and in the output (global average pooling instead of
flattening).
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

__all__ = ["ResBlock1d", "ResNet1dEncoder"]


class ResBlock1d(nn.Module):
    """Post-activation residual block with a down-sampling shortcut.

    Main path: ``conv(k) -> BatchNorm -> ReLU -> Dropout -> conv(k, stride=downsample)
    -> BatchNorm``. Shortcut: ``MaxPool1d(downsample, ceil_mode=True) -> 1x1 conv``.
    Both convolutions of the main path use zero padding ``kernel_size // 2`` and
    no bias. The two paths are cropped to the shorter of their two lengths
    before they are summed, and the sum is passed through ReLU.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        downsample: Stride of the second convolution and size of the shortcut
            max-pooling window.
        dropout: Dropout probability between the two convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 16,
        downsample: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=downsample, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.down = nn.Sequential(
            nn.MaxPool1d(downsample, ceil_mode=True),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block to ``x`` of shape ``(batch, in_channels, length)``."""
        res = self.down(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        m = min(out.shape[-1], res.shape[-1])
        out, res = out[..., :m], res[..., :m]
        return self.relu(out + res)


class ResNet1dEncoder(nn.Module):
    """1D ResNet encoder: stem, four residual blocks and global average pooling.

    The stem is ``Conv1d(in_channels, base, kernel_size) -> BatchNorm -> ReLU``
    (zero padding ``kernel_size // 2``, no bias, no down-sampling). It is
    followed by one :class:`ResBlock1d` per entry of ``blocks``, each reducing
    the length by ``downsample``, and by adaptive average pooling over time.
    With the defaults, a 12 x 4096 ECG is reduced to a 256-dimensional
    representation (``out_dim``).

    The module attribute names (``stem``, ``blocks``, ``pool``) define the
    state-dict keys of the trained encoders, and the reconstruction objectives
    in :mod:`ecg_age_retention.models.ssl` use ``stem``, ``blocks`` and
    ``pool`` directly.

    Args:
        in_channels: Number of ECG leads.
        base: Number of channels produced by the stem.
        blocks: Output channels of the residual blocks.
        kernel_size: Convolution kernel size.
        downsample: Down-sampling factor of each residual block.
        dropout: Dropout probability inside each residual block.
    """

    def __init__(
        self,
        in_channels: int = 12,
        base: int = 64,
        blocks: Sequence[int] = (64, 128, 196, 256),
        kernel_size: int = 16,
        downsample: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(base),
            nn.ReLU(inplace=True),
        )
        channels = [base] + list(blocks)
        self.blocks = nn.Sequential(
            *[
                ResBlock1d(channels[i], channels[i + 1], kernel_size, downsample, dropout)
                for i in range(len(blocks))
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = blocks[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``x`` of shape ``(batch, in_channels, n_samples)`` to ``(batch, out_dim)``."""
        x = self.stem(x)
        x = self.blocks(x)
        return self.pool(x).squeeze(-1)
