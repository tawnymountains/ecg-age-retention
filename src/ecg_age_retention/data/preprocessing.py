"""Signal preprocessing to the common 12 x 4096 input.

Inputs are 12 x 4096 float32 arrays in the lead order of CODE-15%
(I, II, III, aVR, aVL, aVF, V1-V6). Each lead is z-score standardized within
its recording and clipped to [-10, 10]:

    ``x_std = clip((x - mean(x)) / (std(x) + 1e-6), -10, 10)``,

where the mean and the population standard deviation are computed in float32
over the samples of that lead.

A recording with a different lead order or length is processed as::

    x = reorder_leads(signal, lead_names)   # leads first: (12, n_samples)
    x = resample_fft(x, 4096)               # Fourier-domain resampling, float32
    x = standardize_leads(x)                # per-lead z-score and clipping

Resampling maps the whole recording to 4096 samples whatever its duration and
sampling rate. CODE-15% tracings already have 4096 samples in this lead order
and are only standardized; they are stored as ``(4096, 12)`` arrays, so pass
the transpose. If a recording contains non-finite samples, apply
:func:`zero_nonfinite` before resampling.

Only NumPy and SciPy are required.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from scipy.signal import resample

__all__ = ["LEAD_ORDER", "reorder_leads", "resample_fft", "standardize_leads", "zero_nonfinite"]

#: Lead order of the model input (the CODE-15% order).
LEAD_ORDER: Tuple[str, ...] = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


def reorder_leads(signal: np.ndarray, lead_names: Sequence[str], order: Sequence[str] = LEAD_ORDER) -> np.ndarray:
    """Select and reorder the rows of a leads-first signal by lead name.

    Names are matched case-insensitively after stripping surrounding
    whitespace (``"AVR"`` matches ``"aVR"``). If a name occurs more than once in
    ``lead_names``, its last occurrence is used.

    Args:
        signal: Array of shape ``(n_leads, n_samples)``; row ``i`` holds lead
            ``lead_names[i]``. For WFDB records read as ``(n_samples, n_leads)``,
            pass the transpose.
        lead_names: Name of each row of ``signal``.
        order: Target lead order.

    Returns:
        Array of shape ``(len(order), n_samples)`` with the dtype of ``signal``.

    Raises:
        ValueError: If ``signal`` is not two-dimensional, if the number of names
            differs from the number of rows, or if a lead of ``order`` is missing.
    """
    signal = np.asarray(signal)
    if signal.ndim != 2 or signal.shape[0] != len(lead_names):
        raise ValueError(f"expected a (n_leads, n_samples) array with one name per row, got shape {signal.shape}")
    position = {name.strip().lower(): i for i, name in enumerate(lead_names)}
    missing = [name for name in order if name.lower() not in position]
    if missing:
        raise ValueError(f"missing leads: {missing}")
    return signal[[position[name.lower()] for name in order]]


def resample_fft(signal: np.ndarray, n_samples: int = 4096) -> np.ndarray:
    """Resample along the last axis to ``n_samples`` with ``scipy.signal.resample``.

    The Fourier method is applied in the precision of the input (float64 input
    is resampled in float64, float32 input in float32) and the result is cast
    to float32.
    """
    return resample(signal, n_samples, axis=-1).astype(np.float32)


def standardize_leads(x: np.ndarray, eps: float = 1e-6, clip_value: float = 10.0) -> np.ndarray:
    """Per-recording, per-lead z-score standardization followed by clipping.

    Computes ``clip((x - mean) / (std + eps), -clip_value, clip_value)`` in
    float32, where the mean and the population standard deviation (``ddof=0``)
    are taken along the last axis. Any leading dimensions (leads, recordings)
    are kept.

    Args:
        x: Array of shape ``(..., n_leads, n_samples)``; cast to float32 first.
        eps: Constant added to the standard deviation.
        clip_value: Clipping bound.
    """
    x = np.asarray(x, dtype=np.float32)
    mu = x.mean(-1, keepdims=True)
    sd = x.std(-1, keepdims=True) + eps
    return np.clip((x - mu) / sd, -clip_value, clip_value)


def zero_nonfinite(x: np.ndarray) -> np.ndarray:
    """Replace NaN and positive or negative infinity with zero."""
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
