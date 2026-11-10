"""RankMe effective rank of a representation matrix.

RankMe summarizes how evenly the variation of a representation is spread over its
dimensions: the exponential of the Shannon entropy of the normalized singular
value spectrum. It needs no labels; higher values mean that feature variation is
distributed across more effective dimensions.

Definition used here
--------------------
For a representation matrix ``X`` of shape ``(n_rows, n_features)``:

1. Mean-center every column: ``Z = X - mean(X, axis=0)``. Features are not scaled.
2. Singular values of ``Z``: ``s_k = sqrt(max(lambda_k, 0))``, where ``lambda_k``
   are the eigenvalues of the Gram matrix ``Z^T Z``.
3. ``p_k = s_k / (sum_j s_j + 1e-7)``; entries with ``p_k == 0`` are dropped.
4. ``RankMe = exp(-sum_k p_k log p_k)``.

This differs from the original RankMe formulation ("RankMe: Assessing the
downstream performance of pretrained self-supervised representations by their
rank", ICML 2023) in two ways: the columns are centered before the spectrum is
computed, and the small constant is added to the normalizer instead of to each
``p_k``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["rankme"]

#: Constant added to the sum of singular values before normalization.
_NORMALIZER_OFFSET = 1e-7


def rankme(features: ArrayLike) -> float:
    """RankMe of a representation matrix computed on mean-centered, unscaled columns.

    Parameters
    ----------
    features:
        Representation matrix, shape ``(n_rows, n_features)``; converted to float64.
        Every row passed in contributes to the spectrum.

    Returns
    -------
    float
        Effective rank: at least 1 and at most about ``min(n_rows - 1, n_features)``.
        It is exactly 1.0 when the centered matrix is exactly zero.
    """
    centered = np.asarray(features, np.float64)
    centered = centered - centered.mean(0)
    singular = np.sqrt(np.maximum(np.linalg.eigvalsh(centered.T @ centered), 0))
    p = singular / (singular.sum() + _NORMALIZER_OFFSET)
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))
