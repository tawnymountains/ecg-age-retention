"""Rank agreement between a model-level metric and older-patient absolute risk error.

Given one value of a metric per model and the older-patient absolute risk error
``|Q_{m,>=75}(H)|`` of the same models, the rank agreement is the Spearman rank
correlation across models between the metric, oriented so that higher is better,
and the negated older-patient error:

    rho = Spearman( s * metric_m , -|Q_{m,>=75}(H)| ),   s = +1 or -1.

With global age-R^2 as the metric (``s = +1``) this is ``rho_agree`` of eq. (8).
Positive values mean that models ranked higher by the metric tend to have smaller
older-patient absolute risk error. Metrics for which lower values are better, such
as the full-cohort absolute calibration error ``|Q_{m,full}(H)|``, are negated
before ranking (``higher_is_better=False``), so that positive values again
indicate agreement.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import spearmanr

__all__ = ["rank_agreement"]


def rank_agreement(
    metric: ArrayLike,
    older_error: ArrayLike,
    *,
    higher_is_better: bool = True,
) -> float:
    """Spearman rank agreement of a metric with negated older-patient absolute risk error.

    Parameters
    ----------
    metric:
        One value of the metric per model, shape ``(n_models,)``, for example
        global age-R^2 or Uno's C.
    older_error:
        Older-patient absolute risk error ``|Q_{m,>=75}(H)|`` of the same models in
        the same order, shape ``(n_models,)``. Pass absolute values; the function
        negates them and does not take absolute values itself.
    higher_is_better:
        ``True`` if larger metric values are preferable (global age-R^2, Uno's C).
        ``False`` negates the metric before ranking (for example ``|Q_{m,full}(H)|``,
        passed as absolute values).

    Returns
    -------
    float
        Spearman correlation coefficient from ``scipy.stats.spearmanr`` (tied
        values receive average ranks). NaN if either input contains NaN or is
        constant.
    """
    metric = np.asarray(metric, dtype=np.float64)
    older_error = np.asarray(older_error, dtype=np.float64)
    oriented = metric if higher_is_better else -metric
    return float(spearmanr(oriented, -older_error)[0])
