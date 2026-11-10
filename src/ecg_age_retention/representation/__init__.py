"""Representation-level metrics computed on frozen ECG representations.

* :func:`nested_age_probe` and :func:`global_age_r2` - global age-R^2 (eq. 4) from a
  nested, patient-grouped ridge age probe with pooled out-of-fold predictions.
* :func:`rankme` - RankMe effective rank of a representation matrix.
* :func:`rank_agreement` - Spearman rank agreement of a model-level metric with the
  negated older-patient absolute risk error (eq. 8).
"""

from .age_probe import (
    DEFAULT_ALPHA_GRID,
    NestedAgeProbeResult,
    global_age_r2,
    nested_age_probe,
    ridge_age_predictions,
)
from .agreement import rank_agreement
from .rankme import rankme

__all__ = [
    "DEFAULT_ALPHA_GRID",
    "NestedAgeProbeResult",
    "global_age_r2",
    "nested_age_probe",
    "ridge_age_predictions",
    "rank_agreement",
    "rankme",
]
