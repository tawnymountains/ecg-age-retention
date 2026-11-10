"""Independent survival evaluation of frozen ECG representations (Section II-B).

* :mod:`.cox` - frozen-representation Cox readout: representation transform
  (standardization with an optional PCA cap), L2-penalized Breslow Cox model,
  Breslow baseline hazard, predicted risk at the horizon (eq. 5) and penalty
  selection by cross-validated IPCW horizon log-loss.
* :mod:`.survival` - Kaplan-Meier observed risk, signed calibration error ``Q``
  (eq. 6), weighted calibration error over recorded-age bands (eq. 7), the
  older-patient error, Uno's C and the IPCW Brier score at the horizon.
"""

from .cox import (
    DEFAULT_PENALTY_GRID,
    CoxModel,
    CoxPrediction,
    CoxReadout,
    PenaltySelection,
    RepresentationTransform,
    breslow_cumulative_hazard,
    fit_cox_readout,
    fit_penalized_cox,
    ipcw_horizon_log_loss,
    select_cox_penalty,
)
from .survival import (
    DEFAULT_AGE_BANDS,
    OLDER_AGE_THRESHOLD,
    AgeBandCalibration,
    CalibrationSummary,
    age_band_calibration,
    calibration_error,
    calibration_summary,
    ipcw_brier_score,
    kaplan_meier_risk,
    uno_c,
    weighted_calibration_error,
)

__all__ = [
    "DEFAULT_PENALTY_GRID",
    "CoxModel",
    "CoxPrediction",
    "CoxReadout",
    "PenaltySelection",
    "RepresentationTransform",
    "breslow_cumulative_hazard",
    "fit_cox_readout",
    "fit_penalized_cox",
    "ipcw_horizon_log_loss",
    "select_cox_penalty",
    "DEFAULT_AGE_BANDS",
    "OLDER_AGE_THRESHOLD",
    "AgeBandCalibration",
    "CalibrationSummary",
    "age_band_calibration",
    "calibration_error",
    "calibration_summary",
    "ipcw_brier_score",
    "kaplan_meier_risk",
    "uno_c",
    "weighted_calibration_error",
]
