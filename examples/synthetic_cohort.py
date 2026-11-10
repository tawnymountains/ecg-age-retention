"""Simulated patients and frozen representations shared by the examples.

Everything in this module is synthetic. Recorded age and a second prognostic
factor carried by the ECG (independent of age) drive an exponential survival
model. A simulated frozen "representation" is a fixed random linear mixture of
an age signal, the ECG factor and nuisance dimensions. When the age signal is
capped at a ceiling, all patients older than the ceiling look equally old to
any model fitted on the representation, so part of the age signal is missing.
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Sequence

import numpy as np

#: Prediction horizon used by the examples, in years.
HORIZON = 5.0


class SimulatedCohort(NamedTuple):
    """Simulated patients, one ECG per patient.

    Attributes
    ----------
    patient_id:
        Integer identifier of each patient.
    age:
        Recorded age at ECG acquisition, in years.
    ecg_factor:
        Prognostic factor carried by the ECG, independent of age.
    event:
        ``True`` if death was observed, ``False`` if censored.
    time:
        Follow-up time in years.
    """

    patient_id: np.ndarray
    age: np.ndarray
    ecg_factor: np.ndarray
    event: np.ndarray
    time: np.ndarray


def simulate_cohort(
    n: int,
    rng: np.random.Generator,
    *,
    age_effect: float = 0.8,
    ecg_effect: float = 0.6,
    baseline_hazard: float = 0.01,
    end_of_follow_up: float = 8.0,
) -> SimulatedCohort:
    """Draw ``n`` patients with exponential event times and administrative censoring.

    Ages are uniform on [18, 95) years. The hazard is
    ``baseline_hazard * exp(age_effect * (age - 60) / 10 + ecg_effect * ecg_factor)``
    with a standard normal ``ecg_factor``. Censoring times are uniform on
    [1, 12) years and capped at ``end_of_follow_up``.
    """
    age = rng.uniform(18.0, 95.0, n)
    ecg_factor = rng.standard_normal(n)
    log_hazard = age_effect * (age - 60.0) / 10.0 + ecg_effect * ecg_factor
    event_time = rng.exponential(1.0 / (baseline_hazard * np.exp(log_hazard)))
    censor_time = np.minimum(rng.uniform(1.0, 12.0, n), end_of_follow_up)
    event = event_time <= censor_time
    time = np.where(event, event_time, censor_time)
    return SimulatedCohort(np.arange(n), age, ecg_factor, event, time)


def simulate_representation(
    cohort: SimulatedCohort,
    rng: np.random.Generator,
    *,
    age_ceiling: Optional[float] = None,
    n_features: int = 32,
    age_noise: float = 0.15,
    ecg_noise: float = 0.3,
    nuisance_scale: float = 1.0,
) -> np.ndarray:
    """Simulated frozen representation of every patient, shape ``(n, n_features)``.

    The latent columns are the age signal ``(min(age, age_ceiling) - 60) / 15``
    plus Gaussian noise (SD ``age_noise``), the ECG factor plus Gaussian noise
    (SD ``ecg_noise``) and ``n_features - 2`` nuisance columns (SD
    ``nuisance_scale``). They are mixed by one random orthogonal matrix that is
    shared by all patients. ``age_ceiling=None`` keeps the full age signal.
    """
    n = cohort.age.shape[0]
    age = cohort.age if age_ceiling is None else np.minimum(cohort.age, age_ceiling)
    latent = np.column_stack(
        [
            (age - 60.0) / 15.0 + age_noise * rng.standard_normal(n),
            cohort.ecg_factor + ecg_noise * rng.standard_normal(n),
            nuisance_scale * rng.standard_normal((n, n_features - 2)),
        ]
    )
    mixing, _ = np.linalg.qr(rng.standard_normal((n_features, n_features)))
    return latent @ mixing


def split_patients(n: int, fractions: Sequence[float], rng: np.random.Generator) -> List[np.ndarray]:
    """Split ``range(n)`` at random into disjoint index sets with the given fractions."""
    order = rng.permutation(n)
    bounds = np.cumsum(np.round(np.asarray(fractions, dtype=float) * n).astype(int))[:-1]
    return np.split(order, bounds)


def band_label(lower: float, upper: float) -> str:
    """Label of a half-open recorded-age band, e.g. ``15-29`` or ``>=85``."""
    if np.isinf(upper):
        return f">={lower:g}"
    return f"{lower:g}-{upper - 1:g}"
