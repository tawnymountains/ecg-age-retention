"""Survival metrics at a fixed prediction horizon.

All functions take right-censored outcomes as two aligned one-dimensional arrays:
a boolean event indicator (``True`` = death observed, ``False`` = censored; numeric
0/1 is also accepted) and a non-negative, finite follow-up time. The prediction
horizon ``horizon`` (``H`` below) is given in the same time unit. Predicted
risks are absolute event probabilities by ``H``, for example the output of
:meth:`ecg_age_retention.evaluation.cox.CoxModel.predict` (eq. 5).

Kaplan-Meier observed risk
--------------------------
:func:`kaplan_meier_risk` returns ``1 - S(H)``, where ``S`` is the product-limit
estimator. At every distinct follow-up time ``t`` the survival is multiplied by
``1 - d(t) / r(t)``, where ``d(t)`` is the number of events at ``t`` and ``r(t)`` the
number of patients whose follow-up time is ``>= t`` (patients censored at ``t``
are still at risk at ``t``). ``S(H)`` is the value at the largest distinct time
``<= H`` and equals 1 when no follow-up time is ``<= H``.

Signed calibration error, eq. (6)
---------------------------------
For a patient group ``g``,
``Q_g(H) = 1000 * (F_KM_g(H) - mean_{i in g} F_i(H))``: the Kaplan-Meier observed
risk of the group minus its mean predicted risk, per 1,000 patients. Positive
values mean that risk is underestimated (:func:`calibration_error`).

Weighted calibration error, eq. (7)
-----------------------------------
Recorded age is divided into non-overlapping bands, by default the seven bands
15-29, 30-44, 45-59, 60-74, 75-79, 80-84 and >=85 years, implemented as half-open
intervals ``[lower, upper)``. ``WCE = sum_g (n_g / N) |Q_g(H)|``. Every patient must
fall into exactly one band, so ``N`` is both the number of patients and the sum
of the band sizes (:func:`age_band_calibration`, :func:`weighted_calibration_error`).
The older-patient absolute risk error is ``|Q_g(H)|`` for ``g = {i : age_i >= 75}``.

Censoring distribution
----------------------
Uno's C and the IPCW Brier score weight patients by the inverse of ``G``, the
product-limit estimate of the censoring survival function, computed on a separate
reference sample (for example, the patients the Cox model was fitted on). At
each distinct reference time ``t``, ``G`` is multiplied by
``1 - c(t) / (r(t) - d(t))``, where ``c(t)`` counts censorings at ``t`` and events
at ``t`` leave the risk set first. ``G`` is evaluated as a right-continuous step
function: the value at the largest reference time ``<= t``, where a query time
within machine epsilon (absolute) of a reference time counts as that time, and 1
before the first reference time. If the reference sample has no censoring,
``G = 1``. Query times beyond the largest reference time are an error unless ``G``
has already reached 0 there, in which case ``G = 0``. These conventions match
``concordance_index_ipcw`` and ``brier_score`` of scikit-survival.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "DEFAULT_AGE_BANDS",
    "OLDER_AGE_THRESHOLD",
    "kaplan_meier_risk",
    "calibration_error",
    "AgeBandCalibration",
    "age_band_calibration",
    "weighted_calibration_error",
    "CalibrationSummary",
    "calibration_summary",
    "uno_c",
    "ipcw_brier_score",
]

#: Seven recorded-age bands as half-open intervals ``[lower, upper)`` in years:
#: 15-29, 30-44, 45-59, 60-74, 75-79, 80-84 and >=85.
DEFAULT_AGE_BANDS: Tuple[Tuple[float, float], ...] = (
    (15.0, 30.0),
    (30.0, 45.0),
    (45.0, 60.0),
    (60.0, 75.0),
    (75.0, 80.0),
    (80.0, 85.0),
    (85.0, float("inf")),
)

#: Lower age bound (years, inclusive) of the older-patient subgroup.
OLDER_AGE_THRESHOLD: float = 75.0


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------
def _check_outcome(event: ArrayLike, time: ArrayLike, what: str = "") -> Tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Validate a right-censored outcome and return it as (bool, float64) arrays."""
    event = np.asarray(event)
    time = np.asarray(time, dtype=np.float64)
    label = f"{what} " if what else ""
    if event.ndim != 1 or time.ndim != 1:
        raise ValueError(f"{label}event and time must be one-dimensional")
    if event.shape[0] != time.shape[0]:
        raise ValueError(f"{label}event and time must have the same length")
    if time.shape[0] == 0:
        raise ValueError(f"{label}outcome arrays are empty")
    if event.dtype != np.bool_:
        values = np.unique(event)
        if not np.all(np.isin(values, [0, 1])):
            raise ValueError(f"{label}event indicator must be boolean or 0/1")
        event = event.astype(bool)
    if not np.all(np.isfinite(time)):
        raise ValueError(f"{label}time contains non-finite values")
    if np.any(time < 0):
        raise ValueError(f"{label}time contains negative values")
    return event, time


def _check_vector(values: ArrayLike, n: int, name: str, finite: bool = False) -> NDArray[np.float64]:
    """Return a float64 vector of length ``n``."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.shape[0] != n:
        raise ValueError(f"{name} must be one-dimensional with one value per patient")
    if finite and not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    return values


def _check_subset(subset: ArrayLike, n: int) -> np.ndarray:
    """Return a boolean mask of length ``n`` or an integer index array."""
    subset = np.asarray(subset)
    if subset.dtype == np.bool_:
        if subset.ndim != 1 or subset.shape[0] != n:
            raise ValueError("a boolean subset must have one entry per patient")
        return subset
    if not np.issubdtype(subset.dtype, np.integer) or subset.ndim != 1:
        raise ValueError("subset must be a boolean mask or a one-dimensional integer index array")
    return subset


#: Float64 machine epsilon; the snapping distance of a query time just below a
#: reference time of the censoring survival function.
_MACHINE_EPSILON = float(np.finfo(np.float64).eps)


def _tabulate_times(
    event: NDArray[np.bool_], time: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Summarize a validated outcome per distinct follow-up time.

    Returns the distinct times in increasing order and, for each of them, the
    number of patients with exactly that time, the number of events at that
    time and the number of patients whose time is at least that time.
    """
    distinct, slot = np.unique(time, return_inverse=True)
    slot = slot.reshape(-1)
    size = distinct.shape[0]
    patients = np.bincount(slot, minlength=size)
    events = np.bincount(slot[event], minlength=size)
    at_risk = np.cumsum(patients[::-1])[::-1]
    return distinct, patients, events, at_risk


# ---------------------------------------------------------------------------
# Kaplan-Meier observed risk and calibration error
# ---------------------------------------------------------------------------
def kaplan_meier_risk(event: ArrayLike, time: ArrayLike, horizon: float) -> float:
    """Kaplan-Meier observed event risk at ``horizon``.

    Returns ``1 - S(H)`` for ``H = horizon``, where ``S`` is the product-limit
    estimate ``S(H) = prod (1 - d_k / r_k)`` over the distinct follow-up times
    ``u_k <= H``; ``d_k`` is the number of events at ``u_k`` and ``r_k`` the
    number of patients with follow-up time ``>= u_k``. Times are compared
    exactly: events at a time equal to ``H`` are included, and patients censored
    at ``u_k`` are still at risk at ``u_k``. The factors are multiplied in order
    of increasing time.

    Parameters
    ----------
    event : array-like of shape (n,)
        Event indicator, boolean or numeric 0/1.
    time : array-like of shape (n,)
        Non-negative, finite follow-up times.
    horizon : float
        Horizon ``H``, in the unit of ``time``. Any real value is accepted;
        before the first follow-up time the estimate is 0.

    Returns
    -------
    float
        Estimated probability that the event has occurred by ``H``. It is
        exactly 0.0 when no event occurs at or before ``H``.

    Raises
    ------
    ValueError
        If ``event`` and ``time`` are not a valid outcome: not one-dimensional,
        of different or zero length, an event indicator other than boolean or
        0/1, or a negative or non-finite time.
    """
    event, time = _check_outcome(event, time)
    distinct, _, events, at_risk = _tabulate_times(event, time)
    included = int(np.searchsorted(distinct, horizon, side="right"))
    if included == 0:
        return 0.0
    factors = 1.0 - events[:included] / at_risk[:included]
    return float(1.0 - np.cumprod(factors)[-1])


def calibration_error(
    event: ArrayLike,
    time: ArrayLike,
    predicted_risk: ArrayLike,
    horizon: float,
    subset: Optional[ArrayLike] = None,
) -> float:
    """Signed calibration error ``Q_g(H)`` per 1,000 patients, eq. (6).

    ``Q = 1000 * (kaplan_meier_risk - mean(predicted_risk))`` over the patients in
    ``subset`` (all patients when ``subset`` is ``None``). Positive values indicate
    underestimation of risk, negative values overestimation.

    Parameters
    ----------
    event, time:
        Right-censored outcome of the evaluation patients.
    predicted_risk:
        Predicted event probability by ``horizon`` for each patient.
    horizon:
        Prediction horizon ``H``.
    subset:
        Optional boolean mask (one entry per patient) or integer indices selecting
        the patient group ``g``.
    """
    event, time = _check_outcome(event, time)
    risk = _check_vector(predicted_risk, time.shape[0], "predicted_risk")
    if subset is not None:
        keep = _check_subset(subset, time.shape[0])
        event, time, risk = event[keep], time[keep], risk[keep]
        if time.shape[0] == 0:
            raise ValueError("the selected patient group is empty")
    observed = kaplan_meier_risk(event, time, horizon)
    return float(1000 * (observed - risk.mean()))


class AgeBandCalibration(NamedTuple):
    """Per-band calibration of :func:`age_band_calibration`.

    Attributes
    ----------
    bands:
        The age bands ``(lower, upper)`` in input order; membership is
        ``lower <= age < upper``.
    n:
        Number of patients per band.
    n_events_by_horizon:
        Number of observed events with follow-up time ``<= horizon`` per band.
    observed_risk:
        Kaplan-Meier observed risk at the horizon per band.
    mean_predicted_risk:
        Mean predicted risk per band.
    q:
        Signed calibration error ``Q_g(H)`` per band (eq. 6).
    wce:
        Weighted calibration error ``sum_g (n_g / N) |Q_g(H)|`` (eq. 7).
    """

    bands: Tuple[Tuple[float, float], ...]
    n: NDArray[np.int64]
    n_events_by_horizon: NDArray[np.int64]
    observed_risk: NDArray[np.float64]
    mean_predicted_risk: NDArray[np.float64]
    q: NDArray[np.float64]
    wce: float


def age_band_calibration(
    event: ArrayLike,
    time: ArrayLike,
    predicted_risk: ArrayLike,
    age: ArrayLike,
    horizon: float,
    bands: Sequence[Tuple[float, float]] = DEFAULT_AGE_BANDS,
) -> AgeBandCalibration:
    """Signed calibration error in every recorded-age band and the WCE, eqs. (6)-(7).

    For each band ``g`` the patients with ``lower <= age < upper`` form the group,
    ``Q_g(H)`` is computed as in :func:`calibration_error`, and the bands are
    combined into ``WCE = sum_g (n_g / N) |Q_g(H)|``, accumulated in band order.

    Parameters
    ----------
    event, time:
        Right-censored outcome of the evaluation patients.
    predicted_risk:
        Predicted event probability by ``horizon`` for each patient.
    age:
        Recorded age at ECG acquisition for each patient.
    horizon:
        Prediction horizon ``H``.
    bands:
        Non-overlapping age bands ``(lower, upper)``; defaults to
        :data:`DEFAULT_AGE_BANDS`.

    Raises
    ------
    ValueError
        If a patient falls into no band or into more than one band, or if a band
        contains no patients.
    """
    event, time = _check_outcome(event, time)
    n = time.shape[0]
    risk = _check_vector(predicted_risk, n, "predicted_risk")
    age = _check_vector(age, n, "age")
    bands = tuple((float(lower), float(upper)) for lower, upper in bands)
    if len(bands) == 0:
        raise ValueError("at least one age band is required")
    members = [(age >= lower) & (age < upper) for lower, upper in bands]
    if np.any(np.sum(members, axis=0) != 1):
        raise ValueError("every patient must fall into exactly one age band")

    sizes, deaths, observed, predicted, q = [], [], [], [], []
    wce = 0.0
    for (lower, upper), member in zip(bands, members):
        n_g = int(member.sum())
        if n_g == 0:
            raise ValueError(f"age band [{lower:g}, {upper:g}) contains no patients")
        obs = kaplan_meier_risk(event[member], time[member], horizon)
        mean_risk = risk[member].mean()
        q_g = 1000 * (obs - mean_risk)
        wce += n_g / n * abs(q_g)
        sizes.append(n_g)
        deaths.append(int((event[member] & (time[member] <= horizon)).sum()))
        observed.append(obs)
        predicted.append(float(mean_risk))
        q.append(float(q_g))
    return AgeBandCalibration(
        bands=bands,
        n=np.asarray(sizes, dtype=np.int64),
        n_events_by_horizon=np.asarray(deaths, dtype=np.int64),
        observed_risk=np.asarray(observed, dtype=np.float64),
        mean_predicted_risk=np.asarray(predicted, dtype=np.float64),
        q=np.asarray(q, dtype=np.float64),
        wce=float(wce),
    )


def weighted_calibration_error(
    event: ArrayLike,
    time: ArrayLike,
    predicted_risk: ArrayLike,
    age: ArrayLike,
    horizon: float,
    bands: Sequence[Tuple[float, float]] = DEFAULT_AGE_BANDS,
) -> float:
    """Weighted calibration error (WCE) over recorded-age bands, eq. (7).

    Shortcut for ``age_band_calibration(...).wce``; see :func:`age_band_calibration`.
    """
    return age_band_calibration(event, time, predicted_risk, age, horizon, bands).wce


class CalibrationSummary(NamedTuple):
    """Calibration endpoints of :func:`calibration_summary`.

    Attributes
    ----------
    q_full:
        Signed calibration error ``Q_full(H)`` of all evaluation patients.
    q_older:
        Signed calibration error ``Q_{>=75}(H)`` of the older-patient subgroup; its
        absolute value is the older-patient absolute risk error.
    wce:
        Weighted calibration error over the age bands.
    bands:
        Per-band details (:class:`AgeBandCalibration`).
    """

    q_full: float
    q_older: float
    wce: float
    bands: AgeBandCalibration


def calibration_summary(
    event: ArrayLike,
    time: ArrayLike,
    predicted_risk: ArrayLike,
    age: ArrayLike,
    horizon: float,
    bands: Sequence[Tuple[float, float]] = DEFAULT_AGE_BANDS,
    older_threshold: float = OLDER_AGE_THRESHOLD,
) -> CalibrationSummary:
    """Full-cohort ``Q``, older-subgroup ``Q`` (``age >= older_threshold``) and WCE.

    Parameters
    ----------
    event, time:
        Right-censored outcome of the evaluation patients.
    predicted_risk:
        Predicted event probability by ``horizon`` for each patient.
    age:
        Recorded age at ECG acquisition for each patient.
    horizon:
        Prediction horizon ``H``.
    bands:
        Age bands for the WCE (default: :data:`DEFAULT_AGE_BANDS`).
    older_threshold:
        Lower age bound of the older-patient subgroup (default 75 years).
    """
    event, time = _check_outcome(event, time)
    n = time.shape[0]
    risk = _check_vector(predicted_risk, n, "predicted_risk")
    age = _check_vector(age, n, "age")
    band_result = age_band_calibration(event, time, risk, age, horizon, bands)
    return CalibrationSummary(
        q_full=calibration_error(event, time, risk, horizon),
        q_older=calibration_error(event, time, risk, horizon, subset=age >= older_threshold),
        wce=band_result.wce,
        bands=band_result,
    )


# ---------------------------------------------------------------------------
# Censoring distribution, Uno's C and IPCW Brier score
# ---------------------------------------------------------------------------
class _CensoringSurvival:
    """Censoring survival function ``G`` of a reference sample (module docstring)."""

    def __init__(self, event: NDArray[np.bool_], time: NDArray[np.float64]) -> None:
        """Estimate ``G`` from a validated reference outcome.

        At the distinct reference times ``u_1 < ... < u_m``,
        ``G(u_k) = prod_{l <= k} (1 - c_l / (r_l - d_l))``, where ``c_l`` and
        ``d_l`` count the censorings and the events at ``u_l`` and ``r_l`` the
        patients with time ``>= u_l``: events leave the risk set before the
        censorings at the same time are counted. A time without censoring
        contributes the factor 1. ``G(u_m)`` is 0 exactly when the largest
        reference time carries a censoring.

        Parameters
        ----------
        event : ndarray of bool, shape (n,)
            Event indicator of the reference sample, already validated.
        time : ndarray of float64, shape (n,)
            Follow-up times of the reference sample, already validated.

        Raises
        ------
        ValueError
            If no patient of the reference sample has an event.
        """
        if not np.any(event):
            raise ValueError("every patient of the reference sample is censored; G is not estimated")
        distinct, patients, events, at_risk = _tabulate_times(event, time)
        censored = patients - events
        factors = np.ones(distinct.shape[0])
        with_censoring = censored > 0
        factors[with_censoring] = 1.0 - censored[with_censoring] / (at_risk - events)[with_censoring]
        self._times = distinct
        self._values = np.cumprod(factors)

    def __call__(self, query: ArrayLike) -> NDArray[np.float64]:
        """Evaluate ``G`` at the query times.

        ``G`` is the right-continuous step function with steps at the reference
        times and value 1 before the first one. A query ``q`` less than one
        machine epsilon below a reference time ``u`` (``u - q < eps``, absolute)
        takes the value at ``u``. A query after the largest reference time takes the value 0
        when ``G`` is 0 there and is an error otherwise; no snapping applies above
        the largest reference time.

        Parameters
        ----------
        query : array-like of float
            Query times; an empty array gives an empty result.

        Returns
        -------
        ndarray of float64
            ``G`` at the query times, with the shape of ``query``.

        Raises
        ------
        ValueError
            If a query time is NaN, or if a query time is later than the largest
            reference time while ``G`` is still positive at that time.
        """
        query = np.asarray(query, dtype=np.float64)
        flat = query.reshape(-1)
        if np.isnan(flat).any():
            raise ValueError("G cannot be evaluated at a NaN time")
        n_steps = self._times.shape[0]
        # Index of the smallest reference time >= each query (n_steps if none).
        ceiling = np.searchsorted(self._times, flat, side="left")
        beyond = ceiling == n_steps
        if beyond.any() and self._values[-1] > 0:
            raise ValueError(
                "G is undefined after the largest reference time because it is still positive there"
            )
        result = np.zeros(flat.shape[0])
        inside = np.flatnonzero(~beyond)
        step = ceiling[inside]
        snapped = self._times[step] - flat[inside] < _MACHINE_EPSILON
        step = np.where(snapped, step, step - 1)
        result[inside] = np.where(step >= 0, self._values[np.maximum(step, 0)], 1.0)
        return result.reshape(query.shape)


def _first_true(lower: NDArray[np.int64], upper: NDArray[np.int64], predicate) -> NDArray[np.int64]:
    """Vectorized binary search for the first index at which a predicate holds.

    For every query ``k`` the predicate ``predicate(index, k)`` (evaluated on
    arrays of indices and query numbers) is assumed to be false and then true as
    ``index`` runs from ``lower[k]`` to ``upper[k] - 1``. Returns the smallest
    such index where it holds, or ``upper[k]`` when it holds nowhere.
    """
    lower = lower.copy()
    upper = upper.copy()
    open_queries = np.flatnonzero(lower < upper)
    while open_queries.size:
        middle = (lower[open_queries] + upper[open_queries]) // 2
        holds = np.asarray(predicate(middle, open_queries), dtype=bool)
        upper[open_queries[holds]] = middle[holds]
        lower[open_queries[~holds]] = middle[~holds] + 1
        open_queries = open_queries[lower[open_queries] < upper[open_queries]]
    return lower


def _tie_window(
    levels: NDArray[np.float64], own: NDArray[np.int64], score: NDArray[np.float64], tol: float
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Positions ``[start, stop)`` in ``levels`` of the values tied with each score.

    ``levels`` holds the distinct scores in increasing order and ``own[k]`` is the
    position of ``score[k]`` in it. A value ``v`` is tied with ``s`` when the
    float64 result of ``abs(v - s)`` is at most ``tol``. That result never
    decreases as ``v`` moves away from ``s``, so the tied values form one run of
    ``levels`` around ``own``. With a negative or NaN ``tol`` nothing is tied and
    ``start = stop = own``.
    """
    if not tol >= 0.0:
        return own.copy(), own.copy()

    def tied(position, k):
        return np.abs(levels[position] - score[k]) <= tol

    start = _first_true(np.zeros_like(own), own, tied)
    stop = _first_true(own + 1, np.full_like(own, levels.shape[0]), lambda position, k: ~tied(position, k))
    return start, stop


def _count_prefix_below(
    values: NDArray[np.int64], length: NDArray[np.int64], bound: NDArray[np.int64]
) -> NDArray[np.int64]:
    """Count, for every query ``k``, the positions ``p < length[k]`` with ``values[p] < bound[k]``.

    ``values`` are non-negative integers and every bound is at most
    ``values.max() + 1``. The prefix ``[0, length)`` is the union of aligned
    blocks whose sizes are the powers of two in the binary expansion of
    ``length``. For every block size the values are sorted within their blocks
    (as one array of keys ``block number * stride + value``), so a single binary
    search per query counts the values below the bound in one block. Cost:
    ``O(n log(n)**2)`` for ``n`` values, without loops over patients.
    """
    n = values.shape[0]
    stride = int(values.max()) + 2
    position = np.arange(n, dtype=np.int64)
    counts = np.zeros(length.shape[0], dtype=np.int64)
    level = 0
    while (1 << level) <= n:
        uses_level = ((length >> level) & 1).astype(bool)
        if uses_level.any():
            block = (length[uses_level] >> (level + 1)) << 1
            keys = np.sort((position >> level) * stride + values)
            found = np.searchsorted(keys, block * stride + bound[uses_level], side="left")
            # Every block before ``block`` is full and lies entirely below the search key.
            counts[uses_level] += found - (block << level)
        level += 1
    return counts


def uno_c(
    train_event: ArrayLike,
    train_time: ArrayLike,
    event: ArrayLike,
    time: ArrayLike,
    risk_score: ArrayLike,
    horizon: float,
    tied_tol: float = 1e-8,
) -> float:
    """Uno's concordance index truncated at ``horizon``.

    Inverse-probability-of-censoring-weighted concordance (Uno et al., 2011).
    An ordered pair ``(i, j)`` of evaluation patients is comparable when
    patient ``i`` has an observed event and either ``time[j] > time[i]``, or
    ``time[j] == time[i]`` and patient ``j`` is censored; two events at the same
    time are never compared. Only events with ``time[i] < horizon`` start pairs,
    but their partners may have any later time, including times at or after
    the horizon. Every pair started by ``i`` has the weight ``G(time[i]) ** -2``,
    where ``G`` is the censoring survival function estimated on the reference
    sample (module docstring). A pair whose scores are tied, that is
    ``abs(risk_score[j] - risk_score[i]) <= tied_tol`` in float64, earns 1/2;
    otherwise it earns 1 when ``risk_score[j] < risk_score[i]`` (higher score =
    higher risk) and 0 when not. The result is the weighted sum of the earned
    credits divided by the weighted number of comparable pairs. All time
    comparisons are exact.

    Parameters
    ----------
    train_event, train_time : array-like of shape (m,)
        Outcome of the reference sample on which ``G`` is estimated.
    event, time : array-like of shape (n,)
        Outcome of the evaluation patients.
    risk_score : array-like of shape (n,)
        Finite risk scores of the evaluation patients.
    horizon : float
        Truncation time ``H``; ``numpy.inf`` means no truncation.
    tied_tol : float, default 1e-8
        Tie tolerance for two scores; with a negative value no pair is tied.

    Returns
    -------
    float
        The truncated concordance index, or NaN when no event before the horizon
        has a comparable partner.

    Raises
    ------
    ValueError
        If an outcome is invalid or a risk score is not finite; if no evaluation
        patient has an event, or none has an event before ``horizon``; if every
        reference patient is censored; if ``G`` is undefined (an event time before
        the horizon lies after the largest reference time while ``G`` is still
        positive there) or is 0 at an event time before the horizon; or if the
        evaluation sample has a single event and its time is later than every
        other evaluation time, so that no pair can be comparable.
    """
    reference_event, reference_time = _check_outcome(train_event, train_time, "reference")
    event, time = _check_outcome(event, time, "evaluation")
    n = time.shape[0]
    score = _check_vector(risk_score, n, "risk_score", finite=True)
    if not event.any():
        raise ValueError("no evaluation patient has an observed event")
    censoring = _CensoringSurvival(reference_event, reference_time)
    starts_pairs = event & (time < horizon)
    if not starts_pairs.any():
        raise ValueError("no evaluation patient has an observed event before the horizon")
    if np.count_nonzero(event) == 1:
        lone = int(np.flatnonzero(event)[0])
        others = np.delete(time, lone)
        if others.size == 0 or time[lone] > others.max():
            raise ValueError("the evaluation sample contains no comparable pair")
    starter_time = time[starts_pairs]
    g = censoring(starter_time)
    if np.any(g == 0):
        raise ValueError("G is 0 at an observed event time before the horizon")
    weight = np.square(1.0 / g)

    # Partners of an event at time t: every patient with a later time, then the
    # patients censored at exactly t. Sorting by decreasing time, with censored
    # patients before events at equal times, makes every partner set a prefix.
    order = np.lexsort((event, -time))
    sorted_time = np.sort(time)
    censored_time = np.sort(time[~event])
    n_partners = (n - np.searchsorted(sorted_time, starter_time, side="right")) + (
        np.searchsorted(censored_time, starter_time, side="right")
        - np.searchsorted(censored_time, starter_time, side="left")
    )
    levels, score_level = np.unique(score, return_inverse=True)
    score_level = score_level.reshape(-1)
    tie_start, tie_stop = _tie_window(levels, score_level[starts_pairs], score[starts_pairs], float(tied_tol))
    below = _count_prefix_below(
        score_level[order], np.concatenate([n_partners, n_partners]), np.concatenate([tie_start, tie_stop])
    )
    n_starters = n_partners.shape[0]
    concordant = below[:n_starters]
    tied = below[n_starters:] - concordant
    denominator = np.sum(weight * n_partners)
    if denominator == 0:
        return float("nan")
    return float(np.sum(weight * (concordant + 0.5 * tied)) / denominator)


def ipcw_brier_score(
    train_event: ArrayLike,
    train_time: ArrayLike,
    event: ArrayLike,
    time: ArrayLike,
    predicted_risk: ArrayLike,
    horizon: float,
) -> float:
    """Inverse-probability-of-censoring-weighted Brier score at ``horizon``.

    With ``H = horizon`` and predicted survival ``S_i = 1 - predicted_risk[i]``
    (Graf et al., 1999),

    ``BS(H) = (1/n) * sum_i (A_i + B_i)``, where
    ``A_i = S_i**2 / G(time[i])`` when patient ``i`` has an observed event with
    ``time[i] <= H`` and ``B_i = (1 - S_i)**2 / G(H)`` when ``time[i] > H``;
    all other terms are 0. Patients censored at or before ``H`` therefore
    contribute 0 but are counted in ``n``. A term whose value of ``G`` is 0 is 0.
    ``G`` is the censoring survival function estimated on the reference sample
    (module docstring); it is evaluated at ``H`` and at the time of every
    evaluation patient.

    Parameters
    ----------
    train_event, train_time : array-like of shape (m,)
        Outcome of the reference sample on which ``G`` is estimated.
    event, time : array-like of shape (n,)
        Outcome of the evaluation patients.
    predicted_risk : array-like of shape (n,)
        Finite predicted event probabilities by ``H``; the range is not checked.
    horizon : float
        Horizon ``H``; it must satisfy ``min(time) <= H < max(time)`` over all
        evaluation patients.

    Returns
    -------
    float
        The IPCW Brier score at ``H``.

    Raises
    ------
    ValueError
        If an outcome is invalid or a predicted risk is not finite; if no
        evaluation patient has an event; if ``horizon`` lies outside
        ``[min(time), max(time))`` or is NaN; if every reference patient is
        censored; or if ``G`` is undefined at ``H`` or at an evaluation time (a
        time after the largest reference time while ``G`` is still positive
        there).
    """
    reference_event, reference_time = _check_outcome(train_event, train_time, "reference")
    event, time = _check_outcome(event, time, "evaluation")
    n = time.shape[0]
    risk = _check_vector(predicted_risk, n, "predicted_risk", finite=True)
    if not event.any():
        raise ValueError("no evaluation patient has an observed event")
    if not time.min() <= horizon < time.max():
        raise ValueError("the horizon must satisfy min(time) <= horizon < max(time)")
    censoring = _CensoringSurvival(reference_event, reference_time)
    g_patient = censoring(time)
    g_horizon = float(censoring(np.array([horizon], dtype=np.float64))[0])
    survival = 1.0 - risk
    terms = np.zeros(n)
    case = event & (time <= horizon) & (g_patient > 0)
    terms[case] = survival[case] ** 2 / g_patient[case]
    if g_horizon > 0:
        control = time > horizon
        terms[control] = (1.0 - survival[control]) ** 2 / g_horizon
    return float(np.mean(terms))
