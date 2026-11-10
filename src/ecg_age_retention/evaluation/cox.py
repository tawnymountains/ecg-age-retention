"""Frozen-representation Cox readout (Section II-B).

A frozen ECG representation is turned into an absolute risk by ``H`` in four steps.

1. Representation transform (:class:`RepresentationTransform`), fitted on the
   Cox-fit patients only: per-feature standardization to mean 0 and unit
   population variance (a zero-variance feature keeps scale 1). When the
   representation has more than ``max_dim`` columns (default 256), PCA to
   ``max_dim`` components follows and the components are standardized again.
2. L2-penalized Cox proportional-hazards model with the Breslow approximation for
   tied event times (:func:`fit_penalized_cox`). The minimized objective is

   ``L(theta) = (1/n) sum_g [ d_g log(sum_{j: t_j >= t_g} exp(eta_j)) - sum_{i in D_g} eta_i ]
   + alpha / (2 n) * ||theta||^2``,

   with ``eta = z theta``, ``g`` running over the distinct event times ``t_g``,
   ``D_g`` the ``d_g`` events at ``t_g`` and ``n`` the number of patients: the
   negative log partial likelihood plus ``(alpha / 2) ||theta||^2``, divided by
   ``n``.
3. Breslow estimate of the cumulative baseline hazard ``H_0`` at every distinct
   follow-up time of the Cox-fit patients (:func:`breslow_cumulative_hazard`).
   ``H_0`` refers to ``eta = 0``, i.e. to the mean of the transformed Cox-fit
   representation.
4. Predicted risk ``F_i(H) = 1 - exp(-H_0(H) exp(eta_i))``, eq. (5), where ``H_0(H)``
   is the value at the largest Cox-fit follow-up time ``<= H``
   (:meth:`CoxModel.predict`).

Solvers
-------
``solver="newton"``
    Newton-Raphson from ``theta = 0``. Each iteration solves the Newton step from
    the gradient and Hessian of ``L``. If ``L`` at the new point exceeds ``L`` at the
    current point, the current point is instead moved halfway back to the
    previously accepted point and the iteration counts as used. Otherwise the new
    point is accepted, and the fit stops once ``|1 - L_new / L_old| < tol``. Failing
    to stop within ``max_iter`` iterations, or a final maximum absolute gradient of
    at least ``1e-5``, raises :class:`RuntimeError`. These rules are independently
    implemented to give the results of scikit-survival's
    ``CoxPHSurvivalAnalysis(ties="breslow")``; with the same
    ``alpha``, ``n_iter`` and ``tol`` both stop at the same optimum, up to the
    accuracy implied by ``tol`` when a rounding-level change of ``L`` decides the
    step-halving test in the last iteration.
``solver="lbfgs"``
    SciPy L-BFGS-B from ``theta = 0`` on the same objective written with running
    log-sum-exp (``maxiter=2000``, ``gtol=1e-9``, ``ftol=1e-15``, ``maxls=50``,
    ``maxcor=20``). A status other than 0 or 1, or a final maximum absolute
    gradient of at least ``1e-7``, raises :class:`RuntimeError`.

Penalty selection
-----------------
:func:`select_cox_penalty` chooses ``alpha`` from a grid by K-fold
cross-validation inside the Cox-fit patients (scikit-learn ``KFold`` with
shuffling). In every fold the features are standardized on the training part
(no PCA), a ``"newton"`` model is fitted for each candidate, and each validation
patient receives the IPCW horizon log-loss of its prediction
(:func:`ipcw_horizon_log_loss`), with the censoring distribution estimated on the
training part. The criterion is the mean weighted loss over all Cox-fit patients
(each appears in exactly one validation fold; patients with weight 0 count as 0),
and the smallest criterion wins (the first candidate in grid order on exact ties).

Example
-------
::

    selection = select_cox_penalty(x_fit, event_fit, time_fit, horizon, random_state=seed)
    readout = fit_cox_readout(x_fit, event_fit, time_fit, alpha=selection.alpha)
    prediction = readout.predict(x_eval, horizon)
    prediction.risk  # F_i(H), eq. (5)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from .survival import _CensoringSurvival, _check_outcome

__all__ = [
    "DEFAULT_PENALTY_GRID",
    "CoxPrediction",
    "CoxModel",
    "breslow_cumulative_hazard",
    "fit_penalized_cox",
    "ipcw_horizon_log_loss",
    "PenaltySelection",
    "select_cox_penalty",
    "RepresentationTransform",
    "CoxReadout",
    "fit_cox_readout",
]

#: Default candidate penalties of :func:`select_cox_penalty`.
DEFAULT_PENALTY_GRID: Tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)

_NEWTON_GRADIENT_TOL = 1e-5
_LBFGS_GRADIENT_TOL = 1e-7
_LBFGS_OPTIONS = dict(maxiter=2000, gtol=1e-9, ftol=1e-15, maxls=50, maxcor=20)


def _check_features(x: ArrayLike, n_rows: Optional[int] = None, name: str = "x") -> NDArray[np.float64]:
    """Return a finite two-dimensional float64 array."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array (patients x features)")
    if n_rows is not None and x.shape[0] != n_rows:
        raise ValueError(f"{name} must have one row per patient")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values")
    return x


# ---------------------------------------------------------------------------
# Cox model
# ---------------------------------------------------------------------------
class CoxPrediction(NamedTuple):
    """Output of :meth:`CoxModel.predict`.

    Attributes
    ----------
    risk:
        Predicted event probability by the horizon, ``F_i(H)`` of eq. (5).
    linear_predictor:
        Linear risk score ``eta_i = theta^T z_i``.
    log_cumulative_hazard:
        ``log H_0(H) + eta_i``, the log cumulative hazard at the horizon.
    """

    risk: NDArray[np.float64]
    linear_predictor: NDArray[np.float64]
    log_cumulative_hazard: NDArray[np.float64]


@dataclass(frozen=True)
class CoxModel:
    """Fitted L2-penalized Cox model with Breslow baseline.

    Attributes
    ----------
    coef:
        Regression coefficients ``theta``.
    alpha:
        Penalty strength used for the fit.
    unique_times:
        Distinct follow-up times (events and censorings) of the Cox-fit patients,
        ascending.
    cumulative_baseline_hazard:
        Breslow cumulative baseline hazard ``H_0`` at ``unique_times``.
    solver:
        ``"newton"`` or ``"lbfgs"``.
    n_iter:
        Number of solver iterations.
    """

    coef: NDArray[np.float64]
    alpha: float
    unique_times: NDArray[np.float64]
    cumulative_baseline_hazard: NDArray[np.float64]
    solver: str
    n_iter: int

    def linear_predictor(self, z: ArrayLike) -> NDArray[np.float64]:
        """Linear risk score ``eta = z theta`` for transformed representations ``z``."""
        z = _check_features(z, name="z")
        if z.shape[1] != self.coef.shape[0]:
            raise ValueError(f"z has {z.shape[1]} columns, the model expects {self.coef.shape[0]}")
        return np.dot(z, self.coef)

    def baseline_cumulative_hazard(self, horizon: float) -> float:
        """``H_0(H)``: the Breslow cumulative baseline hazard at the largest Cox-fit time ``<= H``."""
        k = int(np.searchsorted(self.unique_times, horizon, side="right")) - 1
        if k < 0:
            raise ValueError("the horizon precedes the first follow-up time of the Cox-fit patients")
        h0 = float(self.cumulative_baseline_hazard[k])
        if not h0 > 0:
            raise ValueError("the Cox-fit patients have no event up to the horizon")
        return h0

    def predict(self, z: ArrayLike, horizon: float) -> CoxPrediction:
        """Predicted risk ``F_i(H) = 1 - exp(-H_0(H) exp(eta_i))``, eq. (5).

        Parameters
        ----------
        z:
            Transformed representations (same transform as for fitting), one row
            per patient.
        horizon:
            Prediction horizon ``H``.
        """
        eta = self.linear_predictor(z)
        log_cumulative_hazard = np.log(self.baseline_cumulative_hazard(horizon)) + eta
        with np.errstate(over="ignore"):
            risk = -np.expm1(-np.exp(log_cumulative_hazard))
        if not (np.all(np.isfinite(eta)) and np.all(np.isfinite(risk))):
            raise FloatingPointError("non-finite linear predictor or risk")
        return CoxPrediction(risk=risk, linear_predictor=eta, log_cumulative_hazard=log_cumulative_hazard)


def breslow_cumulative_hazard(
    linear_predictor: ArrayLike, event: ArrayLike, time: ArrayLike
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Breslow cumulative baseline hazard at every distinct follow-up time.

    ``H_0(t_k) = sum_{l <= k} d_l / sum_{j: t_j >= t_l} exp(eta_j)``, where ``t_1 < t_2 < ...``
    are the distinct follow-up times (event or censoring) and ``d_l`` the number of
    events at ``t_l``.

    Returns
    -------
    unique_times, cumulative_hazard:
        Distinct follow-up times (ascending) and ``H_0`` at those times.
    """
    event, time = _check_outcome(event, time)
    eta = np.asarray(linear_predictor, dtype=np.float64)
    if eta.ndim != 1 or eta.shape[0] != time.shape[0]:
        raise ValueError("linear_predictor must have one value per patient")
    unique_times, inverse = np.unique(time, return_inverse=True)
    inverse = inverse.reshape(-1)
    deaths = np.bincount(inverse, weights=event, minlength=unique_times.shape[0])
    risk_sum = np.bincount(inverse, weights=np.exp(eta), minlength=unique_times.shape[0])
    cumulative_hazard = np.cumsum(deaths / np.cumsum(risk_sum[::-1])[::-1])
    return unique_times, cumulative_hazard


#: Penalties below the float64 machine epsilon are left out of the gradient and Hessian.
_PENALTY_EPSILON = float(np.finfo(np.float64).eps)

#: Approximate number of matrix entries per row block when the Hessian is accumulated.
_BLOCK_ENTRIES = 1 << 20


class _NewtonObjective:
    """Penalized Breslow objective ``L`` with its gradient and Hessian (module docstring)."""

    def __init__(self, z: NDArray[np.float64], event: NDArray[np.bool_], time: NDArray[np.float64], alpha: float):
        """Prepare ``L`` for validated Cox-fit data ``z``, ``event``, ``time`` and penalty ``alpha``.

        The rows are stored in order of increasing follow-up time together with
        the risk-set structure: for every distinct event time ``v_g`` the number of
        events ``d_g`` and the first stored row whose time is ``>= v_g``. The
        caller's arrays are not modified.

        Parameters
        ----------
        z : ndarray of float64, shape (n, p)
            Finite covariates (transformed representations), one row per patient.
        event : ndarray of bool, shape (n,)
            Event indicator.
        time : ndarray of float64, shape (n,)
            Finite, non-negative follow-up times.
        alpha : float
            Penalty strength ``>= 0``.
        """
        order = np.argsort(time, kind="stable")
        self._z = np.ascontiguousarray(z[order], dtype=np.float64)
        sorted_time = time[order]
        sorted_event = np.asarray(event, dtype=bool)[order]
        n = sorted_time.shape[0]
        opens_time = np.ones(n, dtype=bool)
        opens_time[1:] = sorted_time[1:] != sorted_time[:-1]
        time_slot = np.cumsum(opens_time) - 1
        first_row = np.flatnonzero(opens_time)
        events_at_time = np.bincount(time_slot[sorted_event], minlength=first_row.shape[0])
        is_event_time = events_at_time > 0
        # First stored row of the risk set of every event time, and its number of events.
        self._risk_start = first_row[is_event_time]
        self._deaths = events_at_time[is_event_time].astype(np.float64)
        # Per stored row: index of the last event time <= its own time (-1 when none).
        self._last_event_time = (np.cumsum(is_event_time) - 1)[time_slot]
        self._event_rows = np.flatnonzero(sorted_event)
        self._delta = sorted_event.astype(np.float64)
        self._n = n
        self._alpha = float(alpha)

    def _risk_set_sums(self, risk: NDArray[np.float64]) -> NDArray[np.float64]:
        """``S0_g``: the sum of ``risk`` over the risk set of every event time."""
        return np.cumsum(risk[::-1])[::-1][self._risk_start]

    def value(self, w: NDArray[np.float64]) -> float:
        """Objective ``L`` at the coefficients ``w``.

        ``L(w) = (1/n) sum_g [d_g log S0_g - sum_{i in D_g} eta_i]
        + alpha / (2 n) * sum(w**2)``, with ``eta = z w``,
        ``S0_g = sum_{j: t_j >= v_g} exp(eta_j)`` over the distinct event times
        ``v_g`` (Breslow handling of ties), ``D_g`` the ``d_g`` patients with an
        event at ``v_g`` and ``n`` the number of patients. The penalty term is
        always included. ``exp(eta)`` is evaluated directly.

        Parameters
        ----------
        w : ndarray of float64, shape (p,)
            Coefficients; not modified.

        Returns
        -------
        float
            The value of ``L`` at ``w``.
        """
        w = np.asarray(w, dtype=np.float64)
        eta = self._z @ w
        log_risk_sums = np.log(self._risk_set_sums(np.exp(eta)))
        partial = np.sum(self._deaths * log_risk_sums) - np.sum(eta[self._event_rows])
        penalty = self._alpha / (2 * self._n) * np.sum(w * w)
        return float(partial / self._n + penalty)

    def gradient_hessian(self, w: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Gradient and Hessian of ``L`` at the coefficients ``w``.

        With ``m_g = S1_g / S0_g``, ``S1_g = sum_{j: t_j >= v_g} exp(eta_j) z_j`` and
        ``Lambda_j = sum_{g: v_g <= t_j} d_g / S0_g`` (the Breslow cumulative
        hazard at the patient's own time),

        * gradient: ``(1/n) z^T (exp(eta) * Lambda - delta) + (alpha / n) w``;
        * Hessian: ``(1/n) [z^T diag(exp(eta) * Lambda) z - sum_g d_g m_g m_g^T]
          + (alpha / n) I``,

        which equal ``(1/n) sum_g [d_g m_g - sum_{i in D_g} z_i]`` and
        ``(1/n) sum_g d_g [S2_g / S0_g - m_g m_g^T]`` plus the penalty terms. The
        penalty terms are left out when ``alpha`` is below the float64 machine
        epsilon. The matrix products are accumulated over blocks of rows, so no
        temporary matrix is larger than a block of rows of ``z``. The Hessian is
        symmetric up to rounding.

        Parameters
        ----------
        w : ndarray of float64, shape (p,)
            Coefficients; not modified.

        Returns
        -------
        gradient : ndarray of float64, shape (p,)
        hessian : ndarray of float64, shape (p, p)
        """
        w = np.asarray(w, dtype=np.float64)
        z = self._z
        n, p = z.shape
        eta = z @ w
        risk = np.exp(eta)
        s0 = self._risk_set_sums(risk)
        cumulative_hazard = np.cumsum(self._deaths / s0)
        hazard = np.zeros(n)
        reached = self._last_event_time >= 0
        hazard[reached] = cumulative_hazard[self._last_event_time[reached]]
        expected = risk * hazard
        gradient = (z.T @ (expected - self._delta)) / n

        # Rows before the first event time belong to no risk set and have Lambda = 0.
        lowest = int(self._risk_start[0]) if self._risk_start.shape[0] else n
        block_rows = max(1, _BLOCK_ENTRIES // max(p, 1))
        row_moment = np.zeros((p, p))  # sum_j exp(eta_j) Lambda_j z_j z_j^T
        mean_moment = np.zeros((p, p))  # sum_g d_g m_g m_g^T
        later = np.zeros(p)  # sum of exp(eta_j) z_j over the rows after the current block
        stop, g_stop = n, self._risk_start.shape[0]
        while stop > lowest:
            start = max(lowest, stop - block_rows)
            rows = z[start:stop]
            scaled_rows = rows * np.sqrt(expected[start:stop])[:, None]
            row_moment += scaled_rows.T @ scaled_rows
            # Within-block sums from each row to the block end, plus everything later.
            tail = np.cumsum((rows * risk[start:stop, None])[::-1], axis=0)[::-1]
            g_start = int(np.searchsorted(self._risk_start, start, side="left"))
            if g_start < g_stop:
                s1 = tail[self._risk_start[g_start:g_stop] - start] + later
                means = s1 / s0[g_start:g_stop, None]
                scaled_means = means * np.sqrt(self._deaths[g_start:g_stop])[:, None]
                mean_moment += scaled_means.T @ scaled_means
            later = later + tail[0]
            stop, g_stop = start, g_start
        hessian = (row_moment - mean_moment) / n
        if self._alpha >= _PENALTY_EPSILON:
            gradient = gradient + (self._alpha / n) * w
            hessian[np.diag_indices(p)] += self._alpha / n
        return gradient, hessian


def _fit_newton(objective: _NewtonObjective, n_features: int, max_iter: int, tol: float) -> Tuple[NDArray[np.float64], int]:
    """Newton-Raphson minimization of ``objective`` with step halving (module docstring).

    Starts from ``w = 0`` with the reference value ``L_ref = inf``; the anchor
    is the point from which the current point was reached by the last accepted
    step (initially 0). Every pass solves the linear system ``H d = g`` at the
    current point ``w`` and evaluates the candidate ``c = w - d``.

    * If ``L(c) > L_ref``, the candidate is rejected: the current point moves to
      ``(anchor + w) / 2`` and ``L_ref`` becomes ``L`` there.
    * Otherwise the anchor becomes ``w`` and the current point ``c``. The fit has
      converged when ``|1 - L(c) / L_ref| < tol`` with ``L_ref`` from before the
      pass; if not, ``L_ref`` becomes ``L(c)``.

    Rejected passes count toward ``max_iter``. After convergence the maximum
    absolute gradient must be below ``1e-5``.

    Parameters
    ----------
    objective : _NewtonObjective
        Provides ``value(w)`` and ``gradient_hessian(w)``.
    n_features : int
        Number of coefficients ``p``.
    max_iter : int
        Maximum number of passes, rejected passes included.
    tol : float
        Relative tolerance of the stopping rule.

    Returns
    -------
    coef : ndarray of float64, shape (p,)
        Final point.
    n_iter : int
        Number of passes made, including the converging pass and every rejected
        pass.

    Raises
    ------
    ValueError
        If a Newton step contains a NaN or infinite entry.
    RuntimeError
        If the stopping rule is not met within ``max_iter`` passes, or if the
        maximum absolute gradient at the final point is not finite or not below
        ``1e-5``.
    numpy.linalg.LinAlgError
        Propagated from the linear solver, for example for an exactly singular
        Hessian.
    """
    point = np.zeros(n_features, dtype=np.float64)
    anchor = np.zeros(n_features, dtype=np.float64)
    reference = np.float64(np.inf)
    passes = 0
    converged = False
    while passes < max_iter:
        passes += 1
        gradient, hessian = objective.gradient_hessian(point)
        step = solve(hessian, gradient)
        if not np.all(np.isfinite(step)):
            raise ValueError("the Newton step is not finite")
        candidate = point - step
        candidate_value = np.float64(objective.value(candidate))
        if float(candidate_value) > float(reference):
            point = (anchor + point) / 2
            reference = np.float64(objective.value(point))
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            relative_change = np.abs(1.0 - candidate_value / reference)
        anchor, point = point, candidate
        if relative_change < tol:
            converged = True
            break
        reference = candidate_value
    if not converged:
        raise RuntimeError(f"the Newton-Raphson iteration did not meet the stopping rule within {max_iter} passes")
    final_gradient = np.asarray(objective.gradient_hessian(point)[0], dtype=np.float64)
    with np.errstate(invalid="ignore"):
        largest = float(np.max(np.abs(final_gradient))) if final_gradient.size else 0.0
    if not (np.isfinite(largest) and largest < _NEWTON_GRADIENT_TOL):
        raise RuntimeError(
            f"the Newton-Raphson iteration stopped with maximum absolute gradient {largest:.3g}, "
            f"not below {_NEWTON_GRADIENT_TOL:g}"
        )
    return point, passes


class _LogSumExpObjective:
    """Objective ``L`` and its gradient in running log-sum-exp form (rows by descending time)."""

    def __init__(self, z: NDArray[np.float64], event: NDArray[np.bool_], time: NDArray[np.float64], alpha: float):
        order = np.argsort(-time, kind="stable")
        self.x = z[order]
        self.e = event[order].astype(float)
        sorted_time = time[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_time) != 0) + 1]
        ends = np.r_[starts[1:] - 1, len(sorted_time) - 1]
        deaths = np.add.reduceat(self.e, starts)
        self.ends = ends[deaths > 0]
        self.d = deaths[deaths > 0]
        self.logd = np.log(self.d)
        self.n = len(sorted_time)
        self.alpha = float(alpha)

    def __call__(self, w: NDArray[np.float64]) -> Tuple[float, NDArray[np.float64]]:
        eta = self.x @ w
        log_risk = np.logaddexp.accumulate(eta)
        value = (self.d @ log_risk[self.ends] - self.e @ eta + 0.5 * self.alpha * (w @ w)) / self.n
        a = np.full(self.n, -np.inf)
        a[self.ends] = self.logd - log_risk[self.ends]
        suffix = np.logaddexp.accumulate(a[::-1])[::-1]
        grad = (self.x.T @ (np.exp(eta + suffix) - self.e) + self.alpha * w) / self.n
        if not (np.isfinite(value) and np.isfinite(grad).all()):
            raise FloatingPointError("non-finite Cox objective or gradient")
        return float(value), grad


def _fit_lbfgs(objective: _LogSumExpObjective, n_features: int) -> Tuple[NDArray[np.float64], int]:
    result = minimize(objective, np.zeros(n_features), jac=True, method="L-BFGS-B", options=dict(_LBFGS_OPTIONS))
    _, gradient = objective(result.x)
    max_gradient = float(np.max(np.abs(gradient)))
    if result.status not in (0, 1) or not max_gradient < _LBFGS_GRADIENT_TOL:
        raise RuntimeError(
            f"L-BFGS-B failed (status {result.status}: {result.message}); maximum absolute gradient {max_gradient:.3g}"
        )
    return result.x, int(result.nit)


def fit_penalized_cox(
    z: ArrayLike,
    event: ArrayLike,
    time: ArrayLike,
    alpha: float,
    *,
    solver: str = "newton",
    max_iter: int = 200,
    tol: float = 1e-9,
) -> CoxModel:
    """Fit the L2-penalized Breslow Cox model and its Breslow baseline hazard.

    Parameters
    ----------
    z:
        Transformed representations of the Cox-fit patients (patients x features),
        e.g. from :class:`RepresentationTransform`.
    event, time:
        Right-censored outcome of the Cox-fit patients.
    alpha:
        Penalty strength (``>= 0``) in ``alpha / (2 n) ||theta||^2``; no default.
    solver:
        ``"newton"`` or ``"lbfgs"`` (module docstring).
    max_iter, tol:
        Iteration limit and relative-objective stopping tolerance of the
        ``"newton"`` solver; ignored by ``"lbfgs"``.
    """
    z = _check_features(z, name="z")
    event, time = _check_outcome(event, time)
    if z.shape[0] != time.shape[0]:
        raise ValueError("z must have one row per patient")
    if z.shape[0] < 2:
        raise ValueError("at least two patients are required")
    if not event.any():
        raise ValueError("all Cox-fit patients are censored")
    alpha = float(alpha)
    if not alpha >= 0:
        raise ValueError("alpha must be non-negative")
    if solver == "newton":
        coef, n_iter = _fit_newton(_NewtonObjective(z, event, time, alpha), z.shape[1], max_iter, tol)
    elif solver == "lbfgs":
        coef, n_iter = _fit_lbfgs(_LogSumExpObjective(z, event, time, alpha), z.shape[1])
    else:
        raise ValueError("solver must be 'newton' or 'lbfgs'")
    unique_times, cumulative_hazard = breslow_cumulative_hazard(np.dot(z, coef), event, time)
    return CoxModel(
        coef=coef,
        alpha=alpha,
        unique_times=unique_times,
        cumulative_baseline_hazard=cumulative_hazard,
        solver=solver,
        n_iter=n_iter,
    )


# ---------------------------------------------------------------------------
# Penalty selection
# ---------------------------------------------------------------------------
def ipcw_horizon_log_loss(
    log_cumulative_hazard: ArrayLike,
    event: ArrayLike,
    time: ArrayLike,
    train_event: ArrayLike,
    train_time: ArrayLike,
    horizon: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-patient IPCW log-loss of the predicted risk at ``horizon``.

    With ``c_i = log H_0(H) + eta_i`` and predicted risk ``F_i = 1 - exp(-exp(c_i))``:

    * event by the horizon (``event_i`` and ``time_i <= H``): weight
      ``1 / G(time_i)``, loss ``-log F_i`` (evaluated as ``-c_i`` when ``c_i < -20``);
    * follow-up beyond the horizon (``time_i > H``): weight ``1 / G(H)``, loss
      ``exp(c_i) = -log(1 - F_i)``;
    * censored at or before the horizon: weight 0, loss 0.

    ``exp(c_i)`` is evaluated with ``c_i`` capped at 700. ``G`` is the Kaplan-Meier
    censoring survival function of the reference sample (``train_event``,
    ``train_time``), evaluated as described in :mod:`ecg_age_retention.evaluation.survival`.

    Returns
    -------
    weighted_loss, weight:
        ``weight * loss`` and ``weight`` per patient.
    """
    event, time = _check_outcome(event, time, "validation")
    train_event, train_time = _check_outcome(train_event, train_time, "reference")
    c = np.asarray(log_cumulative_hazard, dtype=np.float64)
    if c.ndim != 1 or c.shape[0] != time.shape[0]:
        raise ValueError("log_cumulative_hazard must have one value per patient")
    censoring = _CensoringSurvival(train_event, train_time)
    died = event & (time <= horizon)
    survived = time > horizon
    weight = np.zeros(time.shape[0])
    with np.errstate(divide="ignore"):
        if died.any():
            weight[died] = 1 / censoring(time[died])
        if survived.any():
            weight[survived] = 1 / float(censoring(np.array([horizon], dtype=np.float64))[0])
    if not (np.all(np.isfinite(weight)) and np.all(weight >= 0)):
        raise ValueError("inverse probability of censoring weights are not finite")
    loss = np.zeros(time.shape[0])
    c_died = c[died]
    positive = np.empty_like(c_died)
    small = c_died < -20
    positive[small] = -c_died[small]
    positive[~small] = -np.log(-np.expm1(-np.exp(np.minimum(c_died[~small], 700))))
    loss[died] = positive
    loss[survived] = np.exp(np.minimum(c[survived], 700))
    if not np.all(np.isfinite(loss)):
        raise FloatingPointError("non-finite horizon log-loss")
    return weight * loss, weight


class PenaltySelection(NamedTuple):
    """Output of :func:`select_cox_penalty`.

    Attributes
    ----------
    alpha:
        Selected penalty.
    alphas:
        Candidate penalties in grid order.
    mean_loss:
        Selection criterion per candidate: mean IPCW-weighted horizon log-loss over
        all Cox-fit patients.
    weighted_loss:
        Per-patient weighted loss on its validation fold, shape ``(n, n_candidates)``.
    ipcw_weight:
        Per-patient IPCW weight on its validation fold.
    fold:
        Validation fold index of every patient.
    """

    alpha: float
    alphas: Tuple[float, ...]
    mean_loss: NDArray[np.float64]
    weighted_loss: NDArray[np.float64]
    ipcw_weight: NDArray[np.float64]
    fold: NDArray[np.int64]


def select_cox_penalty(
    x: ArrayLike,
    event: ArrayLike,
    time: ArrayLike,
    horizon: float,
    *,
    random_state: int,
    alphas: Sequence[float] = DEFAULT_PENALTY_GRID,
    n_splits: int = 3,
    max_iter: int = 200,
    tol: float = 1e-9,
) -> PenaltySelection:
    """Select the Cox penalty by cross-validated IPCW horizon log-loss.

    The patients are assigned to folds by
    ``KFold(n_splits, shuffle=True, random_state=random_state)``. For each fold, a
    ``StandardScaler`` is fitted on the training part and applied
    to both parts, a ``"newton"`` Cox model is fitted on the training part for every
    candidate in ``alphas``, and the validation patients are scored with
    :func:`ipcw_horizon_log_loss` (censoring distribution from the training part).
    The candidate with the smallest mean weighted loss over all patients is
    selected (the first in grid order on exact ties).

    Parameters
    ----------
    x:
        Untransformed representations of the Cox-fit patients (patients x features).
    event, time:
        Right-censored outcome of the Cox-fit patients.
    horizon:
        Prediction horizon ``H``.
    random_state:
        Integer seed of the fold shuffling; no default.
    alphas:
        Candidate penalties (default :data:`DEFAULT_PENALTY_GRID`).
    n_splits:
        Number of folds (default 3).
    max_iter, tol:
        Settings of the ``"newton"`` solver.
    """
    x = _check_features(x)
    event, time = _check_outcome(event, time)
    if x.shape[0] != time.shape[0]:
        raise ValueError("x must have one row per patient")
    grid = tuple(float(a) for a in alphas)
    if len(grid) == 0:
        raise ValueError("at least one candidate penalty is required")
    n = x.shape[0]
    weighted_loss = np.full((n, len(grid)), np.nan)
    fold = np.full(n, -1, dtype=np.int64)
    ipcw_weight = np.full(n, np.nan)
    kfold = KFold(n_splits, shuffle=True, random_state=random_state)
    for k, (train, valid) in enumerate(kfold.split(x)):
        scaler = StandardScaler().fit(x[train])
        z_train, z_valid = scaler.transform(x[train]), scaler.transform(x[valid])
        for j, alpha in enumerate(grid):
            model = fit_penalized_cox(z_train, event[train], time[train], alpha, solver="newton", max_iter=max_iter, tol=tol)
            log_cumulative_hazard = model.predict(z_valid, horizon).log_cumulative_hazard
            loss, weight = ipcw_horizon_log_loss(
                log_cumulative_hazard, event[valid], time[valid], event[train], time[train], horizon
            )
            weighted_loss[valid, j] = loss
            fold[valid] = k
            ipcw_weight[valid] = weight
    if not (np.all(np.isfinite(weighted_loss)) and np.all(fold >= 0)):
        raise RuntimeError("the cross-validation did not score every patient")
    mean_loss = weighted_loss.mean(axis=0)
    return PenaltySelection(
        alpha=grid[int(np.argmin(mean_loss))],
        alphas=grid,
        mean_loss=mean_loss,
        weighted_loss=weighted_loss,
        ipcw_weight=ipcw_weight,
        fold=fold,
    )


# ---------------------------------------------------------------------------
# Representation transform and complete readout
# ---------------------------------------------------------------------------
class RepresentationTransform:
    """Standardization, optional PCA cap and re-standardization fitted on Cox-fit patients.

    Parameters
    ----------
    max_dim:
        Representations with more than ``max_dim`` columns are reduced by PCA to
        ``max_dim`` components (scikit-learn ``PCA`` with its default solver choice)
        after standardization, and the components are standardized again. ``None``
        disables the reduction. Default 256.
    random_state:
        Integer seed of the PCA solver; required when the reduction is applied.
    """

    def __init__(self, max_dim: Optional[int] = 256, random_state: Optional[int] = None) -> None:
        self.max_dim = max_dim
        self.random_state = random_state

    def fit(self, x: ArrayLike) -> "RepresentationTransform":
        """Fit the transform on the Cox-fit representations ``x`` (patients x features)."""
        x = _check_features(x)
        self.scaler_ = StandardScaler().fit(x)
        standardized = self.scaler_.transform(x)
        if self.max_dim is not None and x.shape[1] > self.max_dim:
            if self.random_state is None:
                raise ValueError("random_state is required when the PCA reduction is applied")
            self.pca_: Optional[PCA] = PCA(n_components=self.max_dim, random_state=self.random_state).fit(standardized)
            self.rescaler_: Optional[StandardScaler] = StandardScaler().fit(self.pca_.transform(standardized))
        else:
            self.pca_ = None
            self.rescaler_ = None
        self.n_features_in_ = x.shape[1]
        return self

    def transform(self, x: ArrayLike) -> NDArray[np.float64]:
        """Apply the fitted transform to representations ``x``."""
        if not hasattr(self, "scaler_"):
            raise RuntimeError("the transform is not fitted")
        x = _check_features(x)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"x has {x.shape[1]} columns, the transform expects {self.n_features_in_}")
        z = self.scaler_.transform(x)
        if self.pca_ is not None:
            z = self.rescaler_.transform(self.pca_.transform(z))
        return z


@dataclass(frozen=True)
class CoxReadout:
    """Fitted readout: representation transform followed by a Cox model."""

    transform: RepresentationTransform
    model: CoxModel

    def predict(self, x: ArrayLike, horizon: float) -> CoxPrediction:
        """Transform untransformed representations ``x`` and predict risk at ``horizon`` (eq. 5)."""
        return self.model.predict(self.transform.transform(x), horizon)


def fit_cox_readout(
    x: ArrayLike,
    event: ArrayLike,
    time: ArrayLike,
    *,
    alpha: float,
    solver: str = "newton",
    max_dim: Optional[int] = 256,
    random_state: Optional[int] = None,
    max_iter: int = 200,
    tol: float = 1e-9,
) -> CoxReadout:
    """Fit the complete readout on the Cox-fit patients.

    Fits :class:`RepresentationTransform` on ``x``, transforms ``x`` and fits
    :func:`fit_penalized_cox` with penalty ``alpha``. Apply the result unchanged to
    evaluation patients with :meth:`CoxReadout.predict`.

    Parameters
    ----------
    x:
        Untransformed representations of the Cox-fit patients (patients x features).
    event, time:
        Right-censored outcome of the Cox-fit patients.
    alpha:
        Penalty strength; no default (e.g. ``select_cox_penalty(...).alpha``).
    solver:
        ``"newton"`` or ``"lbfgs"``.
    max_dim, random_state:
        PCA cap of :class:`RepresentationTransform`.
    max_iter, tol:
        Settings of the ``"newton"`` solver.
    """
    transform = RepresentationTransform(max_dim=max_dim, random_state=random_state).fit(x)
    model = fit_penalized_cox(transform.transform(x), event, time, alpha, solver=solver, max_iter=max_iter, tol=tol)
    return CoxReadout(transform=transform, model=model)
