"""Cox readout: Breslow estimator, predicted risk (eq. 5), penalized fit and penalty selection."""

import unittest

import numpy as np
from scipy.optimize import brentq

from ecg_age_retention.evaluation import (
    DEFAULT_PENALTY_GRID,
    CoxModel,
    RepresentationTransform,
    breslow_cumulative_hazard,
    fit_cox_readout,
    fit_penalized_cox,
    ipcw_horizon_log_loss,
    select_cox_penalty,
)


def simulated_outcomes(n, p, seed):
    """Exponential event times with administrative censoring (synthetic)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    coef = np.linspace(0.8, -0.4, p)
    event_time = rng.exponential(1.0 / (0.1 * np.exp(x @ coef)))
    censor_time = rng.uniform(0.5, 10.0, n)
    event = event_time <= censor_time
    return x, event, np.minimum(event_time, censor_time)


def breslow_objective(coef, x, event, time, alpha):
    """Mean negative Breslow log partial likelihood plus alpha / (2 n) ||coef||^2 (direct loops)."""
    eta = x @ coef
    total = 0.0
    for t in np.unique(time[event]):
        dying = event & (time == t)
        total += dying.sum() * np.log(np.exp(eta[time >= t]).sum()) - eta[dying].sum()
    return total / len(time) + alpha * coef @ coef / (2 * len(time))


class BreslowTests(unittest.TestCase):
    def test_nelson_aalen_when_linear_predictor_is_zero(self):
        times, hazard = breslow_cumulative_hazard(np.zeros(4), [1, 0, 1, 0], [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(times, [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(hazard, [1 / 4, 1 / 4, 1 / 4 + 1 / 2, 3 / 4])

    def test_weighted_risk_sets(self):
        # exp(eta) = 1, 2, 1, 1: risk-set sums 5 at t=1 and 2 at t=3.
        _, hazard = breslow_cumulative_hazard([0.0, np.log(2.0), 0.0, 0.0], [1, 0, 1, 0], [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(hazard, [0.2, 0.2, 0.7, 0.7])

    def test_tied_deaths(self):
        times, hazard = breslow_cumulative_hazard(np.zeros(3), [1, 1, 0], [1.0, 1.0, 2.0])
        np.testing.assert_array_equal(times, [1.0, 2.0])
        np.testing.assert_allclose(hazard, [2 / 3, 2 / 3])


class PredictedRiskTests(unittest.TestCase):
    model = CoxModel(
        coef=np.array([0.5]),
        alpha=1.0,
        unique_times=np.array([1.0, 2.0, 3.0]),
        cumulative_baseline_hazard=np.array([0.1, 0.2, 0.4]),
        solver="newton",
        n_iter=1,
    )

    def test_eq5(self):
        prediction = self.model.predict([[0.0], [2.0]], 2.5)
        np.testing.assert_allclose(prediction.linear_predictor, [0.0, 1.0])
        np.testing.assert_allclose(prediction.risk, 1 - np.exp(-0.2 * np.exp([0.0, 1.0])))
        np.testing.assert_allclose(prediction.log_cumulative_hazard, np.log(0.2) + np.array([0.0, 1.0]))

    def test_horizon_before_first_time_is_rejected(self):
        with self.assertRaises(ValueError):
            self.model.predict([[0.0]], 0.5)


class PenalizedCoxTests(unittest.TestCase):
    def test_two_patient_optimum(self):
        # Patient A (z=1) dies at t=1, patient B (z=0) is censored at t=2. The penalized
        # objective (log(e^b + 1) - b) / 2 + alpha b^2 / 4 is minimal where alpha b = 1 / (1 + e^b).
        for alpha in (0.5, 1.0, 4.0):
            expected = brentq(lambda b: alpha * b - 1 / (1 + np.exp(b)), 0.0, 10.0, xtol=1e-14)
            for solver in ("newton", "lbfgs"):
                model = fit_penalized_cox([[1.0], [0.0]], [1, 0], [1.0, 2.0], alpha, solver=solver)
                self.assertAlmostEqual(model.coef[0], expected, places=7, msg=f"alpha={alpha}, {solver}")

    def test_minimizes_the_penalized_breslow_objective(self):
        x, event, time = simulated_outcomes(300, 3, seed=1)
        time = np.round(time, 1)  # create tied times
        alpha = 2.0
        model = fit_penalized_cox(x, event, time, alpha)
        f0 = breslow_objective(model.coef, x, event, time, alpha)
        step = 1e-5
        for k in range(3):
            shift = np.zeros(3)
            shift[k] = step
            slope = (breslow_objective(model.coef + shift, x, event, time, alpha)
                     - breslow_objective(model.coef - shift, x, event, time, alpha)) / (2 * step)
            self.assertLess(abs(slope), 1e-6)
            self.assertGreater(breslow_objective(model.coef + 100 * shift, x, event, time, alpha), f0)

    def test_newton_and_lbfgs_agree(self):
        x, event, time = simulated_outcomes(500, 4, seed=2)
        newton = fit_penalized_cox(x, event, time, 1.0, solver="newton")
        lbfgs = fit_penalized_cox(x, event, time, 1.0, solver="lbfgs")
        np.testing.assert_allclose(newton.coef, lbfgs.coef, atol=1e-5)

    def test_penalty_is_required(self):
        x, event, time = simulated_outcomes(50, 2, seed=3)
        with self.assertRaises(TypeError):
            fit_penalized_cox(x, event, time)
        with self.assertRaises(TypeError):
            fit_cox_readout(x, event, time)
        with self.assertRaises(ValueError):
            fit_penalized_cox(x, event, time, -1.0)


class IPCWHorizonLogLossTests(unittest.TestCase):
    def test_weights_and_losses(self):
        # Reference sample without censoring: all weights 1. Horizon 2.
        weighted, weight = ipcw_horizon_log_loss(
            [0.0, 0.0, 0.0, -25.0], [1, 0, 0, 1], [1.0, 3.0, 1.5, 0.5], np.ones(10, bool), np.arange(1.0, 11.0), 2.0
        )
        np.testing.assert_allclose(weight, [1.0, 1.0, 0.0, 1.0])
        # Death by H: -log(1 - exp(-exp(c))); alive at H: exp(c); censored before H: 0; c < -20: -c.
        np.testing.assert_allclose(weighted, [-np.log(1 - np.exp(-1.0)), 1.0, 0.0, 25.0])


class ReadoutTests(unittest.TestCase):
    def test_penalty_selection(self):
        x, event, time = simulated_outcomes(400, 4, seed=4)
        selection = select_cox_penalty(x, event, time, 3.0, random_state=1)
        self.assertIn(selection.alpha, DEFAULT_PENALTY_GRID)
        self.assertEqual(selection.alphas, DEFAULT_PENALTY_GRID)
        self.assertEqual(selection.mean_loss.shape, (len(DEFAULT_PENALTY_GRID),))
        self.assertEqual(selection.alpha, DEFAULT_PENALTY_GRID[int(np.argmin(selection.mean_loss))])
        np.testing.assert_array_equal(np.unique(selection.fold), [0, 1, 2])
        np.testing.assert_allclose(selection.mean_loss, selection.weighted_loss.mean(axis=0))

    def test_random_state_is_required(self):
        x, event, time = simulated_outcomes(60, 2, seed=5)
        with self.assertRaises(TypeError):
            select_cox_penalty(x, event, time, 3.0)

    def test_transform_and_prediction(self):
        x, event, time = simulated_outcomes(300, 6, seed=6)
        x = 5.0 + 3.0 * x
        readout = fit_cox_readout(x, event, time, alpha=1.0)
        z = readout.transform.transform(x)
        np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(z.std(axis=0), 1.0, atol=1e-12)
        prediction = readout.predict(x[:10], 3.0)
        self.assertEqual(prediction.risk.shape, (10,))
        self.assertTrue(np.all((prediction.risk > 0) & (prediction.risk < 1)))

    def test_pca_cap(self):
        x, event, time = simulated_outcomes(200, 10, seed=7)
        with self.assertRaises(ValueError):
            RepresentationTransform(max_dim=4).fit(x)
        transform = RepresentationTransform(max_dim=4, random_state=1).fit(x)
        z = transform.transform(x)
        self.assertEqual(z.shape, (200, 4))
        np.testing.assert_allclose(z.std(axis=0), 1.0, atol=1e-12)
        readout = fit_cox_readout(x, event, time, alpha=1.0, solver="lbfgs", max_dim=4, random_state=1)
        self.assertEqual(readout.model.coef.shape, (4,))


if __name__ == "__main__":
    unittest.main()
