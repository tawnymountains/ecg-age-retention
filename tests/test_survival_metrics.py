"""Survival metrics on small cases that can be checked by hand."""

import unittest

import numpy as np

from ecg_age_retention.evaluation import (
    DEFAULT_AGE_BANDS,
    OLDER_AGE_THRESHOLD,
    age_band_calibration,
    calibration_error,
    calibration_summary,
    ipcw_brier_score,
    kaplan_meier_risk,
    uno_c,
    weighted_calibration_error,
)

# Censoring reference with G = 1 at t = 1, 0.75 at t = 2 and 3, 0.375 at t = 4 and 5.
REF_EVENT = np.array([1, 0, 1, 0, 1])
REF_TIME = np.array([1.0, 2.0, 3.0, 4.0, 5.0])


class KaplanMeierRiskTests(unittest.TestCase):
    def test_product_limit_estimate(self):
        # Deaths at t=1 (4 at risk) and t=3 (2 at risk); censored at t=2 and t=4.
        event, time = [1, 0, 1, 0], [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(kaplan_meier_risk(event, time, 0.5), 0.0)
        self.assertAlmostEqual(kaplan_meier_risk(event, time, 1.0), 0.25)
        self.assertAlmostEqual(kaplan_meier_risk(event, time, 2.5), 0.25)
        self.assertAlmostEqual(kaplan_meier_risk(event, time, 3.0), 1 - 0.75 * 0.5)
        self.assertAlmostEqual(kaplan_meier_risk(event, time, 10.0), 0.625)

    def test_censored_at_death_time_is_still_at_risk(self):
        # At t=2 one death and one censoring: 3 patients at risk, S(2) = 3/4 * 2/3.
        self.assertAlmostEqual(kaplan_meier_risk([1, 1, 0, 0], [1.0, 2.0, 2.0, 3.0], 2.0), 0.5)

    def test_tied_deaths(self):
        # Two deaths at t=1 among 4 patients, one more at t=2 among the remaining 2.
        self.assertAlmostEqual(kaplan_meier_risk([1, 1, 1, 0], [1.0, 1.0, 2.0, 3.0], 2.0), 1 - 0.5 * 0.5)

    def test_without_censoring_equals_fraction_of_deaths(self):
        time = np.arange(1.0, 11.0)
        self.assertAlmostEqual(kaplan_meier_risk(np.ones(10, bool), time, 4.0), 0.4)

    def test_rejects_invalid_outcomes(self):
        with self.assertRaises(ValueError):
            kaplan_meier_risk([1, 2], [1.0, 2.0], 1.0)
        with self.assertRaises(ValueError):
            kaplan_meier_risk([1, 0], [1.0, -2.0], 1.0)
        with self.assertRaises(ValueError):
            kaplan_meier_risk([1, 0, 1], [1.0, 2.0], 1.0)


class CalibrationErrorTests(unittest.TestCase):
    event = np.array([1, 0, 1, 0])
    time = np.array([1.0, 2.0, 3.0, 4.0])
    risk = np.full(4, 0.2)

    def test_full_cohort(self):
        # Q = 1000 * (0.625 - 0.2)
        self.assertAlmostEqual(calibration_error(self.event, self.time, self.risk, 3.0), 425.0)

    def test_overestimation_is_negative(self):
        self.assertAlmostEqual(calibration_error(self.event, self.time, np.full(4, 0.9), 3.0), -275.0)

    def test_subset_as_mask_or_indices(self):
        # First two patients: death at t=1 among 2 at risk, so observed risk 0.5.
        mask = np.array([True, True, False, False])
        self.assertAlmostEqual(calibration_error(self.event, self.time, self.risk, 3.0, subset=mask), 300.0)
        self.assertAlmostEqual(calibration_error(self.event, self.time, self.risk, 3.0, subset=[0, 1]), 300.0)

    def test_empty_subset_is_rejected(self):
        with self.assertRaises(ValueError):
            calibration_error(self.event, self.time, self.risk, 3.0, subset=np.zeros(4, bool))


class AgeBandCalibrationTests(unittest.TestCase):
    # Band [0, 50): ages 30, 40.  Band [50, inf): ages 60, 70, 80.
    age = np.array([30.0, 40.0, 60.0, 70.0, 80.0])
    event = np.array([1, 0, 1, 1, 0])
    time = np.array([1.0, 5.0, 1.0, 3.0, 4.0])
    risk = np.array([0.1, 0.3, 0.5, 0.5, 0.5])
    bands = ((0.0, 50.0), (50.0, np.inf))

    def test_band_q_and_wce(self):
        result = age_band_calibration(self.event, self.time, self.risk, self.age, 2.0, bands=self.bands)
        np.testing.assert_array_equal(result.n, [2, 3])
        np.testing.assert_array_equal(result.n_events_by_horizon, [1, 1])
        np.testing.assert_allclose(result.observed_risk, [0.5, 1 / 3])
        np.testing.assert_allclose(result.mean_predicted_risk, [0.2, 0.5])
        np.testing.assert_allclose(result.q, [300.0, 1000 * (1 / 3 - 0.5)])
        # WCE = 2/5 * 300 + 3/5 * 166.67 = 220
        self.assertAlmostEqual(result.wce, 220.0)
        self.assertAlmostEqual(
            weighted_calibration_error(self.event, self.time, self.risk, self.age, 2.0, bands=self.bands), 220.0
        )

    def test_summary(self):
        summary = calibration_summary(self.event, self.time, self.risk, self.age, 2.0, bands=self.bands)
        # Full cohort: deaths at t=1 (2 of 5), so observed risk 0.4 by t=2; mean predicted 0.38.
        self.assertAlmostEqual(summary.q_full, 20.0)
        # Patients aged >= 75: one censored patient, observed risk 0, predicted 0.5.
        self.assertAlmostEqual(summary.q_older, -500.0)
        self.assertAlmostEqual(summary.wce, 220.0)

    def test_default_bands_are_half_open(self):
        self.assertEqual(OLDER_AGE_THRESHOLD, 75.0)
        self.assertEqual(len(DEFAULT_AGE_BANDS), 7)
        age = np.array([15, 29.9, 30, 45, 60, 74.9, 75, 79.9, 80, 85, 99.0])
        n = age.size
        result = age_band_calibration(np.zeros(n), np.ones(n), np.zeros(n), age, 1.0)
        np.testing.assert_array_equal(result.n, [2, 1, 1, 2, 2, 1, 2])
        self.assertEqual(result.wce, 0.0)

    def test_every_patient_must_fall_in_exactly_one_band(self):
        n = self.age.size
        with self.assertRaises(ValueError):  # age 30 is below the first band
            age_band_calibration(np.zeros(n), np.ones(n), np.zeros(n), self.age, 1.0, bands=((35.0, np.inf),))
        with self.assertRaises(ValueError):  # overlapping bands
            age_band_calibration(np.zeros(n), np.ones(n), np.zeros(n), self.age, 1.0, bands=((0.0, 65.0), (50.0, 100.0)))
        with self.assertRaises(ValueError):  # empty band
            age_band_calibration(
                np.zeros(n), np.ones(n), np.zeros(n), self.age, 1.0, bands=((0.0, 50.0), (50.0, 90.0), (90.0, np.inf))
            )


class UnoCTests(unittest.TestCase):
    # Reference sample without censoring, so the censoring weights are all 1.
    ref_event = np.ones(10, bool)
    ref_time = np.arange(1.0, 11.0)
    event = np.array([1, 1, 1, 0])
    time = np.array([1.0, 2.0, 3.0, 4.0])

    def c(self, score, horizon):
        return uno_c(self.ref_event, self.ref_time, self.event, self.time, score, horizon)

    def test_concordant_and_discordant_scores(self):
        self.assertAlmostEqual(self.c([4.0, 3.0, 2.0, 1.0], 10.0), 1.0)
        self.assertAlmostEqual(self.c([1.0, 2.0, 3.0, 4.0], 10.0), 0.0)

    def test_tied_scores_count_one_half(self):
        # 6 comparable pairs; the pairs (1,2) and (3,4) have tied scores.
        self.assertAlmostEqual(self.c([2.0, 2.0, 1.0, 1.0], 10.0), 5.0 / 6.0)

    def test_truncation_at_horizon(self):
        # Only deaths before the horizon start a pair: 5 pairs before 2.5, 3 pairs before 2.
        self.assertAlmostEqual(self.c([2.0, 2.0, 1.0, 1.0], 2.5), 4.5 / 5.0)
        self.assertAlmostEqual(self.c([2.0, 2.0, 1.0, 1.0], 2.0), 2.5 / 3.0)

    def test_inverse_probability_of_censoring_weights(self):
        # G(1.5) = 1 and G(2.5) = G(3.5) = 0.75, so pairs starting at t=2.5 and t=3.5 weigh 1/0.75^2.
        w = 1 / 0.75**2
        value = uno_c(REF_EVENT, REF_TIME, [1, 1, 1, 0], [1.5, 2.5, 3.5, 4.5], [4.0, 3.0, 1.0, 2.0], 10.0)
        self.assertAlmostEqual(value, (3 + 2 * w) / (3 + 3 * w))

    def test_all_censored_is_rejected(self):
        with self.assertRaises(ValueError):
            uno_c(self.ref_event, self.ref_time, [0, 0, 0, 0], self.time, [1.0, 2.0, 3.0, 4.0], 10.0)


class IPCWBrierScoreTests(unittest.TestCase):
    def test_without_censoring_weights(self):
        ref_event, ref_time = np.ones(10, bool), np.arange(1.0, 11.0)
        # Death by H: (1-0.9)^2; censored before H: 0; alive at H: 0.4^2 and 0.1^2.
        value = ipcw_brier_score(ref_event, ref_time, [1, 0, 1, 0], [1.0, 2.0, 3.0, 4.0], [0.9, 0.2, 0.4, 0.1], 2.5)
        self.assertAlmostEqual(value, (0.01 + 0.0 + 0.16 + 0.01) / 4)

    def test_with_censoring_weights(self):
        # G(1.5) = 1, G(2.5) = 0.75 and G(3) = 0.75.
        value = ipcw_brier_score(
            REF_EVENT, REF_TIME, [1, 1, 0, 1], [1.5, 2.5, 3.5, 4.5], [0.5, 0.6, 0.2, 0.3], 3.0
        )
        expected = (0.5**2 / 1.0 + 0.4**2 / 0.75 + 0.2**2 / 0.75 + 0.3**2 / 0.75) / 4
        self.assertAlmostEqual(value, expected)

    def test_horizon_must_lie_inside_follow_up(self):
        ref_event, ref_time = np.ones(10, bool), np.arange(1.0, 11.0)
        with self.assertRaises(ValueError):
            ipcw_brier_score(ref_event, ref_time, [1, 0], [1.0, 2.0], [0.5, 0.5], 2.0)


if __name__ == "__main__":
    unittest.main()
