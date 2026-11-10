"""Global age-R2 (eq. 4), the nested age probe, RankMe and rank agreement (eq. 8)."""

import unittest

import numpy as np

from ecg_age_retention.representation import (
    DEFAULT_ALPHA_GRID,
    global_age_r2,
    nested_age_probe,
    rank_agreement,
    rankme,
    ridge_age_predictions,
)


def probe_data(n_patients, signal, seed):
    """Synthetic representations with one to three ECGs per patient."""
    rng = np.random.default_rng(seed)
    ecgs = rng.integers(1, 4, n_patients)
    groups = np.repeat(np.arange(n_patients), ecgs)
    age = np.repeat(rng.uniform(20.0, 90.0, n_patients), ecgs)
    x = rng.standard_normal((groups.size, 8))
    x[:, 0] += signal * (age - age.mean()) / age.std()
    return x, age, groups


class GlobalAgeR2Tests(unittest.TestCase):
    def test_definition(self):
        age = np.array([1.0, 2.0, 3.0])
        self.assertEqual(global_age_r2(age, age), 1.0)
        self.assertEqual(global_age_r2(age, np.full(3, 2.0)), 0.0)
        # SSE = 1, total sum of squares = 2.
        self.assertAlmostEqual(global_age_r2(age, [1.0, 2.0, 4.0]), 0.5)
        self.assertLess(global_age_r2(age, [3.0, 2.0, 1.0]), 0.0)


class RidgeAgePredictionTests(unittest.TestCase):
    def test_penalty_grid_limits(self):
        rng = np.random.default_rng(1)
        age = rng.uniform(20.0, 90.0, 50)
        features = np.column_stack([age / 10.0, rng.standard_normal(50)])
        predictions = ridge_age_predictions(features, age, features[:5], alphas=[1e-8, 1e12])
        self.assertEqual(predictions.shape, (5, 2))
        np.testing.assert_allclose(predictions[:, 0], age[:5], atol=1e-4)  # almost no penalty: exact fit
        np.testing.assert_allclose(predictions[:, 1], age.mean(), atol=1e-6)  # huge penalty: mean age

    def test_default_grid(self):
        self.assertEqual(DEFAULT_ALPHA_GRID.shape, (11,))


class NestedAgeProbeTests(unittest.TestCase):
    def test_inner_folds_is_required(self):
        x, age, groups = probe_data(30, 1.0, seed=2)
        with self.assertRaises(TypeError):
            nested_age_probe(x, age, groups)

    def test_out_of_fold_structure(self):
        x, age, groups = probe_data(120, 1.0, seed=3)
        result = nested_age_probe(x, age, groups, inner_folds=3, outer_folds=5)
        self.assertEqual(result.oof_prediction.shape, age.shape)
        self.assertTrue(np.all(np.isfinite(result.oof_prediction)))
        self.assertEqual(len(result.selected_alpha), 5)
        self.assertTrue(set(result.selected_alpha) <= set(DEFAULT_ALPHA_GRID.tolist()))
        np.testing.assert_array_equal(np.unique(result.outer_fold), np.arange(5))
        for patient in np.unique(groups):  # all ECGs of a patient share one outer fold
            self.assertEqual(np.unique(result.outer_fold[groups == patient]).size, 1)

    def test_signal_and_no_signal(self):
        x, age, groups = probe_data(300, 3.0, seed=4)
        strong = global_age_r2(age, nested_age_probe(x, age, groups, inner_folds=3).oof_prediction)
        x, age, groups = probe_data(300, 0.0, seed=5)
        none = global_age_r2(age, nested_age_probe(x, age, groups, inner_folds=3).oof_prediction)
        self.assertGreater(strong, 0.8)
        self.assertLess(none, 0.05)


class RankMeTests(unittest.TestCase):
    def test_equal_spectrum_gives_dimension(self):
        for k in (1, 2, 5):
            x = np.vstack([np.eye(k), -np.eye(k)])  # centred, Z^T Z = 2 I
            self.assertAlmostEqual(rankme(x), k, places=5)

    def test_rank_one(self):
        x = np.outer([1.0, -1.0, 2.0, -2.0], [3.0, 0.0, 0.0])  # Z^T Z = diag(90, 0, 0)
        self.assertAlmostEqual(rankme(x), 1.0, places=5)

    def test_constant_matrix(self):
        self.assertEqual(rankme(np.full((4, 3), 7.0)), 1.0)

    def test_columns_are_centred(self):
        rng = np.random.default_rng(6)
        x = rng.standard_normal((40, 6))
        self.assertAlmostEqual(rankme(x + 100.0), rankme(x), places=8)

    def test_unequal_spectrum(self):
        # Singular values 2 and 1: p = (2/3, 1/3), RankMe = exp(entropy).
        x = np.vstack([np.diag([2.0, 1.0]) / np.sqrt(2), -np.diag([2.0, 1.0]) / np.sqrt(2)])
        p = np.array([2.0, 1.0]) / 3.0
        self.assertAlmostEqual(rankme(x), float(np.exp(-np.sum(p * np.log(p)))), places=6)


class RankAgreementTests(unittest.TestCase):
    def test_perfect_agreement_and_disagreement(self):
        metric = [0.1, 0.2, 0.3, 0.4]
        self.assertAlmostEqual(rank_agreement(metric, [40.0, 30.0, 20.0, 10.0]), 1.0)
        self.assertAlmostEqual(rank_agreement(metric, [10.0, 20.0, 30.0, 40.0]), -1.0)

    def test_lower_is_better_metric(self):
        # |Q_full| grows with older-patient error: agreement once the metric is negated.
        self.assertAlmostEqual(rank_agreement([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0], higher_is_better=False), 1.0)

    def test_ties_receive_average_ranks(self):
        # Ranks (1.5, 1.5, 3) against (1, 2, 3): Pearson correlation of ranks = 1.5 / sqrt(1.5 * 2).
        self.assertAlmostEqual(rank_agreement([1.0, 1.0, 2.0], [3.0, 2.0, 1.0]), 1.5 / np.sqrt(3.0))


if __name__ == "__main__":
    unittest.main()
