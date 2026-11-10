"""ECG preprocessing: lead order, Fourier resampling and per-lead standardization."""

import unittest

import numpy as np

from ecg_age_retention.data import LEAD_ORDER, reorder_leads, resample_fft, standardize_leads, zero_nonfinite


class ReorderLeadsTests(unittest.TestCase):
    def test_matches_names_case_insensitively(self):
        names = ["V6", "v5", " V4 ", "V3", "V2", "V1", "avf", "AVL", "aVR", "III", "II", "I"]
        signal = np.arange(12)[::-1, None] * np.ones((1, 5))  # row i holds the value 11 - i
        out = reorder_leads(signal, names)
        self.assertEqual(out.shape, (12, 5))
        np.testing.assert_array_equal(out[:, 0], np.arange(12))
        self.assertEqual(LEAD_ORDER[:3], ("I", "II", "III"))

    def test_extra_leads_are_dropped(self):
        names = list(LEAD_ORDER) + ["VX", "VY"]
        out = reorder_leads(np.arange(14)[:, None] * np.ones((1, 3)), names)
        np.testing.assert_array_equal(out[:, 0], np.arange(12))

    def test_missing_lead_and_shape_mismatch(self):
        with self.assertRaises(ValueError):
            reorder_leads(np.zeros((11, 5)), list(LEAD_ORDER[:11]))
        with self.assertRaises(ValueError):
            reorder_leads(np.zeros((12, 5)), list(LEAD_ORDER[:11]))


class ResampleTests(unittest.TestCase):
    def test_shape_dtype_and_periodic_signal(self):
        t = np.arange(5000) / 5000.0
        signal = np.vstack([np.sin(2 * np.pi * 3 * t)] * 12)
        out = resample_fft(signal, 4096)
        self.assertEqual(out.shape, (12, 4096))
        self.assertEqual(out.dtype, np.float32)
        expected = np.sin(2 * np.pi * 3 * np.arange(4096) / 4096.0)
        np.testing.assert_allclose(out[0], expected, atol=1e-5)


class StandardizeTests(unittest.TestCase):
    def test_zero_mean_unit_variance_per_lead(self):
        rng = np.random.default_rng(1)
        x = 3.0 + 2.0 * rng.standard_normal((2, 12, 4096))
        out = standardize_leads(x)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(out.mean(axis=-1), 0.0, atol=1e-5)
        np.testing.assert_allclose(out.std(axis=-1), 1.0, atol=1e-4)

    def test_clipping_and_flat_lead(self):
        x = np.zeros((2, 1000), dtype=np.float32)
        x[0, 0] = 1000.0  # a spike far beyond 10 standard deviations
        out = standardize_leads(x)
        self.assertEqual(out[0].max(), 10.0)
        np.testing.assert_array_equal(out[1], 0.0)  # constant lead stays at zero

    def test_zero_nonfinite(self):
        out = zero_nonfinite(np.array([1.0, np.nan, np.inf, -np.inf, -2.0]))
        np.testing.assert_array_equal(out, [1.0, 0.0, 0.0, 0.0, -2.0])


if __name__ == "__main__":
    unittest.main()
