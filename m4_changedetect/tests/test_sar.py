"""
Unit tests for M4 SAR log-ratio mathematical calculation and SAR Analyzer modules.
"""

import unittest
import numpy as np

from m4_changedetect.sar.log_ratio import (
    compute_sar_log_ratio,
    compute_sar_absolute_ratio,
    compute_mean_absolute_sar_delta,
)
from m4_changedetect.sar.sar_analyzer import analyze_sar


class TestSarLogRatioModule(unittest.TestCase):

    # --- Log Ratio Calculation Tests ---
    def test_identical_sar_observations(self):
        sar = np.array([[10.0, 20.0], [30.0, 40.0]])
        log_ratio = compute_sar_log_ratio(sar, sar)
        abs_ratio = compute_sar_absolute_ratio(sar, sar)
        mean_abs = compute_mean_absolute_sar_delta(sar, sar)

        np.testing.assert_allclose(log_ratio, np.zeros((2, 2)), atol=1e-5)
        np.testing.assert_allclose(abs_ratio, np.zeros((2, 2)), atol=1e-5)
        self.assertAlmostEqual(mean_abs, 0.0, places=5)

    def test_positive_change(self):
        # Backscatter increase: t1=10, t2=100 -> log10(100/10) ~ 1.0 > 0
        sar_t1 = np.array([[10.0, 10.0]])
        sar_t2 = np.array([[100.0, 1000.0]])

        log_ratio = compute_sar_log_ratio(sar_t1, sar_t2)
        self.assertTrue(np.all(log_ratio > 0))

    def test_negative_change(self):
        # Backscatter decrease: t1=100, t2=10 -> log10(10/100) ~ -1.0 < 0
        sar_t1 = np.array([[100.0, 1000.0]])
        sar_t2 = np.array([[10.0, 10.0]])

        log_ratio = compute_sar_log_ratio(sar_t1, sar_t2)
        self.assertTrue(np.all(log_ratio < 0))

    def test_zero_value_handling(self):
        # Zero SAR intensity values -> handled safely via eps=1e-7 without NaN or Inf
        sar_zero = np.zeros((2, 2))
        log_ratio = compute_sar_log_ratio(sar_zero, sar_zero)

        self.assertFalse(np.isnan(log_ratio).any(), "SAR output contains NaN!")
        self.assertFalse(np.isinf(log_ratio).any(), "SAR output contains Inf!")
        np.testing.assert_allclose(log_ratio, np.zeros((2, 2)), atol=1e-5)

    def test_very_small_values(self):
        # Very small values near zero
        sar_t1 = np.full((2, 2), 1e-6)
        sar_t2 = np.full((2, 2), 1e-5)

        log_ratio = compute_sar_log_ratio(sar_t1, sar_t2)
        self.assertFalse(np.isnan(log_ratio).any())
        self.assertFalse(np.isinf(log_ratio).any())

    def test_known_manually_calculable_log_ratio(self):
        # t1=10.0, t2=100.0 (eps=0 for clean math) -> log10(100/10) = log10(10) = 1.0
        sar_t1 = np.array([[10.0]])
        sar_t2 = np.array([[100.0]])

        log_ratio = compute_sar_log_ratio(sar_t1, sar_t2, eps=0.0)
        self.assertAlmostEqual(log_ratio[0, 0], 1.0, places=5)

        abs_ratio = compute_sar_absolute_ratio(sar_t1, sar_t2, eps=0.0)
        self.assertAlmostEqual(abs_ratio[0, 0], 1.0, places=5)

        mean_abs = compute_mean_absolute_sar_delta(sar_t1, sar_t2, eps=0.0)
        self.assertAlmostEqual(mean_abs, 1.0, places=5)

    def test_shape_preservation_and_multidimensional_arrays(self):
        # 1D array, 2D matrix, 3D tensor
        sar_1d_t1 = np.array([10.0, 20.0])
        sar_1d_t2 = np.array([100.0, 200.0])
        self.assertEqual(compute_sar_log_ratio(sar_1d_t1, sar_1d_t2).shape, (2,))

        sar_3d_t1 = np.full((2, 3, 4), 10.0)
        sar_3d_t2 = np.full((2, 3, 4), 50.0)
        self.assertEqual(compute_sar_log_ratio(sar_3d_t1, sar_3d_t2).shape, (2, 3, 4))

    def test_no_input_array_mutation(self):
        sar_t1 = np.array([[10.0, 20.0]])
        sar_t2 = np.array([[30.0, 40.0]])
        t1_copy = sar_t1.copy()
        t2_copy = sar_t2.copy()

        _ = compute_sar_log_ratio(sar_t1, sar_t2)
        np.testing.assert_array_equal(sar_t1, t1_copy)
        np.testing.assert_array_equal(sar_t2, t2_copy)

    def test_deterministic_repeated_calculation(self):
        sar_t1 = np.array([[12.34, 56.78]])
        sar_t2 = np.array([[90.12, 34.56]])

        first = compute_sar_log_ratio(sar_t1, sar_t2)
        for _ in range(100):
            np.testing.assert_array_equal(compute_sar_log_ratio(sar_t1, sar_t2), first)

    def test_shape_mismatch_raises_value_error(self):
        sar_t1 = np.array([[10.0, 20.0]])
        sar_t2 = np.array([[10.0]])
        with self.assertRaises(ValueError):
            compute_sar_log_ratio(sar_t1, sar_t2)

    # --- SAR Analyzer Orchestrator Tests ---
    def test_analyze_sar_normal_t1_t2(self):
        img_t1 = {"sar": np.array([[10.0, 20.0]])}
        img_t2 = {"sar": np.array([[100.0, 200.0]])}

        res = analyze_sar(img_t1, img_t2)
        self.assertIn("sar_t1", res)
        self.assertIn("sar_t2", res)
        self.assertIn("sar_log_ratio", res)
        self.assertIn("sar_abs_ratio", res)
        self.assertIn("mean_abs_sar_delta", res)
        self.assertIsInstance(res["mean_abs_sar_delta"], float)
        self.assertAlmostEqual(res["mean_abs_sar_delta"], 1.0, places=5)

    def test_analyze_sar_identical_observations(self):
        img = {"sar": np.array([[10.0, 20.0]])}
        res = analyze_sar(img, img)

        np.testing.assert_allclose(res["sar_log_ratio"], np.zeros((1, 2)), atol=1e-5)
        self.assertAlmostEqual(res["mean_abs_sar_delta"], 0.0, places=5)

    def test_analyze_sar_positive_change(self):
        img_t1 = {"sar": np.array([[10.0]])}
        img_t2 = {"sar": np.array([[100.0]])}

        res = analyze_sar(img_t1, img_t2)
        self.assertGreater(res["sar_log_ratio"][0, 0], 0.0)

    def test_analyze_sar_negative_change(self):
        img_t1 = {"sar": np.array([[100.0]])}
        img_t2 = {"sar": np.array([[10.0]])}

        res = analyze_sar(img_t1, img_t2)
        self.assertLess(res["sar_log_ratio"][0, 0], 0.0)

    def test_analyze_sar_zero_and_small_values(self):
        img_t1 = {"sar": np.array([[0.0, 1e-6]])}
        img_t2 = {"sar": np.array([[0.0, 1e-5]])}

        res = analyze_sar(img_t1, img_t2)
        self.assertFalse(np.isnan(res["sar_log_ratio"]).any())
        self.assertFalse(np.isinf(res["sar_log_ratio"]).any())

    def test_analyze_sar_missing_required_sar_input_raises(self):
        with self.assertRaises(ValueError):
            analyze_sar(None, {"sar": np.array([[10.0]])})

        with self.assertRaises(ValueError):
            analyze_sar({"invalid_key": 10}, {"sar": np.array([[10.0]])})

    def test_analyze_sar_shape_mismatch_raises(self):
        img_t1 = {"sar": np.array([[10.0, 20.0]])}
        img_t2 = {"sar": np.array([[10.0]])}

        with self.assertRaises(ValueError):
            analyze_sar(img_t1, img_t2)

    def test_analyze_sar_expected_output_keys(self):
        img_t1 = np.array([[10.0]])
        img_t2 = np.array([[10.0]])

        res = analyze_sar(img_t1, img_t2)
        expected_keys = {"sar_t1", "sar_t2", "sar_log_ratio", "sar_abs_ratio", "mean_abs_sar_delta"}
        self.assertEqual(set(res.keys()), expected_keys)

    def test_analyze_sar_deterministic_repeated_execution(self):
        img_t1 = {"sar": np.array([[10.0, 50.0]])}
        img_t2 = {"sar": np.array([[100.0, 25.0]])}

        first_res = analyze_sar(img_t1, img_t2)
        for _ in range(100):
            res = analyze_sar(img_t1, img_t2)
            self.assertEqual(res["mean_abs_sar_delta"], first_res["mean_abs_sar_delta"])

    def test_analyze_sar_input_non_mutation(self):
        sar_t1 = np.array([[10.0, 20.0]])
        sar_t2 = np.array([[100.0, 200.0]])
        t1_copy = sar_t1.copy()
        t2_copy = sar_t2.copy()

        _ = analyze_sar(sar_t1, sar_t2)
        np.testing.assert_array_equal(sar_t1, t1_copy)
        np.testing.assert_array_equal(sar_t2, t2_copy)


if __name__ == "__main__":
    unittest.main()
