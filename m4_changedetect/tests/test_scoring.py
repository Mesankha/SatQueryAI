"""
Unit tests for M4 change scoring and normalization module.
"""

import unittest
import numpy as np

from m4_changedetect.scoring.change_score import normalize_change_map, calculate_change_score


class TestScoringModule(unittest.TestCase):

    def test_normalize_change_map(self):
        # Raw delta array [-1.0, 2.0] -> abs values [1.0, 2.0] -> / 2.0 -> [0.5, 1.0]
        raw_map = np.array([[-1.0, 0.0], [1.0, 2.0]])
        norm = normalize_change_map(raw_map)

        self.assertIsInstance(norm, np.ndarray)
        self.assertEqual(norm.shape, (2, 2))
        np.testing.assert_allclose(norm, np.array([[0.5, 0.0], [0.5, 1.0]]))
        self.assertTrue(np.all(norm >= 0.0))
        self.assertTrue(np.all(norm <= 1.0))

    def test_zero_change(self):
        opt_results = {"mean_abs_ndvi_delta": 0.0, "mean_abs_ndwi_delta": 0.0, "has_ndwi": True}
        score = calculate_change_score(optical_results=opt_results)
        self.assertEqual(score, 0.0)

    def test_small_change(self):
        # mean_abs_ndvi_delta = 0.2 -> normalized = 0.1 -> score in (0.0, 0.3)
        opt_results = {"mean_abs_ndvi_delta": 0.2, "has_ndwi": False}
        score = calculate_change_score(optical_results=opt_results)
        self.assertIsNotNone(score)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 0.3)

    def test_moderate_change(self):
        # mean_abs_ndvi_delta = 0.8 -> normalized = 0.4 -> score in [0.3, 0.7]
        opt_results = {"mean_abs_ndvi_delta": 0.8, "has_ndwi": False}
        score = calculate_change_score(optical_results=opt_results)
        self.assertIsNotNone(score)
        self.assertGreaterEqual(score, 0.3)
        self.assertLessEqual(score, 0.7)

    def test_large_change(self):
        # mean_abs_ndvi_delta = 1.6 -> normalized = 0.8 -> score in (0.7, 1.0)
        opt_results = {"mean_abs_ndvi_delta": 1.6, "has_ndwi": False}
        score = calculate_change_score(optical_results=opt_results)
        self.assertIsNotNone(score)
        self.assertGreater(score, 0.7)
        self.assertLessEqual(score, 1.0)

    def test_maximum_boundary_values(self):
        # Maximum index delta = 2.0 -> normalized = 1.0
        opt_results = {"mean_abs_ndvi_delta": 2.0, "mean_abs_ndwi_delta": 2.0, "has_ndwi": True}
        score = calculate_change_score(optical_results=opt_results)
        self.assertEqual(score, 1.0)

    def test_ndvi_only_input(self):
        opt_results = {"mean_abs_ndvi_delta": 0.6, "has_ndwi": False}
        score = calculate_change_score(optical_results=opt_results)
        # 0.6 / 2.0 = 0.3
        self.assertEqual(score, 0.3)

    def test_ndvi_plus_ndwi_input(self):
        # NDVI delta = 0.4 -> score_ndvi = 0.2; NDWI delta = 0.8 -> score_ndwi = 0.4
        # Combined = 0.5 * 0.2 + 0.5 * 0.4 = 0.3
        opt_results = {
            "mean_abs_ndvi_delta": 0.4,
            "mean_abs_ndwi_delta": 0.8,
            "has_ndwi": True,
        }
        score = calculate_change_score(optical_results=opt_results)
        self.assertEqual(score, 0.3)

    def test_missing_partial_optical_measurement(self):
        # Empty / None optical_results -> returns None
        self.assertIsNone(calculate_change_score(None, None))
        self.assertIsNone(calculate_change_score({}, None))
        self.assertIsNone(calculate_change_score({"has_ndwi": False}, None))

    def test_score_always_nonnegative(self):
        opt_results = {"mean_abs_ndvi_delta": 0.0, "has_ndwi": False}
        score = calculate_change_score(optical_results=opt_results)
        self.assertGreaterEqual(score, 0.0)

    def test_score_always_upper_bounded(self):
        # Over-scale inputs -> should clamp to 1.0
        opt_results = {"mean_abs_ndvi_delta": 10.0, "mean_abs_ndwi_delta": 10.0, "has_ndwi": True}
        score = calculate_change_score(optical_results=opt_results)
        self.assertLessEqual(score, 1.0)
        self.assertEqual(score, 1.0)

    def test_deterministic_repeated_calculation(self):
        opt_results = {"mean_abs_ndvi_delta": 0.5432, "mean_abs_ndwi_delta": 0.1234, "has_ndwi": True}
        first_score = calculate_change_score(optical_results=opt_results)
        for _ in range(100):
            self.assertEqual(calculate_change_score(optical_results=opt_results), first_score)


if __name__ == "__main__":
    unittest.main()
