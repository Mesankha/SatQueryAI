"""
Unit tests for M4 optical analysis module (Sentinel-2 NDVI, NDWI & Optical Analyzer).
"""

import unittest
import numpy as np

from m4_changedetect.optical.ndvi import (
    compute_ndvi,
    compute_ndvi_difference,
    compute_mean_absolute_ndvi_delta,
)
from m4_changedetect.optical.ndwi import (
    compute_ndwi,
    compute_ndwi_difference,
    compute_mean_absolute_ndwi_delta,
)
from m4_changedetect.optical.optical_analyzer import analyze_optical


class TestOpticalModule(unittest.TestCase):

    # --- NDVI Test Cases ---
    def test_normal_ndvi_calculation(self):
        # NIR = 0.8, Red = 0.2 -> (0.8 - 0.2) / (0.8 + 0.2) = 0.6 / 1.0 = 0.6
        nir = np.array([[0.8, 0.6], [0.7, 0.9]])
        red = np.array([[0.2, 0.2], [0.1, 0.1]])
        expected = np.array([[0.6, 0.5], [0.75, 0.8]])

        result = compute_ndvi(nir, red)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 2))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_positive_vegetation_change(self):
        # Vegetation growth: t1 low NDVI, t2 high NDVI -> delta > 0
        nir_t1 = np.array([[0.3, 0.4]])
        red_t1 = np.array([[0.2, 0.2]])  # NDVI t1 = [0.2, 0.3333]

        nir_t2 = np.array([[0.8, 0.9]])
        red_t2 = np.array([[0.1, 0.1]])  # NDVI t2 = [0.7778, 0.8]

        ndvi_t1 = compute_ndvi(nir_t1, red_t1)
        ndvi_t2 = compute_ndvi(nir_t2, red_t2)

        delta = compute_ndvi_difference(ndvi_t1, ndvi_t2)
        mean_abs_delta = compute_mean_absolute_ndvi_delta(ndvi_t1, ndvi_t2)

        self.assertTrue(np.all(delta > 0))
        self.assertGreater(mean_abs_delta, 0.4)

    def test_negative_vegetation_change(self):
        # Deforestation/degradation: t1 high NDVI, t2 low NDVI -> delta < 0
        nir_t1 = np.array([[0.8, 0.8]])
        red_t1 = np.array([[0.1, 0.1]])  # NDVI t1 = [0.7778, 0.7778]

        nir_t2 = np.array([[0.2, 0.2]])
        red_t2 = np.array([[0.3, 0.3]])  # NDVI t2 = [-0.2, -0.2]

        ndvi_t1 = compute_ndvi(nir_t1, red_t1)
        ndvi_t2 = compute_ndvi(nir_t2, red_t2)

        delta = compute_ndvi_difference(ndvi_t1, ndvi_t2)
        self.assertTrue(np.all(delta < 0))

    def test_zero_denominator_handling(self):
        # Zero NIR and Red -> NIR + Red = 0 -> Should return 0.0, no NaN or Inf
        nir = np.array([[0.0, 0.5], [0.0, 0.0]])
        red = np.array([[0.0, 0.5], [0.0, 0.0]])

        result = compute_ndvi(nir, red)
        self.assertFalse(np.isnan(result).any(), "NDVI output contains NaN!")
        self.assertFalse(np.isinf(result).any(), "NDVI output contains Inf!")
        self.assertEqual(result[0, 0], 0.0)
        self.assertEqual(result[1, 0], 0.0)
        self.assertEqual(result[1, 1], 0.0)
        self.assertEqual(result[0, 1], 0.0)  # (0.5-0.5)/(0.5+0.5) = 0.0

    def test_identical_t1_t2_observations(self):
        # Identical observations -> delta = 0.0
        nir = np.array([[0.7, 0.6], [0.5, 0.8]])
        red = np.array([[0.2, 0.1], [0.3, 0.2]])

        ndvi_t1 = compute_ndvi(nir, red)
        ndvi_t2 = compute_ndvi(nir, red)

        delta = compute_ndvi_difference(ndvi_t1, ndvi_t2)
        mean_abs_delta = compute_mean_absolute_ndvi_delta(ndvi_t1, ndvi_t2)

        np.testing.assert_allclose(delta, np.zeros((2, 2)), atol=1e-7)
        self.assertAlmostEqual(mean_abs_delta, 0.0, places=7)

    def test_known_ndvi_difference(self):
        # Known difference test
        ndvi_t1 = np.array([[0.2, 0.5]])
        ndvi_t2 = np.array([[0.8, 0.3]])

        delta = compute_ndvi_difference(ndvi_t1, ndvi_t2)
        expected_delta = np.array([[0.6, -0.2]])
        np.testing.assert_allclose(delta, expected_delta, rtol=1e-5)

        mean_abs_delta = compute_mean_absolute_ndvi_delta(ndvi_t1, ndvi_t2)
        # mean(|0.6|, |-0.2|) = (0.6 + 0.2) / 2 = 0.4
        self.assertAlmostEqual(mean_abs_delta, 0.4, places=5)

    def test_output_shape_preservation(self):
        # 1D array, 2D matrix, 3D tensor
        nir_1d = np.array([0.5, 0.6])
        red_1d = np.array([0.1, 0.2])
        self.assertEqual(compute_ndvi(nir_1d, red_1d).shape, (2,))

        nir_3d = np.zeros((2, 3, 4))
        red_3d = np.ones((2, 3, 4))
        self.assertEqual(compute_ndvi(nir_3d, red_3d).shape, (2, 3, 4))

    def test_no_input_array_mutation(self):
        nir = np.array([[0.8, 0.6]])
        red = np.array([[0.2, 0.1]])
        nir_copy = nir.copy()
        red_copy = red.copy()

        _ = compute_ndvi(nir, red)
        np.testing.assert_array_equal(nir, nir_copy)
        np.testing.assert_array_equal(red, red_copy)

    # --- NDWI Test Cases ---
    def test_normal_ndwi_calculation(self):
        # Green = 0.6, NIR = 0.2 -> (0.6 - 0.2) / (0.6 + 0.2) = 0.4 / 0.8 = 0.5
        green = np.array([[0.6, 0.4], [0.8, 0.5]])
        nir = np.array([[0.2, 0.1], [0.2, 0.5]])
        expected = np.array([[0.5, 0.6], [0.6, 0.0]])

        result = compute_ndwi(green, nir)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 2))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_positive_water_change(self):
        # Flooding / water expansion: t1 dry (low NDWI), t2 flooded (high NDWI) -> delta > 0
        green_t1 = np.array([[0.2, 0.2]])
        nir_t1 = np.array([[0.8, 0.6]])  # NDWI t1 < 0

        green_t2 = np.array([[0.7, 0.8]])
        nir_t2 = np.array([[0.1, 0.2]])  # NDWI t2 > 0

        ndwi_t1 = compute_ndwi(green_t1, nir_t1)
        ndwi_t2 = compute_ndwi(green_t2, nir_t2)

        delta = compute_ndwi_difference(ndwi_t1, ndwi_t2)
        mean_abs_delta = compute_mean_absolute_ndwi_delta(ndwi_t1, ndwi_t2)

        self.assertTrue(np.all(delta > 0))
        self.assertGreater(mean_abs_delta, 0.5)

    def test_negative_water_change(self):
        # Water body recession / drying: t1 high NDWI, t2 low NDWI -> delta < 0
        green_t1 = np.array([[0.8, 0.7]])
        nir_t1 = np.array([[0.1, 0.2]])

        green_t2 = np.array([[0.2, 0.3]])
        nir_t2 = np.array([[0.8, 0.7]])

        ndwi_t1 = compute_ndwi(green_t1, nir_t1)
        ndwi_t2 = compute_ndwi(green_t2, nir_t2)

        delta = compute_ndwi_difference(ndwi_t1, ndwi_t2)
        self.assertTrue(np.all(delta < 0))

    def test_ndwi_zero_denominator_handling(self):
        # Zero Green and NIR -> Green + NIR = 0 -> Should return 0.0, no NaN or Inf
        green = np.array([[0.0, 0.4], [0.0, 0.0]])
        nir = np.array([[0.0, 0.4], [0.0, 0.0]])

        result = compute_ndwi(green, nir)
        self.assertFalse(np.isnan(result).any(), "NDWI output contains NaN!")
        self.assertFalse(np.isinf(result).any(), "NDWI output contains Inf!")
        self.assertEqual(result[0, 0], 0.0)
        self.assertEqual(result[1, 0], 0.0)
        self.assertEqual(result[1, 1], 0.0)
        self.assertEqual(result[0, 1], 0.0)

    def test_ndwi_identical_t1_t2_observations(self):
        green = np.array([[0.5, 0.6]])
        nir = np.array([[0.2, 0.3]])

        ndwi_t1 = compute_ndwi(green, nir)
        ndwi_t2 = compute_ndwi(green, nir)

        delta = compute_ndwi_difference(ndwi_t1, ndwi_t2)
        mean_abs_delta = compute_mean_absolute_ndwi_delta(ndwi_t1, ndwi_t2)

        np.testing.assert_allclose(delta, np.zeros((1, 2)), atol=1e-7)
        self.assertAlmostEqual(mean_abs_delta, 0.0, places=7)

    def test_ndwi_known_difference(self):
        ndwi_t1 = np.array([[0.1, 0.4]])
        ndwi_t2 = np.array([[0.7, 0.2]])

        delta = compute_ndwi_difference(ndwi_t1, ndwi_t2)
        expected_delta = np.array([[0.6, -0.2]])
        np.testing.assert_allclose(delta, expected_delta, rtol=1e-5)

        mean_abs_delta = compute_mean_absolute_ndwi_delta(ndwi_t1, ndwi_t2)
        self.assertAlmostEqual(mean_abs_delta, 0.4, places=5)

    def test_ndwi_no_input_array_mutation(self):
        green = np.array([[0.6, 0.5]])
        nir = np.array([[0.2, 0.1]])
        green_copy = green.copy()
        nir_copy = nir.copy()

        _ = compute_ndwi(green, nir)
        np.testing.assert_array_equal(green, green_copy)
        np.testing.assert_array_equal(nir, nir_copy)

    # --- Optical Analyzer Orchestrator Test Cases ---
    def test_analyze_optical_normal_t1_t2(self):
        img_t1 = {"nir": np.array([[0.8, 0.6]]), "red": np.array([[0.2, 0.2]]), "green": np.array([[0.3, 0.4]])}
        img_t2 = {"nir": np.array([[0.3, 0.4]]), "red": np.array([[0.4, 0.5]]), "green": np.array([[0.6, 0.7]])}

        res = analyze_optical(img_t1, img_t2)
        self.assertIn("ndvi_t1", res)
        self.assertIn("ndvi_t2", res)
        self.assertIn("ndvi_diff", res)
        self.assertIn("mean_abs_ndvi_delta", res)
        self.assertIn("ndwi_t1", res)
        self.assertIn("ndwi_t2", res)
        self.assertIn("ndwi_diff", res)
        self.assertIn("mean_abs_ndwi_delta", res)
        self.assertTrue(res["has_ndwi"])
        self.assertIsInstance(res["mean_abs_ndvi_delta"], float)
        self.assertIsInstance(res["mean_abs_ndwi_delta"], float)

    def test_analyze_optical_without_green_band(self):
        img_t1 = {"nir": np.array([[0.8, 0.6]]), "red": np.array([[0.2, 0.2]])}
        img_t2 = {"nir": np.array([[0.3, 0.4]]), "red": np.array([[0.4, 0.5]])}

        res = analyze_optical(img_t1, img_t2)
        self.assertFalse(res["has_ndwi"])
        self.assertIsNone(res["ndwi_t1"])
        self.assertIsNone(res["ndwi_t2"])
        self.assertIsNone(res["ndwi_diff"])
        self.assertIsNone(res["mean_abs_ndwi_delta"])
        self.assertIsNotNone(res["ndvi_diff"])

    def test_analyze_optical_positive_change(self):
        # Vegetation and water growth
        img_t1 = {"nir": np.array([[0.3]]), "red": np.array([[0.3]]), "green": np.array([[0.2]])}
        img_t2 = {"nir": np.array([[0.8]]), "red": np.array([[0.1]]), "green": np.array([[0.7]])}

        res = analyze_optical(img_t1, img_t2)
        self.assertGreater(res["mean_abs_ndvi_delta"], 0.0)
        self.assertGreater(res["mean_abs_ndwi_delta"], 0.0)

    def test_analyze_optical_negative_change(self):
        # Deforestation & drying
        img_t1 = {"nir": np.array([[0.8]]), "red": np.array([[0.1]]), "green": np.array([[0.7]])}
        img_t2 = {"nir": np.array([[0.2]]), "red": np.array([[0.4]]), "green": np.array([[0.1]])}

        res = analyze_optical(img_t1, img_t2)
        self.assertLess(res["ndvi_diff"][0, 0], 0.0)
        self.assertLess(res["ndwi_diff"][0, 0], 0.0)

    def test_analyze_optical_identical_observations(self):
        img = {"nir": np.array([[0.8, 0.6]]), "red": np.array([[0.2, 0.1]]), "green": np.array([[0.5, 0.4]])}

        res = analyze_optical(img, img)
        self.assertAlmostEqual(res["mean_abs_ndvi_delta"], 0.0, places=7)
        self.assertAlmostEqual(res["mean_abs_ndwi_delta"], 0.0, places=7)
        np.testing.assert_allclose(res["ndvi_diff"], np.zeros((1, 2)), atol=1e-7)

    def test_analyze_optical_expected_aggregate_values(self):
        img_t1 = {"nir": np.array([[0.8]]), "red": np.array([[0.2]]), "green": np.array([[0.6]])}
        img_t2 = {"nir": np.array([[0.4]]), "red": np.array([[0.4]]), "green": np.array([[0.4]])}

        # ndvi_t1 = (0.8-0.2)/1.0 = 0.6; ndvi_t2 = (0.4-0.4)/0.8 = 0.0 -> delta = -0.6 -> mean_abs = 0.6
        # ndwi_t1 = (0.6-0.8)/1.4 = -0.142857; ndwi_t2 = (0.4-0.4)/0.8 = 0.0 -> delta = 0.142857 -> mean_abs = 0.142857
        res = analyze_optical(img_t1, img_t2)
        self.assertAlmostEqual(res["mean_abs_ndvi_delta"], 0.6, places=5)
        self.assertAlmostEqual(res["mean_abs_ndwi_delta"], 2 / 14, places=5)

    def test_analyze_optical_shape_mismatch_raises(self):
        img_t1 = {"nir": np.array([[0.8, 0.6]]), "red": np.array([[0.2, 0.2]])}
        img_t2 = {"nir": np.array([[0.3], [0.4]]), "red": np.array([[0.4], [0.5]])}

        with self.assertRaises(ValueError):
            analyze_optical(img_t1, img_t2)

    def test_analyze_optical_missing_required_band_data_raises(self):
        img_t1 = {"red": np.array([[0.2]])}  # missing NIR
        img_t2 = {"nir": np.array([[0.3]]), "red": np.array([[0.4]])}

        with self.assertRaises(ValueError):
            analyze_optical(img_t1, img_t2)

    def test_analyze_optical_zero_denominator_propagated_safely(self):
        img_t1 = {"nir": np.array([[0.0]]), "red": np.array([[0.0]]), "green": np.array([[0.0]])}
        img_t2 = {"nir": np.array([[0.0]]), "red": np.array([[0.0]]), "green": np.array([[0.0]])}

        res = analyze_optical(img_t1, img_t2)
        self.assertFalse(np.isnan(res["ndvi_t1"]).any())
        self.assertFalse(np.isinf(res["ndvi_t1"]).any())
        self.assertEqual(res["mean_abs_ndvi_delta"], 0.0)

    def test_analyze_optical_no_input_array_mutation(self):
        nir_t1 = np.array([[0.8]])
        red_t1 = np.array([[0.2]])
        nir_t1_copy = nir_t1.copy()
        red_t1_copy = red_t1.copy()

        img_t1 = {"nir": nir_t1, "red": red_t1}
        img_t2 = {"nir": np.array([[0.3]]), "red": np.array([[0.4]])}

        _ = analyze_optical(img_t1, img_t2)
        np.testing.assert_array_equal(nir_t1, nir_t1_copy)
        np.testing.assert_array_equal(red_t1, red_t1_copy)


if __name__ == "__main__":
    unittest.main()
