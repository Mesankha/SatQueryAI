"""
Comprehensive integration tests for M4 ChangeDetector orchestrator and detect_change interface.
"""

import unittest
import numpy as np

from m4_changedetect.change_detector import ChangeDetector, detect_change
from shared.schemas import ChangeResult as CanonicalChangeResult
from m4_changedetect.schemas import ChangeResult as LocalReexportedChangeResult


class TestChangeDetectorIntegration(unittest.TestCase):

    def setUp(self):
        self.detector = ChangeDetector()
        self.meta_opt_t1 = {"constellation": "SENTINEL-2", "date": "2024-01-01"}
        self.meta_opt_t2 = {"constellation": "SENTINEL-2", "date": "2024-02-01"}
        self.meta_sar_t1 = {"constellation": "SENTINEL-1", "date": "2024-01-01"}
        self.meta_sar_t2 = {"constellation": "SENTINEL-1", "date": "2024-02-01"}

    def test_detect_change_interface_contract(self):
        result = detect_change("path/to/img_t1.tif", "path/to/img_t2.tif", self.meta_opt_t1, self.meta_opt_t2)
        self.assertIsInstance(result, CanonicalChangeResult)
        self.assertIs(type(result), CanonicalChangeResult)
        self.assertIs(LocalReexportedChangeResult, CanonicalChangeResult)
        self.assertTrue(hasattr(result, "change_score"))
        self.assertTrue(hasattr(result, "change_description"))
        self.assertIsInstance(result.change_description, str)

    def test_missing_second_image(self):
        result = self.detector.detect_change("path/to/img_t1.tif", None, self.meta_opt_t1, self.meta_opt_t2)
        self.assertIsInstance(result, CanonicalChangeResult)
        self.assertIsNone(result.change_score)
        self.assertIn("Missing bi-temporal image pair", result.change_description)

    def test_missing_metadata(self):
        result = self.detector.detect_change("path/to/img_t1.tif", "path/to/img_t2.tif", None, self.meta_opt_t2)
        self.assertIsInstance(result, CanonicalChangeResult)
        self.assertIsNone(result.change_score)
        self.assertIn("Missing metadata", result.change_description)

    def test_optical_only_pipeline(self):
        img_t1 = {"nir": np.array([[0.8, 0.6]]), "red": np.array([[0.2, 0.2]]), "green": np.array([[0.3, 0.4]])}
        img_t2 = {"nir": np.array([[0.3, 0.4]]), "red": np.array([[0.4, 0.5]]), "green": np.array([[0.6, 0.7]])}

        result = self.detector.detect_change(img_t1, img_t2, self.meta_opt_t1, self.meta_opt_t2)
        self.assertIsInstance(result, CanonicalChangeResult)
        self.assertIsNotNone(result.change_score)
        self.assertGreater(result.change_score, 0.0)
        self.assertIn("Sentinel-2 optical metrics show", result.change_description)

    def test_sar_only_pipeline(self):
        img_t1 = {"sar": np.array([[10.0, 20.0]])}
        img_t2 = {"sar": np.array([[100.0, 200.0]])}

        result = self.detector.detect_change(img_t1, img_t2, self.meta_sar_t1, self.meta_sar_t2)
        self.assertIsInstance(result, CanonicalChangeResult)
        self.assertIsNotNone(result.change_score)
        self.assertGreater(result.change_score, 0.0)
        self.assertIn("Sentinel-1 SAR metrics show", result.change_description)

    def test_optical_plus_sar_pipeline(self):
        # Combined metadata
        meta_combo = {"constellation": "SENTINEL-1 & SENTINEL-2", "date": "2024-01-01"}
        img_t1 = {"nir": np.array([[0.8]]), "red": np.array([[0.2]]), "sar": np.array([[10.0]])}
        img_t2 = {"nir": np.array([[0.3]]), "red": np.array([[0.4]]), "sar": np.array([[100.0]])}

        result = self.detector.detect_change(img_t1, img_t2, meta_combo, meta_combo)
        self.assertIsInstance(result, CanonicalChangeResult)
        self.assertIsNotNone(result.change_score)
        self.assertIn("Sentinel-2 optical metrics show", result.change_description)
        self.assertIn("Sentinel-1 SAR metrics show", result.change_description)

    def test_missing_bands_fallback(self):
        img_t1 = {"blue": np.array([[0.1]])}  # missing NIR and Red
        img_t2 = {"blue": np.array([[0.2]])}

        result = self.detector.detect_change(img_t1, img_t2, self.meta_opt_t1, self.meta_opt_t2)
        self.assertIsInstance(result, CanonicalChangeResult)
        self.assertIsNone(result.change_score)
        self.assertIn("Insufficient or missing required band/sensor data", result.change_description)

    def test_partial_measurements(self):
        # Missing optional Green band -> computes NDVI only
        img_t1 = {"nir": np.array([[0.8]]), "red": np.array([[0.2]])}
        img_t2 = {"nir": np.array([[0.3]]), "red": np.array([[0.4]])}

        result = self.detector.detect_change(img_t1, img_t2, self.meta_opt_t1, self.meta_opt_t2)
        self.assertIsInstance(result, CanonicalChangeResult)
        self.assertIsNotNone(result.change_score)
        self.assertIn("vegetation index", result.change_description)
        self.assertNotIn("water index", result.change_description)

    def test_malformed_input_handling(self):
        # Invalid file path string when image files do not exist
        result = self.detector.detect_change("invalid_path_t1.tif", "invalid_path_t2.tif", self.meta_opt_t1, self.meta_opt_t2)
        self.assertIsInstance(result, CanonicalChangeResult)
        self.assertIsNone(result.change_score)
        self.assertIn("Insufficient or missing required band/sensor data", result.change_description)

    def test_zero_change(self):
        img = {"nir": np.array([[0.8, 0.6]]), "red": np.array([[0.2, 0.2]])}
        result = self.detector.detect_change(img, img, self.meta_opt_t1, self.meta_opt_t2)

        self.assertIsInstance(result, CanonicalChangeResult)
        self.assertEqual(result.change_score, 0.0)
        self.assertIn("no significant observed temporal change", result.change_description)

    def test_significant_change(self):
        img_t1 = {"nir": np.array([[0.9]]), "red": np.array([[0.1]])}
        img_t2 = {"nir": np.array([[0.1]]), "red": np.array([[0.9]])}

        result = self.detector.detect_change(img_t1, img_t2, self.meta_opt_t1, self.meta_opt_t2)
        self.assertIsInstance(result, CanonicalChangeResult)
        self.assertGreater(result.change_score, 0.7)
        self.assertIn("significant observed temporal change", result.change_description)

    def test_deterministic_explanation(self):
        img_t1 = {"nir": np.array([[0.8]]), "red": np.array([[0.2]])}
        img_t2 = {"nir": np.array([[0.3]]), "red": np.array([[0.4]])}

        res1 = self.detector.detect_change(img_t1, img_t2, self.meta_opt_t1, self.meta_opt_t2)
        res2 = self.detector.detect_change(img_t1, img_t2, self.meta_opt_t1, self.meta_opt_t2)

        self.assertEqual(res1.change_score, res2.change_score)
        self.assertEqual(res1.change_description, res2.change_description)

    def test_valid_llm_polish(self):
        valid_polished_text = "The satellite measurements are consistent with moderate land surface change."
        detector = ChangeDetector(
            config={"enable_llm_polish": True},
            llm_callable=lambda text: valid_polished_text
        )

        img_t1 = {"nir": np.array([[0.8]]), "red": np.array([[0.2]])}
        img_t2 = {"nir": np.array([[0.3]]), "red": np.array([[0.4]])}

        result = detector.detect_change(img_t1, img_t2, self.meta_opt_t1, self.meta_opt_t2)
        self.assertEqual(result.change_description, valid_polished_text)

    def test_invalid_llm_polish_fallback(self):
        # LLM output contains forbidden word "confirms" -> must fall back to Stage 3 deterministic
        invalid_polished_text = "This confirms consistent with significant change."
        detector = ChangeDetector(
            config={"enable_llm_polish": True},
            llm_callable=lambda text: invalid_polished_text
        )

        img_t1 = {"nir": np.array([[0.8]]), "red": np.array([[0.2]])}
        img_t2 = {"nir": np.array([[0.3]]), "red": np.array([[0.4]])}

        result = detector.detect_change(img_t1, img_t2, self.meta_opt_t1, self.meta_opt_t2)
        self.assertNotIn("confirms", result.change_description)
        self.assertIn("moderate observed temporal change", result.change_description)

    def test_canonical_change_result_output(self):
        img_t1 = {"nir": np.array([[0.8]]), "red": np.array([[0.2]])}
        img_t2 = {"nir": np.array([[0.8]]), "red": np.array([[0.2]])}

        result = detect_change(img_t1, img_t2, self.meta_opt_t1, self.meta_opt_t2)
        self.assertIsInstance(result, CanonicalChangeResult)
        self.assertIs(type(result), CanonicalChangeResult)


if __name__ == "__main__":
    unittest.main()
