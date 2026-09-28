"""
Unit tests for M4 explanation module (Stage 3 deterministic template generator & Stage 4 LLM polisher).
"""

import unittest
import numpy as np

from m4_changedetect.explanation.templates import generate_change_description
from m4_changedetect.explanation.llm_polish import polish_change_description, validate_llm_guardrails


class TestStage3ExplanationGenerator(unittest.TestCase):

    def test_zero_change(self):
        opt = {"ndvi_diff": np.zeros((2, 2)), "ndwi_diff": np.zeros((2, 2)), "has_ndwi": True}
        desc = generate_change_description(0.0, optical_results=opt)
        self.assertIn("no significant observed temporal change", desc)
        self.assertIn("change score: 0.0000", desc)
        self.assertIn("stable vegetation index", desc)
        self.assertIn("stable water index", desc)

    def test_small_change(self):
        opt = {"ndvi_diff": np.full((2, 2), 0.1), "has_ndwi": False}
        desc = generate_change_description(0.15, optical_results=opt)
        self.assertIn("minor observed temporal change", desc)
        self.assertIn("change score: 0.1500", desc)
        self.assertIn("increase in vegetation index", desc)

    def test_moderate_change(self):
        opt = {"ndvi_diff": np.full((2, 2), -0.4), "has_ndwi": False}
        desc = generate_change_description(0.50, optical_results=opt)
        self.assertIn("moderate observed temporal change", desc)
        self.assertIn("change score: 0.5000", desc)
        self.assertIn("decrease in vegetation index", desc)

    def test_large_change(self):
        opt = {"ndvi_diff": np.full((2, 2), 0.8), "has_ndwi": False}
        desc = generate_change_description(0.85, optical_results=opt)
        self.assertIn("significant observed temporal change", desc)
        self.assertIn("change score: 0.8500", desc)
        self.assertIn("increase in vegetation index", desc)

    def test_ndvi_only(self):
        opt = {"ndvi_diff": np.full((2, 2), 0.3), "has_ndwi": False}
        desc = generate_change_description(0.25, optical_results=opt)
        self.assertIn("Sentinel-2 optical metrics show", desc)
        self.assertIn("vegetation index", desc)
        self.assertNotIn("water index", desc)
        self.assertNotIn("SAR", desc)

    def test_ndwi_only(self):
        opt = {"mean_abs_ndwi_delta": 0.4, "has_ndwi": True}
        desc = generate_change_description(0.20, optical_results=opt)
        self.assertIn("observed water index variation", desc)
        self.assertNotIn("vegetation index", desc)
        self.assertNotIn("SAR", desc)

    def test_sar_only(self):
        sar = {"sar_log_ratio": np.full((2, 2), 0.6)}
        desc = generate_change_description(0.30, sar_results=sar)
        self.assertIn("Sentinel-1 SAR metrics show", desc)
        self.assertIn("increase in SAR backscatter intensity", desc)
        self.assertNotIn("Sentinel-2 optical", desc)

    def test_optical_plus_sar(self):
        opt = {"ndvi_diff": np.full((2, 2), -0.3), "has_ndwi": False}
        sar = {"sar_log_ratio": np.full((2, 2), 0.5)}
        desc = generate_change_description(0.40, optical_results=opt, sar_results=sar)
        self.assertIn("Sentinel-2 optical metrics show decrease in vegetation index", desc)
        self.assertIn("Sentinel-1 SAR metrics show increase in SAR backscatter intensity", desc)

    def test_missing_ndvi(self):
        opt = {"has_ndwi": True, "ndwi_diff": np.full((2, 2), 0.4)}
        desc = generate_change_description(0.20, optical_results=opt)
        self.assertIn("increase in water index", desc)
        self.assertNotIn("vegetation index", desc)

    def test_missing_ndwi(self):
        opt = {"ndvi_diff": np.full((2, 2), 0.2), "has_ndwi": False}
        desc = generate_change_description(0.10, optical_results=opt)
        self.assertIn("vegetation index", desc)
        self.assertNotIn("water index", desc)

    def test_missing_sar(self):
        opt = {"ndvi_diff": np.full((2, 2), 0.1), "has_ndwi": False}
        desc = generate_change_description(0.05, optical_results=opt, sar_results=None)
        self.assertNotIn("SAR", desc)

    def test_partial_measurements(self):
        opt = {"mean_abs_ndvi_delta": 0.1}
        desc = generate_change_description(0.05, optical_results=opt)
        self.assertIn("observed vegetation index variation", desc)

    def test_deterministic_repeated_calls(self):
        opt = {"ndvi_diff": np.full((2, 2), 0.35), "has_ndwi": False}
        sar = {"sar_log_ratio": np.full((2, 2), -0.25)}
        first = generate_change_description(0.30, optical_results=opt, sar_results=sar)
        for _ in range(100):
            self.assertEqual(generate_change_description(0.30, optical_results=opt, sar_results=sar), first)

    def test_unsupported_malformed_input_handling(self):
        desc = generate_change_description(None, optical_results=None, sar_results=None)
        self.assertEqual(desc, "Insufficient data provided for temporal change description.")


class TestStage4LLMPolisher(unittest.TestCase):

    def setUp(self):
        self.deterministic_text = "Bi-temporal analysis indicates moderate observed temporal change."

    def test_valid_polished_output(self):
        valid_llm_text = "The satellite measurements are consistent with moderate land surface change."
        result = polish_change_description(
            self.deterministic_text,
            config={"enabled": True},
            llm_callable=lambda text: valid_llm_text
        )
        self.assertEqual(result, valid_llm_text)

    def test_missing_consistent_with(self):
        # Missing required phrase "consistent with"
        bad_llm_text = "Moderate land surface change was observed across observations."
        result = polish_change_description(
            self.deterministic_text,
            config={"enabled": True},
            llm_callable=lambda text: bad_llm_text
        )
        self.assertEqual(result, self.deterministic_text)

    def test_output_containing_confirms(self):
        bad_llm_text = "This confirms consistent with significant deforestation in the region."
        result = polish_change_description(
            self.deterministic_text,
            config={"enabled": True},
            llm_callable=lambda text: bad_llm_text
        )
        self.assertEqual(result, self.deterministic_text)

    def test_output_containing_detects(self):
        bad_llm_text = "The model detects changes consistent with urban growth."
        result = polish_change_description(
            self.deterministic_text,
            config={"enabled": True},
            llm_callable=lambda text: bad_llm_text
        )
        self.assertEqual(result, self.deterministic_text)

    def test_output_containing_proves(self):
        bad_llm_text = "The spectral analysis proves changes consistent with vegetation loss."
        result = polish_change_description(
            self.deterministic_text,
            config={"enabled": True},
            llm_callable=lambda text: bad_llm_text
        )
        self.assertEqual(result, self.deterministic_text)

    def test_mixed_case_forbidden_words(self):
        # Mixed case forbidden words: CONFIRMS, Detects, PROVES
        for forbidden_text in [
            "This CONFIRMS consistent with vegetation loss.",
            "Algorithm Detects pattern consistent with flooding.",
            "The data PROVES change consistent with land clearing.",
        ]:
            result = polish_change_description(
                self.deterministic_text,
                config={"enabled": True},
                llm_callable=lambda text: forbidden_text
            )
            self.assertEqual(result, self.deterministic_text, f"Failed for {forbidden_text}")

    def test_llm_exception(self):
        def failing_llm(text):
            raise RuntimeError("API timeout connection error")

        result = polish_change_description(
            self.deterministic_text,
            config={"enabled": True},
            llm_callable=failing_llm
        )
        self.assertEqual(result, self.deterministic_text)

    def test_llm_unavailable(self):
        result = polish_change_description(
            self.deterministic_text,
            config={"enabled": True},
            llm_callable=None
        )
        self.assertEqual(result, self.deterministic_text)

    def test_empty_llm_output(self):
        result = polish_change_description(
            self.deterministic_text,
            config={"enabled": True},
            llm_callable=lambda text: ""
        )
        self.assertEqual(result, self.deterministic_text)

    def test_config_disabled(self):
        valid_llm_text = "The satellite measurements are consistent with moderate land surface change."
        result = polish_change_description(
            self.deterministic_text,
            config={"enabled": False},
            llm_callable=lambda text: valid_llm_text
        )
        self.assertEqual(result, self.deterministic_text)

    def test_validate_llm_guardrails_standalone(self):
        self.assertTrue(validate_llm_guardrails("Output consistent with observations."))
        self.assertFalse(validate_llm_guardrails("Output shows change."))  # Missing "consistent with"
        self.assertFalse(validate_llm_guardrails("Data confirms consistent with change."))
        self.assertFalse(validate_llm_guardrails("Model detects consistent with change."))
        self.assertFalse(validate_llm_guardrails("Analysis proves consistent with change."))
        self.assertFalse(validate_llm_guardrails(""))
        self.assertFalse(validate_llm_guardrails(None))


if __name__ == "__main__":
    unittest.main()
