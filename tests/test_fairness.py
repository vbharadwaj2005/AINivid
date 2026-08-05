"""
Tests for core.fairness module.
"""

import unittest
import pandas as pd
from core.fairness import (
    calculate_fairness_metrics,
    calculate_regression_fairness,
    intersectional_analysis,
    calibration_parity,
    tune_thresholds,
    generate_retraining_suggestions,
)


class TestFairness(unittest.TestCase):
    def setUp(self):
        self.y_true = [1, 0, 1, 0, 1, 0, 1, 0]
        self.y_pred = [1, 0, 1, 0, 0, 0, 1, 1]
        self.sens = ["A", "A", "A", "A", "B", "B", "B", "B"]

    def test_calculate_fairness_metrics(self):
        metrics = calculate_fairness_metrics(self.y_true, self.y_pred, self.sens)
        self.assertIn("spd", metrics)
        self.assertIn("di", metrics)
        self.assertIn("eod", metrics)
        self.assertIn("aod", metrics)
        self.assertIn("group_metrics", metrics)

    def test_calculate_regression_fairness(self):
        y_t = pd.Series([10.0, 20.0, 30.0, 40.0])
        y_p = pd.Series([12.0, 18.0, 32.0, 38.0])
        sens = ["Group1", "Group1", "Group2", "Group2"]
        res = calculate_regression_fairness(y_t, y_p, sens)
        self.assertIn("meanPredictionDifference", res)
        self.assertIn("maxMaeDisparity", res)

    def test_intersectional_analysis(self):
        df_sens = pd.DataFrame({"race": ["W", "W", "B", "B"], "gender": ["M", "F", "M", "F"]})
        y_t = [1, 0, 1, 0]
        y_p = [1, 0, 0, 0]
        res = intersectional_analysis(y_t, y_p, df_sens, ["race", "gender"])
        self.assertIn("intersection_groups", res)
        self.assertIn("fairness", res)

    def test_calibration_parity(self):
        y_t = [1, 0, 1, 0]
        y_proba = [0.9, 0.1, 0.8, 0.2]
        sens = ["A", "A", "B", "B"]
        res = calibration_parity(y_t, y_proba, sens, n_bins=5)
        self.assertIn("calibrationError", res)

    def test_tune_thresholds(self):
        y_t = [1, 0, 1, 0, 1, 0]
        y_proba = [0.9, 0.2, 0.8, 0.1, 0.7, 0.3]
        sens = ["A", "A", "A", "B", "B", "B"]
        res = tune_thresholds(y_t, y_proba, sens)
        self.assertIn("optimalFairness", res)
        self.assertIn("recommended", res)

    def test_generate_retraining_suggestions(self):
        df = pd.DataFrame({"feat1": [1, 2, 3, 4], "sens": ["A", "A", "B", "B"]})
        res = generate_retraining_suggestions(df, "sens")
        self.assertIn("resamplingSuggestion", res)
        self.assertIn("reweightingSuggestion", res)


if __name__ == "__main__":
    unittest.main()
