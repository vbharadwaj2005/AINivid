"""
Tests for core.utils module.
"""

import unittest
import pandas as pd
from core.utils import (
    is_numeric_dtype,
    validate_sensitive_column,
    validate_model_filename,
    extract_target,
    safe_run,
)


class TestUtils(unittest.TestCase):
    def test_is_numeric_dtype(self):
        s1 = pd.Series([1, 2, 3])
        s2 = pd.Series(["a", "b", "c"])
        self.assertTrue(is_numeric_dtype(s1))
        self.assertFalse(is_numeric_dtype(s2))

    def test_validate_sensitive_column(self):
        df = pd.DataFrame({"sex": ["M", "F"], "age": [20, 30]})
        self.assertEqual(validate_sensitive_column(df, "sex"), "sex")
        with self.assertRaises(ValueError):
            validate_sensitive_column(df, "race")
        with self.assertRaises(ValueError):
            validate_sensitive_column(df, "")

    def test_validate_model_filename(self):
        self.assertEqual(validate_model_filename("model.pkl"), "model.pkl")
        self.assertEqual(validate_model_filename("model.joblib"), "model.joblib")
        with self.assertRaises(ValueError):
            validate_model_filename("model.exe")

    def test_extract_target(self):
        df = pd.DataFrame({"age": [20, 30], "income": ["<=50K", ">50K"]})
        X, y, col_name = extract_target(df, "income")
        self.assertEqual(col_name, "income")
        self.assertIn("age", X.columns)
        self.assertNotIn("income", X.columns)
        self.assertListEqual(list(y), [0, 1])

        # Test generic target auto-detection
        df2 = pd.DataFrame({"feature1": [1.0, 2.0], "label": ["No", "Yes"]})
        X2, y2, col_name2 = extract_target(df2)
        self.assertEqual(col_name2, "label")
        self.assertListEqual(list(y2), [0, 1])

    def test_safe_run(self):
        def good_fn(a, b):
            return {"res": a + b}

        def bad_fn():
            raise ValueError("Something broke")

        self.assertEqual(safe_run(good_fn, 2, 3), {"res": 5})
        res = safe_run(bad_fn)
        self.assertIn("error", res)
        self.assertIn("Something broke", res["error"])


if __name__ == "__main__":
    unittest.main()
