"""
Tests verifying backwards compatibility of audit_core.py.
"""

import unittest
import audit_core
import core


class TestBackwardsCompatibility(unittest.TestCase):
    def test_exported_symbols_match(self):
        for name in audit_core.__all__:
            self.assertTrue(hasattr(audit_core, name), f"audit_core missing {name}")
            self.assertTrue(hasattr(core, name), f"core missing {name}")

    def test_constants_identical(self):
        self.assertEqual(audit_core.MAX_MODEL_BYTES, core.MAX_MODEL_BYTES)
        self.assertEqual(audit_core.MAX_DATASET_BYTES, core.MAX_DATASET_BYTES)
        self.assertEqual(audit_core.ALLOWED_MODEL_EXTS, core.ALLOWED_MODEL_EXTS)


if __name__ == "__main__":
    unittest.main()
