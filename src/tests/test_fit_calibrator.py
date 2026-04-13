import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from calibration.fit_calibrator import _ece_mce_from_rows, _reliability_rows
except ModuleNotFoundError as exc:  # pragma: no cover
    if exc.name in {"numpy", "sklearn"}:
        _ece_mce_from_rows = None
        _reliability_rows = None
    else:
        raise


@unittest.skipIf(_reliability_rows is None, "fit_calibrator dependencies are unavailable in this environment")
class FitCalibratorTests(unittest.TestCase):
    def test_reliability_rows_and_ece_mce(self):
        scores = [0.1, 0.2, 0.8, 0.9]
        targets = [0, 0, 1, 0]

        rows = _reliability_rows(scores, targets, bins=2)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["count"], 2)
        self.assertAlmostEqual(rows[0]["conf_mean"], 0.15)
        self.assertAlmostEqual(rows[0]["acc"], 0.0)
        self.assertAlmostEqual(rows[0]["gap"], 0.15)
        self.assertEqual(rows[1]["count"], 2)
        self.assertAlmostEqual(rows[1]["conf_mean"], 0.85)
        self.assertAlmostEqual(rows[1]["acc"], 0.5)
        self.assertAlmostEqual(rows[1]["gap"], 0.35)

        ece, mce = _ece_mce_from_rows(rows)
        self.assertAlmostEqual(ece, 0.25)
        self.assertAlmostEqual(mce, 0.35)

    def test_ece_mce_nan_for_empty_bins_only(self):
        ece, mce = _ece_mce_from_rows(
            [
                {"count": 0, "gap": None},
                {"count": 0, "gap": None},
            ]
        )
        self.assertTrue(math.isnan(ece))
        self.assertTrue(math.isnan(mce))


if __name__ == "__main__":
    unittest.main()
