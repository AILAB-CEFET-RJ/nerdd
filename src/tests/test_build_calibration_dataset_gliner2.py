import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_calibration_dataset_gliner2 import _normalize_entity_types


class BuildCalibrationDatasetGLiNER2Tests(unittest.TestCase):
    def test_normalize_entity_types_maps_supported_labels(self):
        self.assertEqual(
            _normalize_entity_types(["Person", "Location", "Organization"]),
            ["person", "location", "organization"],
        )
        self.assertEqual(_normalize_entity_types(["Person", "Unknown"]), ["person"])


if __name__ == "__main__":
    unittest.main()
