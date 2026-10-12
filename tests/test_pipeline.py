import tempfile
import unittest
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from src.quality.pipeline import run_quality_pipeline
from src.quality.profiler import profile_csv


class PipelineTests(unittest.TestCase):
    def run_csv(self, frame: pd.DataFrame) -> dict:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            delete=False,
            newline="",
        ) as file:
            path = Path(file.name)
            frame.to_csv(file, index=False)

        try:
            return run_quality_pipeline(str(path), rules={})
        finally:
            path.unlink(missing_ok=True)

    def test_numeric_dataset_returns_flagged_records(self):
        frame = pd.DataFrame({
            "value": [1, 2, 2, 3, 4, 5, 6, 7, 8, 1000],
        })

        report = self.run_csv(frame)

        self.assertEqual(report["anomaly_status"], "completed")
        self.assertEqual(report["anomalous_rows"], len(report["anomaly_records"]))
        self.assertTrue(report["anomaly_records"])
        self.assertTrue(
            any(issue["type"] == "anomalous_rows" for issue in report["issues"])
        )

    def test_non_numeric_dataset_skips_anomaly_detection(self):
        frame = pd.DataFrame({"category": ["a", "b", "c", "d", "e"]})

        report = self.run_csv(frame)

        self.assertEqual(
            report["anomaly_status"],
            "skipped",
        )
        self.assertEqual(report["anomalous_rows"], 0)
        self.assertEqual(report["anomaly_records"], [])

    def test_small_dataset_skips_anomaly_detection(self):
        frame = pd.DataFrame({"value": [1, 2, 3, 4]})

        report = self.run_csv(frame)

        self.assertEqual(
            report["anomaly_status"],
            "skipped",
        )
        self.assertEqual(report["anomaly_records"], [])

    def test_empty_csv_is_rejected(self):
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            delete=False,
        ) as file:
            path = Path(file.name)

        try:
            with self.assertRaises((ValueError, EmptyDataError)):
                profile_csv(str(path))
        finally:
            path.unlink(missing_ok=True)

    def test_malformed_csv_is_rejected(self):
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            delete=False,
        ) as file:
            file.write("name,value\n")
            file.write('"unterminated,10\n')
            path = Path(file.name)

        try:
            with self.assertRaises(ParserError):
                profile_csv(str(path))
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
