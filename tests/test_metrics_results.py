import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lasso_demo.metrics import evaluate_solution, support_metrics
from lasso_demo.results import build_result_row, export_result_rows


class MetricsAndResultsTests(unittest.TestCase):
    def test_support_and_test_metrics(self) -> None:
        x_star = np.array([1.0, 0.0, -2.0, 0.0])
        x = np.array([1.0, 0.5, 0.0, 0.0])
        support = support_metrics(x, x_star)
        self.assertAlmostEqual(support["support_precision"], 0.5)
        self.assertAlmostEqual(support["support_recall"], 0.5)
        self.assertAlmostEqual(support["support_f1"], 0.5)

        A = np.eye(4)
        metrics = evaluate_solution(
            A,
            x_star,
            x,
            0.1,
            elapsed_time=0.2,
            A_val=A,
            b_val=x_star,
            A_test=A,
            b_test=x_star,
            x_star=x_star,
        )
        self.assertIn("objective", metrics)
        self.assertIn("relative_error", metrics)
        self.assertIn("validation_mse", metrics)
        self.assertIn("test_mse", metrics)

    def test_csv_export_uses_stable_columns(self) -> None:
        row = build_result_row(
            dataset="synthetic",
            algorithm="test",
            mu=0.1,
            metrics={"objective": 1.2, "nonzero_count": 2, "cpu_time": 0.1},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = export_result_rows([row], Path(directory) / "results.csv")
            with path.open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
        self.assertEqual(rows[0]["dataset"], "synthetic")
        self.assertEqual(rows[0]["algorithm"], "test")


if __name__ == "__main__":
    unittest.main()
