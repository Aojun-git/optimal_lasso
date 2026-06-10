import os
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from lasso_demo.core import create_history, grad_smooth, record_history, soft_threshold
from lasso_demo.data import make_synthetic_lasso
from lasso_demo.pipeline import run_experiment, run_mu_sweep


def mock_solver(A, b, mu, config):
    x = np.zeros(A.shape[1])
    history = create_history()
    start = time.perf_counter()
    record_history(
        history,
        iteration=0,
        elapsed_time=time.perf_counter() - start,
        A=A,
        b=b,
        x=x,
        mu=mu,
        x_star=config.get("x_star"),
    )
    x = soft_threshold(x - 0.1 * grad_smooth(A, b, x), 0.1 * mu)
    record_history(
        history,
        iteration=1,
        elapsed_time=time.perf_counter() - start,
        A=A,
        b=b,
        x=x,
        mu=mu,
        x_star=config.get("x_star"),
    )
    return {"x": x, "history": history}


class PipelineTests(unittest.TestCase):
    def test_end_to_end_outputs(self) -> None:
        dataset = make_synthetic_lasso(m=20, n=30, k=3, seed=5, name="pipeline")
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            rows, histories = run_experiment(
                dataset,
                {"mock": mock_solver},
                mu=0.05,
                output_root=output_root,
                solver_configs={"mock": {"x_star": dataset.x_star}},
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(histories["mock"]["objective"]), 2)
            self.assertTrue((output_root / "tables/pipeline_results.csv").exists())
            self.assertTrue((output_root / "logs/pipeline_mock_history.csv").exists())
            self.assertEqual(len(list((output_root / "figures").glob("*.png"))), 3)

    def test_mu_sweep_outputs(self) -> None:
        dataset = make_synthetic_lasso(
            m=20,
            n=30,
            k=3,
            seed=5,
            name="mu_pipeline",
        )
        dataset.A_val = dataset.A_train[:5]
        dataset.b_val = dataset.b_train[:5]
        dataset.A_test = dataset.A_train[5:10]
        dataset.b_test = dataset.b_train[5:10]
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            rows = run_mu_sweep(
                dataset,
                {"mock solver": mock_solver},
                [0.01, 0.05, 0.1],
                output_root=output_root,
                solver_configs={"mock solver": {"x_star": dataset.x_star}},
            )
            self.assertEqual(len(rows), 3)
            self.assertEqual([row["mu"] for row in rows], [0.01, 0.05, 0.1])
            self.assertIn("validation_mse", rows[0])
            self.assertNotIn("test_mse", rows[0])
            self.assertTrue(
                (output_root / "tables/mu_pipeline_mu_sweep_results.csv").exists()
            )
            self.assertEqual(
                len(list((output_root / "logs/mu_sweep").glob("*.csv"))),
                3,
            )
            self.assertTrue(
                (output_root / "figures/mu_pipeline_mock_solver_mu_path.png").exists()
            )


if __name__ == "__main__":
    unittest.main()
