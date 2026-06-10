import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

import numpy as np

from lasso_demo.plotting import plot_convergence, plot_regularization_path


class PlottingTests(unittest.TestCase):
    def test_convergence_plot_is_saved(self) -> None:
        histories = {
            "solver": {
                "iteration": [0, 1, 2],
                "objective": [3.0, 2.0, 1.0],
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plot.png"
            fig = plot_convergence(histories, output_path=path)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)
            plt.close(fig)

    def test_regularization_path_is_saved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mu_path.png"
            fig = plot_regularization_path(
                np.array([0.01, 0.1, 1.0]),
                np.array([2.0, 1.0, 1.5]),
                np.array([10, 5, 0]),
                metric_label="Test MSE",
                output_path=path,
            )
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
