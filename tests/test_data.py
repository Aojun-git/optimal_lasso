import tempfile
import unittest
from pathlib import Path

import numpy as np

from lasso_demo.data import (
    load_dataset,
    load_diabetes_dataset,
    make_synthetic_lasso,
    save_dataset,
)


class DataTests(unittest.TestCase):
    def test_synthetic_is_reproducible_and_consistent(self) -> None:
        first = make_synthetic_lasso(m=40, n=60, k=6, noise=0.0, seed=7)
        second = make_synthetic_lasso(m=40, n=60, k=6, noise=0.0, seed=7)
        np.testing.assert_allclose(first.A_train, second.A_train)
        np.testing.assert_allclose(first.b_train, first.A_train @ first.x_star)
        np.testing.assert_array_equal(
            np.flatnonzero(first.x_star),
            first.support,
        )

    def test_dataset_round_trip(self) -> None:
        dataset = make_synthetic_lasso(m=20, n=30, k=3, seed=3)
        with tempfile.TemporaryDirectory() as directory:
            path = save_dataset(dataset, Path(directory) / "data.npz")
            loaded = load_dataset(path)
        self.assertEqual(loaded.name, dataset.name)
        np.testing.assert_allclose(loaded.A_train, dataset.A_train)
        np.testing.assert_array_equal(loaded.support, dataset.support)
        self.assertEqual(loaded.metadata, dataset.metadata)

    def test_diabetes_split(self) -> None:
        dataset = load_diabetes_dataset(seed=0)
        self.assertEqual(dataset.A_train.shape, (264, 10))
        self.assertEqual(dataset.A_val.shape, (89, 10))
        self.assertEqual(dataset.A_test.shape, (89, 10))
        self.assertAlmostEqual(float(dataset.A_train.mean()), 0.0, places=12)
        self.assertAlmostEqual(float(dataset.b_train.mean()), 0.0, places=12)
        with tempfile.TemporaryDirectory() as directory:
            path = save_dataset(dataset, Path(directory) / "diabetes.npz")
            loaded = load_dataset(path)
        np.testing.assert_allclose(loaded.A_val, dataset.A_val)
        np.testing.assert_allclose(loaded.b_val, dataset.b_val)


if __name__ == "__main__":
    unittest.main()
