import unittest

import numpy as np

from lasso_demo.core import (
    create_history,
    grad_smooth,
    lasso_mu_max,
    lasso_objective,
    make_mu_grid,
    record_history,
    soft_threshold,
    validate_solver_result,
)


class CoreTests(unittest.TestCase):
    def test_objective_gradient_and_threshold(self) -> None:
        A = np.eye(2)
        b = np.array([1.0, -1.0])
        x = np.array([0.5, -0.5])
        self.assertAlmostEqual(lasso_objective(A, b, x, 0.2), 0.45)
        np.testing.assert_allclose(grad_smooth(A, b, x), [-0.5, 0.5])
        np.testing.assert_allclose(
            soft_threshold(np.array([-2.0, -0.2, 0.3, 2.0]), 0.5),
            [-1.5, 0.0, 0.0, 1.5],
        )
        self.assertAlmostEqual(lasso_mu_max(A, b), 1.0)

    def test_mu_grid_is_positive_and_ends_at_mu_max(self) -> None:
        A = np.eye(2)
        b = np.array([2.0, -1.0])
        mu_values = make_mu_grid(A, b, n_values=4, min_ratio=1e-2)
        self.assertEqual(len(mu_values), 4)
        self.assertTrue(np.all(np.diff(mu_values) > 0.0))
        self.assertAlmostEqual(mu_values[-1], 2.0)

    def test_history_contract(self) -> None:
        history = create_history()
        A = np.eye(2)
        b = np.ones(2)
        x = np.zeros(2)
        record_history(
            history,
            iteration=0,
            elapsed_time=0.01,
            A=A,
            b=b,
            x=x,
            mu=0.1,
            x_star=np.ones(2),
        )
        result = {"x": x, "history": history}
        validate_solver_result(result, n_features=2)
        self.assertEqual(history["sparsity"], [0])
        self.assertAlmostEqual(history["error"][0], 1.0)

    def test_solver_contract_rejects_wrong_x_shape(self) -> None:
        history = create_history()
        record_history(
            history,
            iteration=0,
            elapsed_time=0.01,
            A=np.eye(2),
            b=np.ones(2),
            x=np.zeros(2),
            mu=0.1,
        )
        result = {"x": np.zeros(3), "history": history}
        with self.assertRaisesRegex(ValueError, "must have shape"):
            validate_solver_result(result, n_features=2)

    def test_solver_contract_rejects_incomplete_history(self) -> None:
        result = {
            "x": np.zeros(2),
            "history": {
                "iteration": [0],
                "time": [0.01],
                "objective": [1.0],
            },
        }
        with self.assertRaisesRegex(ValueError, "missing required keys"):
            validate_solver_result(result, n_features=2)


if __name__ == "__main__":
    unittest.main()
