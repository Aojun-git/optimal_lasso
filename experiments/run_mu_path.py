"""Template for comparing a solver across a data-derived mu grid."""

from lasso_demo.core import make_mu_grid
from lasso_demo.data import load_dataset
from lasso_demo.pipeline import run_mu_sweep

# Replace this comment after a solver is implemented:
# from lasso_demo.algorithms.module_b.ista import ista


def main() -> None:
    # This template demonstrates one dataset. Repeat the sweep for the datasets
    # listed in the experiment matrix in README.md.
    dataset = load_dataset("data/processed/diabetes/diabetes.npz")
    solvers = {
        # "ISTA": ista,
    }
    if not solvers:
        raise RuntimeError(
            "No solver is registered. Implement a solver and add it to 'solvers'."
        )

    mu_values = make_mu_grid(
        dataset.A_train,
        dataset.b_train,
        n_values=10,
        min_ratio=1e-3,
        max_ratio=1.0,
    )
    solver_configs = {
        name: {"max_iter": 1000}
        for name in solvers
    }
    run_mu_sweep(
        dataset,
        solvers,
        mu_values,
        solver_configs=solver_configs,
    )


if __name__ == "__main__":
    main()
