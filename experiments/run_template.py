"""Template for the final experiment scripts after solvers are available."""

from lasso_demo.core import lasso_mu_max
from lasso_demo.data import load_dataset
from lasso_demo.pipeline import run_experiment

# Replace these comments with imports from module_b and module_c:
# from lasso_demo.algorithms.module_b.ista import ista
# from lasso_demo.algorithms.module_c.admm import admm


def main() -> None:
    # This template intentionally runs one explicitly selected dataset.
    # The complete experiment matrix and multi-dataset loop are documented in README.md.
    dataset = load_dataset("data/processed/synthetic/synthetic_demo.npz")
    mu_ratio = 0.05
    mu = mu_ratio * lasso_mu_max(dataset.A_train, dataset.b_train)
    solvers = {
        # "ISTA": ista,
        # "ADMM": admm,
    }
    if not solvers:
        raise RuntimeError(
            "No solver is registered. Implement module B/C solvers and add them here."
        )
    solver_configs = {
        name: {"max_iter": 1000, "x_star": dataset.x_star}
        for name in solvers
    }
    run_experiment(
        dataset,
        solvers,
        mu=mu,
        solver_configs=solver_configs,
    )


if __name__ == "__main__":
    main()
