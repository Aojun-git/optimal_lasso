"""Run the complete LASSO optimisation experiment suite.

Produces all 8 required figures and summary CSV tables for:
  - Synthetic sparse data (main experiment)
  - Diabetes real data
  - E2006-tfidf subset (extension)
  - ADMM residual analysis
  - Parameter sensitivity (mu path)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script use

import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lasso_demo.core import lasso_mu_max, make_mu_grid, validate_solver_result
from lasso_demo.data import load_dataset
from lasso_demo.metrics import evaluate_solution, relative_objective_histories
from lasso_demo.plotting import (
    plot_admm_residuals,
    plot_coefficient_comparison,
    plot_convergence,
    plot_l1_l2_geometry,
    plot_mu_path,
    plot_relative_gap,
    plot_result_bars,
)
from lasso_demo.results import build_result_row, export_result_rows

from lasso_demo.algorithms.module_b.subgradient import subgradient
from lasso_demo.algorithms.module_b.ista import ista
from lasso_demo.algorithms.module_b.fista import fista
from lasso_demo.algorithms.module_c.coordinate_descent import coordinate_descent
from lasso_demo.algorithms.module_c.admm import admm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SOLVERS = {
    "Subgradient": subgradient,
    "ISTA": ista,
    "FISTA": fista,
    "CD": coordinate_descent,
    "ADMM": admm,
}

OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DATA_ROOT = PROJECT_ROOT / "data" / "processed"

SYNTHETIC_DATASETS = [
    DATA_ROOT / "synthetic" / "synthetic_demo.npz",
]

DIABETES_DATASET = DATA_ROOT / "diabetes" / "diabetes.npz"

E2006_DATASET = DATA_ROOT / "e2006" / "e2006_subset.npz"


def _default_configs(x_star=None, max_iter=1000):
    """Build solver configs for all 5 solvers."""
    return {
        "Subgradient": {"max_iter": max_iter, "alpha0": 0.01, "x_star": x_star},
        "ISTA": {"max_iter": max_iter, "x_star": x_star},
        "FISTA": {"max_iter": max_iter, "x_star": x_star},
        "CD": {"max_iter": max_iter, "x_star": x_star},
        "ADMM": {"max_iter": max_iter, "rho": 1.0, "x_star": x_star},
    }


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_synthetic_main(mu_ratio=0.05):
    """Level 1: main synthetic experiment — Fig.2,3,4,5."""
    print("\n" + "=" * 60)
    print("Level 1: Synthetic main experiment")
    print("=" * 60)

    dataset = load_dataset(SYNTHETIC_DATASETS[0])
    mu = mu_ratio * lasso_mu_max(dataset.A_train, dataset.b_train)
    configs = _default_configs(dataset.x_star, max_iter=1000)

    results = []
    histories = {}
    estimates = {}

    for name, solver in SOLVERS.items():
        print(f"  Running {name} ...")
        result = solver(dataset.A_train, dataset.b_train, mu, configs[name])
        validate_solver_result(result, dataset.A_train.shape[1])
        histories[name] = result["history"]
        estimates[name] = result["x"]

        # Evaluate
        elapsed = result["history"]["time"][-1]
        m = evaluate_solution(
            dataset.A_train, dataset.b_train, result["x"], mu,
            elapsed_time=elapsed, x_star=dataset.x_star,
        )
        results.append(build_result_row(
            dataset=dataset.name, algorithm=name, mu=mu, metrics=m,
        ))

    # --- Export results table ---
    export_result_rows(results, OUTPUT_ROOT / "tables" / "synthetic_demo_results.csv")
    print("  → Saved results table")

    # --- Fig.1: L1/L2 geometry ---
    plot_l1_l2_geometry(output_path=OUTPUT_ROOT / "figures" / "Fig1_L1_L2_geometry.png")
    plt.close()
    print("  → Fig.1 saved")

    # --- Fig.2: True vs recovered coefficients ---
    plot_coefficient_comparison(
        dataset.x_star, estimates,
        title="True vs recovered coefficients (synthetic)",
        output_path=OUTPUT_ROOT / "figures" / "Fig2_coefficient_comparison.png",
    )
    plt.close()
    print("  → Fig.2 saved")

    # --- Fig.3: Objective vs iteration ---
    plot_convergence(
        histories,
        title="Objective value vs iteration",
        output_path=OUTPUT_ROOT / "figures" / "Fig3_objective_iteration.png",
    )
    plt.close()
    print("  → Fig.3 saved")

    # --- Fig.4: Relative objective gap (semi-log) ---
    plot_relative_gap(
        histories,
        title="Relative objective gap (semi-log)",
        output_path=OUTPUT_ROOT / "figures" / "Fig4_relative_gap.png",
    )
    plt.close()
    print("  → Fig.4 saved")

    # --- Fig.5: Objective vs CPU time ---
    plot_convergence(
        histories, x_key="time",
        title="Objective value vs CPU time",
        output_path=OUTPUT_ROOT / "figures" / "Fig5_objective_time.png",
    )
    plt.close()
    print("  → Fig.5 saved")

    # --- Fig.7: ADMM residuals ---
    admm_hist = histories.get("ADMM")
    if admm_hist is not None:
        plot_admm_residuals(
            admm_hist,
            title="ADMM: primal and dual residuals",
            output_path=OUTPUT_ROOT / "figures" / "Fig7_ADMM_residuals.png",
        )
        plt.close()
        print("  → Fig.7 saved")

    return histories, results


def run_mu_path_sensitivity():
    """Level 2: parameter sensitivity — Fig.6."""
    print("\n" + "=" * 60)
    print("Level 2: mu path (parameter sensitivity)")
    print("=" * 60)

    # --- Synthetic mu sweep ---
    dataset_synth = load_dataset(SYNTHETIC_DATASETS[0])
    mu_values = make_mu_grid(dataset_synth.A_train, dataset_synth.b_train,
                             n_values=10, min_ratio=1e-3, max_ratio=0.5)

    # Use FISTA as representative solver for synthetic mu path
    synth_mse_list = []
    synth_nnz_list = []
    for mu in mu_values:
        result = fista(dataset_synth.A_train, dataset_synth.b_train, mu,
                       {"max_iter": 500, "x_star": dataset_synth.x_star})
        x = result["x"]
        rel_err = float(np.linalg.norm(x - dataset_synth.x_star) / max(np.linalg.norm(dataset_synth.x_star), 1e-12))
        synth_mse_list.append(rel_err)
        synth_nnz_list.append(int(np.count_nonzero(np.abs(x) > 1e-6)))

    plot_mu_path(
        mu_values, np.array(synth_mse_list), np.array(synth_nnz_list),
        output_path=OUTPUT_ROOT / "figures" / "Fig6a_synthetic_mu_path.png",
    )
    plt.close()
    print("  → Fig.6a (synthetic mu path) saved")

    # --- Diabetes mu sweep ---
    dataset_diabetes = load_dataset(DIABETES_DATASET)
    mu_values_d = make_mu_grid(dataset_diabetes.A_train, dataset_diabetes.b_train,
                               n_values=10, min_ratio=1e-3, max_ratio=1.0)

    diabetes_mse_list = []
    diabetes_nnz_list = []
    for mu in mu_values_d:
        result = fista(dataset_diabetes.A_train, dataset_diabetes.b_train, mu,
                       {"max_iter": 500})
        x = result["x"]
        if dataset_diabetes.A_test is not None:
            residual = dataset_diabetes.A_test @ x - dataset_diabetes.b_test
            test_mse = float(np.mean(residual ** 2))
        else:
            test_mse = 0.0
        diabetes_mse_list.append(test_mse)
        diabetes_nnz_list.append(int(np.count_nonzero(np.abs(x) > 1e-6)))

    plot_mu_path(
        mu_values_d, np.array(diabetes_mse_list), np.array(diabetes_nnz_list),
        output_path=OUTPUT_ROOT / "figures" / "Fig6b_diabetes_mu_path.png",
    )
    plt.close()
    print("  → Fig.6b (diabetes mu path) saved")


def run_admm_rho_experiment():
    """ADMM rho sensitivity — extra Fig.7 variants."""
    print("\n" + "=" * 60)
    print("ADMM rho sensitivity experiment")
    print("=" * 60)

    dataset = load_dataset(SYNTHETIC_DATASETS[0])
    mu = 0.05 * lasso_mu_max(dataset.A_train, dataset.b_train)

    rho_values = [0.1, 1.0, 10.0]
    for rho in rho_values:
        result = admm(dataset.A_train, dataset.b_train, mu,
                      {"max_iter": 500, "rho": rho, "x_star": dataset.x_star})
        plot_admm_residuals(
            result["history"],
            title=f"ADMM residuals (rho={rho})",
            output_path=OUTPUT_ROOT / "figures" / f"Fig7_ADMM_rho_{rho}.png",
        )
        plt.close()
        print(f"  → ADMM rho={rho} residuals saved")


def run_diabetes_experiment():
    """Level 3: Diabetes real-data validation."""
    print("\n" + "=" * 60)
    print("Level 3: Diabetes real-data experiment")
    print("=" * 60)

    dataset = load_dataset(DIABETES_DATASET)
    mu = 0.05 * lasso_mu_max(dataset.A_train, dataset.b_train)
    configs = _default_configs(None, max_iter=500)

    results = []
    histories = {}

    for name, solver in SOLVERS.items():
        print(f"  Running {name} ...")
        result = solver(dataset.A_train, dataset.b_train, mu, configs[name])
        validate_solver_result(result, dataset.A_train.shape[1])
        histories[name] = result["history"]
        elapsed = result["history"]["time"][-1]
        m = evaluate_solution(
            dataset.A_train, dataset.b_train, result["x"], mu,
            elapsed_time=elapsed,
            A_val=dataset.A_val, b_val=dataset.b_val,
            A_test=dataset.A_test, b_test=dataset.b_test,
        )
        results.append(build_result_row(
            dataset=dataset.name, algorithm=name, mu=mu, metrics=m,
        ))

    export_result_rows(results, OUTPUT_ROOT / "tables" / "diabetes_results.csv")
    print("  → Diabetes results table saved")

    # Convergence curves for diabetes
    plot_convergence(
        histories,
        title="Diabetes: objective vs iteration",
        output_path=OUTPUT_ROOT / "figures" / "diabetes_objective_iteration.png",
    )
    plt.close()
    plot_convergence(
        histories, x_key="time",
        title="Diabetes: objective vs CPU time",
        output_path=OUTPUT_ROOT / "figures" / "diabetes_objective_time.png",
    )
    plt.close()
    print("  → Diabetes convergence figures saved")


def run_e2006_experiment():
    """Level 3 extension: E2006-tfidf subset — Fig.8."""
    print("\n" + "=" * 60)
    print("Level 3 extension: E2006-tfidf subset")
    print("=" * 60)

    if not E2006_DATASET.exists():
        print("  [SKIP] E2006 subset data not found, skipping.")
        return

    # Check if file is a valid .npz (not a Git LFS pointer)
    import os
    if os.path.getsize(E2006_DATASET) < 1000:
        print("  [SKIP] E2006 subset is a Git LFS pointer, run 'git lfs pull' first.")
        return

    try:
        dataset = load_dataset(E2006_DATASET)
    except Exception as e:
        print(f"  [SKIP] Cannot load E2006 subset: {e}")
        return
    mu = 0.05 * lasso_mu_max(dataset.A_train, dataset.b_train)

    # Use only FISTA, CD, ADMM for large-scale (as per assignment)
    fast_solvers = {
        "FISTA": fista,
        "CD": coordinate_descent,
        "ADMM": admm,
    }
    configs = {
        "FISTA": {"max_iter": 300},
        "CD": {"max_iter": 300},
        "ADMM": {"max_iter": 300, "rho": 5.0, "adaptive_rho": True, "tol": 1e-4},
    }

    results = []
    histories = {}

    for name, solver in fast_solvers.items():
        print(f"  Running {name} ...")
        result = solver(dataset.A_train, dataset.b_train, mu, configs[name])
        validate_solver_result(result, dataset.A_train.shape[1])
        histories[name] = result["history"]
        elapsed = result["history"]["time"][-1]
        m = evaluate_solution(
            dataset.A_train, dataset.b_train, result["x"], mu,
            elapsed_time=elapsed,
            A_val=dataset.A_val, b_val=dataset.b_val,
            A_test=dataset.A_test, b_test=dataset.b_test,
        )
        results.append(build_result_row(
            dataset=dataset.name, algorithm=name, mu=mu, metrics=m,
        ))

    export_result_rows(results, OUTPUT_ROOT / "tables" / "e2006_results.csv")
    print("  → E2006 results table saved")

    # Fig.8: E2006 summary bar chart
    names = [r["algorithm"] for r in results]
    test_mses = [float(r.get("test_mse", 0)) for r in results]
    nnzs = [int(r.get("nonzero_count", 0)) for r in results]
    times = [float(r.get("cpu_time", 0)) for r in results]

    from lasso_demo.plotting import plot_e2006_summary
    plot_e2006_summary(
        names, test_mses, nnzs, times,
        output_path=OUTPUT_ROOT / "figures" / "Fig8_e2006_summary.png",
    )
    plt.close()
    print("  → Fig.8 saved")


def run_corr_noise_experiments():
    """Extra: correlation and noise sensitivity."""
    print("\n" + "=" * 60)
    print("Extra: correlation & noise sensitivity")
    print("=" * 60)

    presets = [
        ("synthetic_corr_050.npz", "corr=0.5"),
        ("synthetic_corr_095.npz", "corr=0.95"),
        ("synthetic_noise_050.npz", "noise=0.05"),
        ("synthetic_noise_100.npz", "noise=0.1"),
    ]

    for fname, label in presets:
        path = DATA_ROOT / "synthetic" / fname
        if not path.exists():
            print(f"  [SKIP] {fname} not found")
            continue

        dataset = load_dataset(path)
        mu = 0.05 * lasso_mu_max(dataset.A_train, dataset.b_train)
        configs = _default_configs(dataset.x_star, max_iter=500)

        results = []
        for name, solver in SOLVERS.items():
            result = solver(dataset.A_train, dataset.b_train, mu, configs[name])
            elapsed = result["history"]["time"][-1]
            m = evaluate_solution(
                dataset.A_train, dataset.b_train, result["x"], mu,
                elapsed_time=elapsed, x_star=dataset.x_star,
            )
            results.append(build_result_row(
                dataset=dataset.name, algorithm=name, mu=mu, metrics=m,
            ))

        export_result_rows(results, OUTPUT_ROOT / "tables" / f"{dataset.name}_results.csv")
        print(f"  → {label} results saved")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "figures").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "logs").mkdir(parents=True, exist_ok=True)

    # Level 1: Main synthetic experiment (Fig.1-5, 7)
    run_synthetic_main()

    # Level 2: Parameter sensitivity (Fig.6)
    run_mu_path_sensitivity()

    # ADMM rho sensitivity
    run_admm_rho_experiment()

    # Level 3: Real data (Diabetes)
    run_diabetes_experiment()

    # Level 3: E2006 extension (Fig.8)
    run_e2006_experiment()

    # Extra: Correlation & noise
    run_corr_noise_experiments()

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Figures → {OUTPUT_ROOT / 'figures'}")
    print(f"Tables  → {OUTPUT_ROOT / 'tables'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
