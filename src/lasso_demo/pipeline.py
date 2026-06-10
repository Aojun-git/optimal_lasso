"""Common experiment runner used after module B/C solvers are implemented."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .core import validate_solver_result
from .data import DatasetBundle
from .metrics import (
    evaluate_solution,
    relative_objective_histories,
)
from .plotting import plot_convergence, plot_regularization_path
from .results import build_result_row, export_history, export_result_rows


Solver = Callable[
    [Any, Any, float, Mapping[str, Any]],
    dict[str, Any],
]


def _filename_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return normalized.strip("_.") or "unnamed"


def _mu_filename(mu: float) -> str:
    return f"{mu:.8g}".replace("-", "m").replace(".", "p").replace("+", "")


def run_experiment(
    dataset: DatasetBundle,
    solvers: Mapping[str, Solver],
    *,
    mu: float,
    output_root: str | Path = "outputs",
    solver_configs: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, str | float | int]], dict[str, dict[str, list]]]:
    """Run registered solvers, validate outputs, export tables, and draw curves."""
    if not solvers:
        raise ValueError("at least one solver is required")
    dataset.validate()
    solver_configs = solver_configs or {}
    output_root = Path(output_root)

    results: list[dict[str, str | float | int]] = []
    histories: dict[str, dict[str, list]] = {}
    raw_results: dict[str, dict[str, Any]] = {}

    for name, solver in solvers.items():
        result = solver(
            dataset.A_train,
            dataset.b_train,
            mu,
            solver_configs.get(name, {}),
        )
        validate_solver_result(result, dataset.A_train.shape[1])
        raw_results[name] = result
        histories[name] = result["history"]

    objective_reference = min(
        float(result["history"]["objective"][-1])
        for result in raw_results.values()
        if result["history"]["objective"]
    )
    for name, result in raw_results.items():
        history = result["history"]
        elapsed_time = float(history["time"][-1]) if history["time"] else 0.0
        metrics = evaluate_solution(
            dataset.A_train,
            dataset.b_train,
            result["x"],
            mu,
            elapsed_time=elapsed_time,
            A_val=dataset.A_val,
            b_val=dataset.b_val,
            A_test=dataset.A_test,
            b_test=dataset.b_test,
            x_star=dataset.x_star,
            objective_reference=objective_reference,
        )
        results.append(
            build_result_row(
                dataset=dataset.name,
                algorithm=name,
                mu=mu,
                metrics=metrics,
            )
        )
        export_history(
            history,
            output_root / "logs" / f"{dataset.name}_{name}_history.csv",
        )

    export_result_rows(
        results,
        output_root / "tables" / f"{dataset.name}_results.csv",
    )
    plot_convergence(
        histories,
        title=f"{dataset.name}: objective by iteration",
        output_path=output_root / "figures" / f"{dataset.name}_objective_iteration.png",
    )
    plt.close()
    plot_convergence(
        histories,
        x_key="time",
        title=f"{dataset.name}: objective by CPU time",
        output_path=output_root / "figures" / f"{dataset.name}_objective_time.png",
    )
    plt.close()
    gap_histories = relative_objective_histories(histories)
    plot_convergence(
        gap_histories,
        y_key="relative_objective_gap",
        log_y=True,
        title=f"{dataset.name}: relative objective gap",
        output_path=output_root / "figures" / f"{dataset.name}_relative_gap.png",
    )
    plt.close()

    return results, histories


def run_mu_sweep(
    dataset: DatasetBundle,
    solvers: Mapping[str, Solver],
    mu_values: list[float] | np.ndarray,
    *,
    output_root: str | Path = "outputs",
    solver_configs: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, str | float | int]]:
    """Run each solver over a positive mu grid and export path results."""
    if not solvers:
        raise ValueError("at least one solver is required")
    dataset.validate()

    mu_array = np.asarray(mu_values, dtype=float)
    if mu_array.ndim != 1 or mu_array.size == 0:
        raise ValueError("mu_values must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(mu_array)) or np.any(mu_array <= 0.0):
        raise ValueError("all mu values must be finite and positive")
    if len(np.unique(mu_array)) != len(mu_array):
        raise ValueError("mu_values must not contain duplicates")
    mu_array = np.sort(mu_array)

    solver_configs = solver_configs or {}
    output_root = Path(output_root)
    rows: list[dict[str, str | float | int]] = []
    metrics_by_solver: dict[str, list[dict[str, float | int]]] = {
        name: [] for name in solvers
    }

    for name, solver in solvers.items():
        config = solver_configs.get(name, {})
        safe_name = _filename_component(name)
        for mu in mu_array:
            result = solver(
                dataset.A_train,
                dataset.b_train,
                float(mu),
                config,
            )
            validate_solver_result(result, dataset.A_train.shape[1])
            history = result["history"]
            elapsed_time = float(history["time"][-1])
            metrics = evaluate_solution(
                dataset.A_train,
                dataset.b_train,
                result["x"],
                float(mu),
                elapsed_time=elapsed_time,
                A_val=dataset.A_val,
                b_val=dataset.b_val,
                x_star=dataset.x_star,
            )
            rows.append(
                build_result_row(
                    dataset=dataset.name,
                    algorithm=name,
                    mu=float(mu),
                    metrics=metrics,
                )
            )
            metrics_by_solver[name].append(metrics)
            export_history(
                history,
                output_root
                / "logs"
                / "mu_sweep"
                / f"{dataset.name}_{safe_name}_mu_{_mu_filename(float(mu))}.csv",
            )

    export_result_rows(
        rows,
        output_root / "tables" / f"{dataset.name}_mu_sweep_results.csv",
    )

    for name, metrics_list in metrics_by_solver.items():
        if all("validation_mse" in metrics for metrics in metrics_list):
            metric_key = "validation_mse"
            metric_label = "Validation MSE"
        elif all("relative_error" in metrics for metrics in metrics_list):
            metric_key = "relative_error"
            metric_label = "Relative recovery error"
        elif all("test_mse" in metrics for metrics in metrics_list):
            metric_key = "test_mse"
            metric_label = "Test MSE"
        else:
            continue

        fig = plot_regularization_path(
            mu_array,
            np.asarray([metrics[metric_key] for metrics in metrics_list]),
            np.asarray([metrics["nonzero_count"] for metrics in metrics_list]),
            metric_label=metric_label,
            title=f"{dataset.name}: {name} regularization path",
            output_path=(
                output_root
                / "figures"
                / f"{dataset.name}_{_filename_component(name)}_mu_path.png"
            ),
        )
        plt.close(fig)

    return rows
