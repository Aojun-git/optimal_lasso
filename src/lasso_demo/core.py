"""Mathematical primitives and the shared solver output contract."""

from __future__ import annotations

from typing import Any

import numpy as np


REQUIRED_HISTORY_KEYS = ("iteration", "time", "objective", "sparsity", "error")


def lasso_objective(
    A: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
    mu: float,
) -> float:
    """Return 0.5 * ||Ax-b||_2^2 + mu * ||x||_1."""
    if mu < 0:
        raise ValueError("mu must be non-negative")
    residual = np.asarray(A) @ np.asarray(x) - np.asarray(b)
    return float(0.5 * residual @ residual + mu * np.linalg.norm(x, ord=1))


def grad_smooth(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Gradient of the least-squares term 0.5 * ||Ax-b||_2^2."""
    A_array = np.asarray(A)
    return A_array.T @ (A_array @ np.asarray(x) - np.asarray(b))


def soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    """Apply the elementwise L1 proximal operator."""
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    values_array = np.asarray(values)
    return np.sign(values_array) * np.maximum(np.abs(values_array) - threshold, 0.0)


def lasso_mu_max(A: np.ndarray, b: np.ndarray) -> float:
    """Return the smallest mu for which x=0 is a LASSO optimum."""
    return float(np.linalg.norm(np.asarray(A).T @ np.asarray(b), ord=np.inf))


def make_mu_grid(
    A: np.ndarray,
    b: np.ndarray,
    *,
    n_values: int = 8,
    min_ratio: float = 1e-3,
    max_ratio: float = 1.0,
) -> np.ndarray:
    """Build a log-spaced mu grid relative to ||A.T @ b||_inf."""
    if n_values < 2:
        raise ValueError("n_values must be at least 2")
    if not 0.0 < min_ratio < max_ratio:
        raise ValueError("ratios must satisfy 0 < min_ratio < max_ratio")

    mu_max = lasso_mu_max(A, b)
    if mu_max <= 0.0:
        raise ValueError("cannot build a positive mu grid when ||A.T @ b||_inf is zero")
    return mu_max * np.geomspace(min_ratio, max_ratio, num=n_values)


def create_history() -> dict[str, list[float | int]]:
    """Create the common history structure used by every solver."""
    return {
        "iteration": [],
        "time": [],
        "objective": [],
        "sparsity": [],
        "error": [],
        "primal_residual": [],
        "dual_residual": [],
    }


def record_history(
    history: dict[str, list[float | int]],
    *,
    iteration: int,
    elapsed_time: float,
    A: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
    mu: float,
    x_star: np.ndarray | None = None,
    zero_threshold: float = 1e-6,
    primal_residual: float | None = None,
    dual_residual: float | None = None,
) -> None:
    """Append one iteration to a solver history."""
    x_array = np.asarray(x)
    if x_star is None:
        error = np.nan
    else:
        denominator = max(float(np.linalg.norm(x_star)), np.finfo(float).eps)
        error = float(np.linalg.norm(x_array - x_star) / denominator)

    history["iteration"].append(int(iteration))
    history["time"].append(float(elapsed_time))
    history["objective"].append(lasso_objective(A, b, x_array, mu))
    history["sparsity"].append(int(np.count_nonzero(np.abs(x_array) > zero_threshold)))
    history["error"].append(error)
    history["primal_residual"].append(
        np.nan if primal_residual is None else float(primal_residual)
    )
    history["dual_residual"].append(
        np.nan if dual_residual is None else float(dual_residual)
    )


def validate_solver_result(result: dict[str, Any], n_features: int) -> None:
    """Validate the interface shared by module B and module C solvers."""
    if set(("x", "history")) - result.keys():
        raise ValueError("solver result must contain 'x' and 'history'")

    x = np.asarray(result["x"])
    if x.shape != (n_features,):
        raise ValueError(f"result['x'] must have shape ({n_features},), got {x.shape}")

    history = result["history"]
    if not isinstance(history, dict):
        raise TypeError("result['history'] must be a dictionary")
    missing = set(REQUIRED_HISTORY_KEYS) - history.keys()
    if missing:
        raise ValueError(f"history is missing required keys: {sorted(missing)}")

    lengths = {len(history[key]) for key in REQUIRED_HISTORY_KEYS}
    if len(lengths) != 1:
        raise ValueError("all required history series must have the same length")
    expected_length = lengths.pop()
    if expected_length == 0:
        raise ValueError("history must contain at least one recorded iteration")
    for key in ("primal_residual", "dual_residual"):
        if key in history and len(history[key]) != expected_length:
            raise ValueError(f"history['{key}'] must match the common history length")
