"""Unified final metrics and convergence-gap helpers."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .core import lasso_objective


def support_metrics(
    x: np.ndarray,
    x_star: np.ndarray,
    *,
    zero_threshold: float = 1e-6,
) -> dict[str, float]:
    predicted = set(np.flatnonzero(np.abs(x) > zero_threshold).tolist())
    truth = set(np.flatnonzero(np.abs(x_star) > zero_threshold).tolist())
    true_positive = len(predicted & truth)

    precision = true_positive / len(predicted) if predicted else float(not truth)
    recall = true_positive / len(truth) if truth else float(not predicted)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return {
        "support_precision": float(precision),
        "support_recall": float(recall),
        "support_f1": float(f1),
    }


def evaluate_solution(
    A_train: np.ndarray,
    b_train: np.ndarray,
    x: np.ndarray,
    mu: float,
    *,
    elapsed_time: float,
    A_val: np.ndarray | None = None,
    b_val: np.ndarray | None = None,
    A_test: np.ndarray | None = None,
    b_test: np.ndarray | None = None,
    x_star: np.ndarray | None = None,
    objective_reference: float | None = None,
    zero_threshold: float = 1e-6,
) -> dict[str, float | int]:
    """Compute the common final result-table metrics."""
    objective = lasso_objective(A_train, b_train, x, mu)
    metrics: dict[str, float | int] = {
        "objective": objective,
        "nonzero_count": int(np.count_nonzero(np.abs(x) > zero_threshold)),
        "cpu_time": float(elapsed_time),
    }

    if objective_reference is not None:
        denominator = max(abs(objective_reference), np.finfo(float).eps)
        metrics["relative_objective_gap"] = (
            objective - objective_reference
        ) / denominator

    if x_star is not None:
        denominator = max(float(np.linalg.norm(x_star)), np.finfo(float).eps)
        metrics["relative_error"] = float(np.linalg.norm(x - x_star) / denominator)
        metrics.update(support_metrics(x, x_star, zero_threshold=zero_threshold))

    if A_val is not None and b_val is not None:
        residual = A_val @ x - b_val
        metrics["validation_mse"] = float(np.mean(residual**2))

    if A_test is not None and b_test is not None:
        residual = A_test @ x - b_test
        metrics["test_mse"] = float(np.mean(residual**2))

    return metrics


def relative_objective_histories(
    histories: Mapping[str, Mapping[str, list[float | int]]],
) -> dict[str, dict[str, list[float | int]]]:
    """Return history copies with a common relative_objective_gap series."""
    final_objectives = [
        float(history["objective"][-1])
        for history in histories.values()
        if history["objective"]
    ]
    if not final_objectives:
        raise ValueError("at least one non-empty objective history is required")

    reference = min(final_objectives)
    denominator = max(abs(reference), np.finfo(float).eps)
    output: dict[str, dict[str, list[float | int]]] = {}
    for name, history in histories.items():
        copied = {key: list(values) for key, values in history.items()}
        copied["relative_objective_gap"] = [
            max((float(value) - reference) / denominator, np.finfo(float).eps)
            for value in history["objective"]
        ]
        output[name] = copied
    return output
