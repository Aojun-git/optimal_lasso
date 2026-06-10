"""Reusable plotting functions with a consistent visual style."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save_or_return(fig: plt.Figure, output_path: str | Path | None) -> plt.Figure:
    fig.tight_layout()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
    return fig


def plot_convergence(
    histories: Mapping[str, Mapping[str, list[float | int]]],
    *,
    x_key: str = "iteration",
    y_key: str = "objective",
    log_y: bool = False,
    title: str | None = None,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot one common history metric for multiple algorithms."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for name, history in histories.items():
        if x_key not in history or y_key not in history:
            raise KeyError(f"history for {name!r} lacks {x_key!r} or {y_key!r}")
        x_values = np.asarray(history[x_key], dtype=float)
        y_values = np.asarray(history[y_key], dtype=float)
        mask = np.isfinite(x_values) & np.isfinite(y_values)
        if log_y:
            mask &= y_values > 0.0
        ax.plot(x_values[mask], y_values[mask], label=name, linewidth=1.8)

    ax.set_xlabel(x_key.replace("_", " ").title())
    ax.set_ylabel(y_key.replace("_", " ").title())
    if title:
        ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return _save_or_return(fig, output_path)


def plot_coefficient_comparison(
    x_star: np.ndarray,
    estimates: Mapping[str, np.ndarray],
    *,
    title: str = "True and recovered coefficients",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot true coefficients together with one or more recovered vectors."""
    feature_index = np.arange(len(x_star))
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.vlines(feature_index, 0.0, x_star, color="black", alpha=0.45, linewidth=1.0)
    ax.scatter(feature_index, x_star, color="black", s=18, label="x_star", zorder=3)

    for name, estimate in estimates.items():
        if np.asarray(estimate).shape != np.asarray(x_star).shape:
            raise ValueError(f"estimate {name!r} has a shape different from x_star")
        ax.scatter(feature_index, estimate, s=12, alpha=0.7, label=name)

    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Coefficient")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend()
    return _save_or_return(fig, output_path)


def plot_regularization_path(
    mu_values: np.ndarray,
    metric_values: np.ndarray,
    nonzero_count: np.ndarray,
    *,
    metric_label: str,
    title: str | None = None,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot one quality metric and sparsity against the regularization weight."""
    mu_values = np.asarray(mu_values, dtype=float)
    metric_values = np.asarray(metric_values, dtype=float)
    nonzero_count = np.asarray(nonzero_count, dtype=float)
    if not (
        mu_values.shape == metric_values.shape == nonzero_count.shape
        and mu_values.ndim == 1
    ):
        raise ValueError("mu, metric, and nonzero arrays must be one-dimensional")
    if np.any(mu_values <= 0.0):
        raise ValueError("mu values must be positive for a logarithmic path plot")

    fig, left_axis = plt.subplots(figsize=(7.2, 4.6))
    right_axis = left_axis.twinx()

    left_axis.plot(mu_values, metric_values, marker="o", color="#2563eb")
    right_axis.plot(mu_values, nonzero_count, marker="s", color="#dc2626")
    left_axis.set_xscale("log")
    left_axis.set_xlabel("mu")
    left_axis.set_ylabel(metric_label, color="#2563eb")
    right_axis.set_ylabel("Nonzero count", color="#dc2626")
    if title:
        left_axis.set_title(title)
    left_axis.grid(True, alpha=0.25)
    return _save_or_return(fig, output_path)


def plot_mu_path(
    mu_values: np.ndarray,
    test_mse: np.ndarray,
    nonzero_count: np.ndarray,
    *,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Backward-compatible helper for the Diabetes test-MSE mu path."""
    return plot_regularization_path(
        mu_values,
        test_mse,
        nonzero_count,
        metric_label="Test MSE",
        output_path=output_path,
    )
