"""Reusable plotting functions with a consistent visual style.

Provides all 8 figure types required by the assignment:
  Fig.1 – L1/L2 geometric sparsity illustration
  Fig.2 – True vs recovered coefficients
  Fig.3 – Objective value vs iteration
  Fig.4 – Relative objective gap (semi-log)
  Fig.5 – Objective value vs CPU time
  Fig.6 – mu vs sparsity & test MSE
  Fig.7 – ADMM primal & dual residuals
  Fig.8 – E2006 subset results
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _save_or_return(fig: plt.Figure, output_path: str | Path | None) -> plt.Figure:
    fig.tight_layout()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Fig.1 – L1 / L2 geometric sparsity illustration
# ---------------------------------------------------------------------------

def plot_l1_l2_geometry(
    *,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Show why L1 produces sparse solutions: diamond vs circle in 2-D."""
    theta = np.linspace(0, 2 * np.pi, 400)

    fig, ax = plt.subplots(figsize=(6.0, 6.0))

    # L1 ball (diamond)
    t = np.linspace(-1, 1, 200)
    l1_x = np.concatenate([t, np.ones_like(t), t[::-1], -np.ones_like(t)])
    l1_y = np.concatenate([1 - np.abs(t), np.abs(t[::-1]), -(1 - np.abs(t)), -np.abs(t)])
    ax.plot(l1_x, l1_y, color="#2563eb", linewidth=2.2, label=r"$\|x\|_1 \leq 1$ (L1 ball)")

    # L2 ball (circle)
    ax.plot(np.cos(theta), np.sin(theta), color="#dc2626", linewidth=2.2,
            label=r"$\|x\|_2 \leq 1$ (L2 ball)")

    # Contour lines of least-squares loss (ellipses) centred off-axis
    centre = np.array([1.8, 1.2])
    for s in [0.3, 0.6, 1.0]:
        ell_x = centre[0] + s * np.cos(theta)
        ell_y = centre[1] + s * np.sin(theta)
        ax.plot(ell_x, ell_y, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    # Mark the L1 tangent point (on axis => sparse)
    ax.plot([0, 1], [1, 0], "k--", linewidth=0.8, alpha=0.5)
    ax.plot(0, 1, "o", color="#2563eb", markersize=8, zorder=5)
    ax.annotate("L1 touches axis\n→ sparse solution",
                xy=(0, 1), xytext=(0.35, 1.35),
                fontsize=10, color="#2563eb",
                arrowprops=dict(arrowstyle="->", color="#2563eb"))

    ax.set_xlim(-1.6, 2.4)
    ax.set_ylim(-1.6, 2.0)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("L1 vs L2 Regularisation Geometry")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.2)
    return _save_or_return(fig, output_path)


# ---------------------------------------------------------------------------
# Fig.2 – True vs recovered coefficients
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Fig.3 & Fig.5 – Generic convergence plot (objective vs iteration / time)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Fig.4 – Relative objective gap (semi-log)
# ---------------------------------------------------------------------------

def plot_relative_gap(
    histories: Mapping[str, Mapping[str, list[float | int]]],
    *,
    x_key: str = "iteration",
    title: str | None = None,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot relative objective gap on a semi-log scale for multiple algorithms."""
    # Compute relative gap with a common reference
    final_objectives = [
        float(h["objective"][-1]) for h in histories.values() if h["objective"]
    ]
    if not final_objectives:
        raise ValueError("no objective data")
    reference = min(final_objectives)
    denominator = max(abs(reference), np.finfo(float).eps)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for name, history in histories.items():
        x_values = np.asarray(history[x_key], dtype=float)
        obj_values = np.asarray(history["objective"], dtype=float)
        gap = np.maximum((obj_values - reference) / denominator, np.finfo(float).eps)
        mask = np.isfinite(x_values) & np.isfinite(gap) & (gap > 0)
        ax.plot(x_values[mask], gap[mask], label=name, linewidth=1.8)

    ax.set_xlabel(x_key.replace("_", " ").title())
    ax.set_ylabel("Relative objective gap")
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Relative objective gap (semi-log)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return _save_or_return(fig, output_path)


# ---------------------------------------------------------------------------
# Fig.6 – mu path (regularisation path)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Fig.7 – ADMM primal & dual residuals
# ---------------------------------------------------------------------------

def plot_admm_residuals(
    history: Mapping[str, list[float | int]],
    *,
    title: str = "ADMM: primal and dual residuals",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot ADMM primal residual ||x-z|| and dual residual rho*||z-z_old||."""
    iterations = np.asarray(history["iteration"], dtype=float)
    primal = np.asarray(history.get("primal_residual", []), dtype=float)
    dual = np.asarray(history.get("dual_residual", []), dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    valid_primal = np.isfinite(primal)
    valid_dual = np.isfinite(dual)
    if valid_primal.any():
        ax.plot(iterations[valid_primal], primal[valid_primal],
                label="Primal residual  $\\|x - z\\|$", linewidth=1.8, color="#2563eb")
    if valid_dual.any():
        ax.plot(iterations[valid_dual], dual[valid_dual],
                label="Dual residual  $\\rho\\|z^k - z^{k-1}\\|$", linewidth=1.8,
                color="#dc2626", linestyle="--")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return _save_or_return(fig, output_path)


# ---------------------------------------------------------------------------
# Fig.8 – E2006 subset summary (time vs test MSE + nonzero bar)
# ---------------------------------------------------------------------------

def plot_e2006_summary(
    solver_names: list[str],
    test_mse: list[float],
    nonzero_counts: list[int],
    cpu_times: list[float],
    *,
    title: str = "E2006 subset: algorithm comparison",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart comparing solvers on E2006 subset: test MSE, nonzero count, CPU time."""
    x_pos = np.arange(len(solver_names))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(9.0, 5.0))
    bars1 = ax1.bar(x_pos - width, test_mse, width, label="Test MSE", color="#2563eb")
    ax1.set_ylabel("Test MSE", color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb")

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x_pos, nonzero_counts, width, label="Nonzero count", color="#f97316")
    ax2.set_ylabel("Nonzero count", color="#f97316")
    ax2.tick_params(axis="y", labelcolor="#f97316")

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.15))
    bars3 = ax3.bar(x_pos + width, cpu_times, width, label="CPU time (s)", color="#10b981")
    ax3.set_ylabel("CPU time (s)", color="#10b981")
    ax3.tick_params(axis="y", labelcolor="#10b981")

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(solver_names, rotation=15)
    ax1.set_title(title)

    # Combined legend
    all_bars = [bars1, bars2, bars3]
    labels = [b.get_label() for b in all_bars]
    ax1.legend(all_bars, labels, loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.15, axis="y")
    return _save_or_return(fig, output_path)


# ---------------------------------------------------------------------------
# Summary table plot (bar chart for result comparison)
# ---------------------------------------------------------------------------

def plot_result_bars(
    solver_names: list[str],
    metrics_dict: dict[str, list[float]],
    *,
    title: str = "Algorithm comparison",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Grouped bar chart comparing multiple metrics across solvers."""
    n_solvers = len(solver_names)
    n_metrics = len(metrics_dict)
    x_pos = np.arange(n_solvers)
    width = 0.8 / max(n_metrics, 1)

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for i, (metric_name, values) in enumerate(metrics_dict.items()):
        offset = (i - n_metrics / 2 + 0.5) * width
        ax.bar(x_pos + offset, values, width, label=metric_name)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(solver_names, rotation=15)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15, axis="y")
    return _save_or_return(fig, output_path)
