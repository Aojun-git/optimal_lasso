"""Coordinate Descent solver for the LASSO problem.

Minimises  F(x) = 0.5 * ||Ax - b||_2^2 + mu * ||x||_1
by updating one coordinate at a time with the closed-form soft-threshold
solution while keeping the residual vector up to date.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from lasso_demo.core import create_history, record_history, soft_threshold


def coordinate_descent(
    A: np.ndarray,
    b: np.ndarray,
    mu: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Solve LASSO with coordinate descent.

    Parameters
    ----------
    A : ndarray, shape (m, n)
        Design matrix.
    b : ndarray, shape (m,)
        Observation vector.
    mu : float
        L1 regularisation weight (must be >= 0).
    config : mapping
        Optional keys:
        - ``max_iter``  : maximum number of full epochs (default 500).
        - ``tol``       : relative objective improvement tolerance (default 1e-8).
        - ``x_star``    : true sparse vector for recovery-error tracking.

    Returns
    -------
    dict with keys ``'x'`` and ``'history'``.
    """
    max_iter = int(config.get("max_iter", 500))
    tol = float(config.get("tol", 1e-8))
    x_star = config.get("x_star", None)

    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m, n = A.shape

    # Pre-compute column norms squared
    col_norm_sq = np.sum(A ** 2, axis=0)
    col_norm_sq[col_norm_sq == 0.0] = 1.0  # avoid division by zero

    # Initialise
    x = np.zeros(n)
    r = b - A @ x  # residual
    history = create_history()
    start = time.perf_counter()

    prev_obj = np.inf

    for epoch in range(max_iter):
        for j in range(n):
            # Add back the contribution of x[j] to the residual
            r = r + A[:, j] * x[j]
            # Compute rho_j = a_j^T * r
            rho_j = A[:, j].dot(r)
            # Soft-threshold update
            x_new_j = soft_threshold(np.array([rho_j]), mu)[0] / col_norm_sq[j]
            # Subtract the new contribution
            r = r - A[:, j] * x_new_j
            x[j] = x_new_j

        # Record after each full epoch
        record_history(
            history,
            iteration=epoch,
            elapsed_time=time.perf_counter() - start,
            A=A,
            b=b,
            x=x,
            mu=mu,
            x_star=x_star,
        )

        # Check convergence
        curr_obj = history["objective"][-1]
        if abs(prev_obj - curr_obj) / max(abs(prev_obj), 1.0) < tol:
            break
        prev_obj = curr_obj

    return {"x": x, "history": history}
