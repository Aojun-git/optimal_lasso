"""ISTA (Iterative Shrinkage-Thresholding Algorithm) solver for the LASSO problem.

Minimises  F(x) = 0.5 * ||Ax - b||_2^2 + mu * ||x||_1
using proximal gradient descent:
    x_{k+1} = S_{mu/L}(x_k - (1/L) * A^T(A x_k - b))
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from lasso_demo.core import create_history, record_history, soft_threshold


def ista(
    A: np.ndarray,
    b: np.ndarray,
    mu: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Solve LASSO with ISTA (proximal gradient descent).

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
        - ``max_iter``  : maximum iterations (default 1000).
        - ``tol``       : relative objective improvement tolerance (default 1e-8).
        - ``x_star``    : true sparse vector for recovery-error tracking.

    Returns
    -------
    dict with keys ``'x'`` and ``'history'``.
    """
    max_iter = int(config.get("max_iter", 1000))
    tol = float(config.get("tol", 1e-8))
    x_star = config.get("x_star", None)

    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    # Estimate Lipschitz constant L = largest eigenvalue of A^T A
    L = np.linalg.norm(A, ord=2) ** 2

    x = np.zeros(A.shape[1])
    history = create_history()
    start = time.perf_counter()
    prev_obj = np.inf

    for k in range(max_iter):
        # Gradient step
        grad = A.T @ (A @ x - b)
        y = x - (1.0 / L) * grad
        # Proximal step (soft thresholding)
        x = soft_threshold(y, mu / L)

        record_history(
            history,
            iteration=k,
            elapsed_time=time.perf_counter() - start,
            A=A,
            b=b,
            x=x,
            mu=mu,
            x_star=x_star,
        )

        # Convergence check
        curr_obj = history["objective"][-1]
        if abs(prev_obj - curr_obj) / max(abs(prev_obj), 1.0) < tol:
            break
        prev_obj = curr_obj

    return {"x": x, "history": history}
