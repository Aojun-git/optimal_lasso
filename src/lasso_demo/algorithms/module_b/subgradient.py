"""Subgradient Descent solver for the LASSO problem.

Minimises  F(x) = 0.5 * ||Ax - b||_2^2 + mu * ||x||_1
using the sub-gradient of the L1 term and a diminishing step size.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from lasso_demo.core import create_history, record_history


def subgradient(
    A: np.ndarray,
    b: np.ndarray,
    mu: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Solve LASSO with subgradient descent.

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
        - ``alpha0``    : initial step size (default 0.01).
        - ``x_star``    : true sparse vector for recovery-error tracking.

    Returns
    -------
    dict with keys ``'x'`` and ``'history'``.
    """
    max_iter = int(config.get("max_iter", 1000))
    alpha0 = float(config.get("alpha0", 0.01))
    x_star = config.get("x_star", None)

    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m, n = A.shape

    # Estimate L for smooth part (largest eigenvalue of A^T A)
    L = np.linalg.norm(A, ord=2) ** 2

    x = np.zeros(n)
    history = create_history()
    start = time.perf_counter()

    for k in range(max_iter):
        # Gradient of smooth term
        grad = A.T @ (A @ x - b)
        # Subgradient of L1 term: sign(x), with convention sign(0)=0
        subgrad_l1 = np.sign(x)
        # Full subgradient
        g = grad + mu * subgrad_l1

        # Diminishing step size
        step = alpha0 / np.sqrt(k + 1)

        x = x - step * g

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

    return {"x": x, "history": history}
