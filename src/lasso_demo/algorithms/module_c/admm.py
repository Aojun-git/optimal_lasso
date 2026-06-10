"""ADMM (Alternating Direction Method of Multipliers) solver for the LASSO problem.

Solves the split formulation:
    min_{x,z}  0.5 * ||Ax - b||_2^2 + mu * ||z||_1
    s.t.  x - z = 0

ADMM scaled-form updates:
    x^{k+1} = (A^T A + rho I)^{-1} (A^T b + rho (z^k - u^k))
    z^{k+1} = S_{mu/rho}(x^{k+1} + u^k)
    u^{k+1} = u^k + x^{k+1} - z^{k+1}

Supports adaptive rho (Boyd et al. 2010) for improved convergence.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from lasso_demo.core import create_history, record_history, soft_threshold


def _build_factorisation(A: np.ndarray, rho: float, m: int, n: int):
    """Build Cholesky factorisation for the x-update.

    Uses Woodbury identity when m < n to reduce system size.
    """
    Atb = A.T @ b if False else None  # computed outside

    use_woodbury = m < n
    if use_woodbury:
        S = A @ A.T + rho * np.eye(m)
        cho = np.linalg.cholesky(S)
    else:
        M = A.T @ A + rho * np.eye(n)
        cho = np.linalg.cholesky(M)

    return cho, use_woodbury


def _solve_x_update(A, rhs, rho, cho, use_woodbury, m, n):
    """Solve the x-update linear system using pre-computed factorisation."""
    if use_woodbury:
        w = np.linalg.solve(cho, A @ rhs)
        w = np.linalg.solve(cho.T, w)
        x = (rhs - A.T @ w) / rho
    else:
        y = np.linalg.solve(cho, rhs)
        x = np.linalg.solve(cho.T, y)
    return x


def admm(
    A: np.ndarray,
    b: np.ndarray,
    mu: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Solve LASSO with ADMM.

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
        - ``max_iter``      : maximum iterations (default 1000).
        - ``rho``           : ADMM penalty parameter (default 1.0).
        - ``adaptive_rho``  : enable adaptive rho adjustment (default True).
        - ``tol``           : convergence tolerance on primal+dual residuals (default 1e-6).
        - ``x_star``        : true sparse vector for recovery-error tracking.

    Returns
    -------
    dict with keys ``'x'`` and ``'history'``.
    """
    max_iter = int(config.get("max_iter", 1000))
    rho = float(config.get("rho", 1.0))
    adaptive_rho = bool(config.get("adaptive_rho", True))
    tol = float(config.get("tol", 1e-6))
    x_star = config.get("x_star", None)

    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m, n = A.shape

    # Pre-compute A^T b (constant part of x-update rhs)
    Atb = A.T @ b

    # Initial factorisation
    cho, use_woodbury = _build_factorisation(A, rho, m, n)

    # Initialise primal, split variable, and dual
    x = np.zeros(n)
    z = np.zeros(n)
    u = np.zeros(n)

    # Adaptive rho parameters (Boyd et al. 2010)
    mu_rho = 10.0       # scaling threshold
    tau_incr = 2.0       # rho increase factor
    tau_decr = 2.0       # rho decrease factor

    history = create_history()
    start = time.perf_counter()

    for k in range(max_iter):
        # --- x-update ---
        rhs = Atb + rho * (z - u)
        x = _solve_x_update(A, rhs, rho, cho, use_woodbury, m, n)

        # --- z-update: soft-threshold ---
        z_old = z.copy()
        z = soft_threshold(x + u, mu / rho)

        # --- u-update: dual ascent ---
        u = u + x - z

        # Primal and dual residuals
        primal_res = float(np.linalg.norm(x - z))
        dual_res = float(rho * np.linalg.norm(z - z_old))

        # --- Adaptive rho adjustment ---
        if adaptive_rho and k > 0 and k % 10 == 0:
            if primal_res > mu_rho * dual_res:
                # Primal residual too large → increase rho
                rho_new = rho * tau_incr
                cho, use_woodbury = _build_factorisation(A, rho_new, m, n)
                u = u * (rho / rho_new)   # scale dual variable
                rho = rho_new
            elif dual_res > mu_rho * primal_res:
                # Dual residual too large → decrease rho
                rho_new = rho / tau_decr
                cho, use_woodbury = _build_factorisation(A, rho_new, m, n)
                u = u * (rho / rho_new)   # scale dual variable
                rho = rho_new

        record_history(
            history,
            iteration=k,
            elapsed_time=time.perf_counter() - start,
            A=A,
            b=b,
            x=x,
            mu=mu,
            x_star=x_star,
            primal_residual=primal_res,
            dual_residual=dual_res,
        )

        # Convergence check
        if primal_res < tol and dual_res < tol:
            break

    return {"x": x, "history": history}
