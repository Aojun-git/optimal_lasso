"""Copy this interface when implementing a module B solver."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def solver_template(
    A: np.ndarray,
    b: np.ndarray,
    mu: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return {'x': final_vector, 'history': common_history}."""
    raise NotImplementedError("Implement Subgradient, ISTA, or FISTA here")

