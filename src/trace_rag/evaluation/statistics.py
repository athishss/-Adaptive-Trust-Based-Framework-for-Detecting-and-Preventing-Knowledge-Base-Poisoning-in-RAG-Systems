"""Statistical analysis helpers for Person C experiments."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def bootstrap_ci(
    values: Sequence[float],
    seed: int = 20260921,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Compute a percentile bootstrap confidence interval."""
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        raise ValueError("values must not be empty")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    rng = np.random.default_rng(seed)

    samples = rng.choice(
        values,
        size=(n_bootstrap, values.size),
        replace=True,
    )

    means = samples.mean(axis=1)

    alpha = 1.0 - confidence

    return (
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1.0 - alpha / 2)),
    )

def mcnemar_exact(
    a_only: int,
    b_only: int,
) -> float:
    """Compute the exact two-sided McNemar p-value."""
    from math import comb

    discordant = a_only + b_only

    if discordant == 0:
        return 1.0

    smaller = min(a_only, b_only)

    probability = sum(
        comb(discordant, k)
        for k in range(smaller + 1)
    ) / (2 ** discordant)

    return min(1.0, 2.0 * probability)
