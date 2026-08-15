"""Pure metric functions over Grad-CAM map arrays.

All functions take (N, H, W) arrays and return floats or (float, float) tuples.
No I/O, no model, no database — this is what makes them unit-testable against
analytical values.
"""

from itertools import combinations

import numpy as np


def mean_pairwise_correlation(cams: np.ndarray) -> tuple[float, float]:
    """Mean and std of all pairwise Pearson correlations over N maps.

    At N=30 there are 435 pairs. A constant (all-identical) map produces nan
    from np.corrcoef because its standard deviation is zero.  nan_to_num
    substitutes 0.0 — a map with no spatial variation carries no information,
    so reporting zero correlation is correct.

    **Production** callers rely on this substitution.
    **Tests** must assert non-degeneracy BEFORE calling this function.
    If constant maps were absorbed silently, a blank-CAM failure (every map
    all-zero) would produce mean=0.0 and trivially pass the ``r < 0.99``
    gate, hiding the real defect.  See TROUBLESHOOTING.md item 14.

    Args:
        cams: Float array of shape (N, H, W).  N >= 2 required.

    Returns:
        Tuple (mean, std) of all C(N, 2) pairwise Pearson correlations.
        Values in [-1, 1].  Constant maps contribute 0.0 to both statistics.
    """
    n = cams.shape[0]
    flat = cams.reshape(n, -1)
    values = [
        float(np.nan_to_num(np.corrcoef(flat[i], flat[j])[0, 1], nan=0.0))
        for i, j in combinations(range(n), 2)
    ]
    arr = np.array(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())
