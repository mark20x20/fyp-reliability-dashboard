"""Prediction-side metric unit tests — TC1 through TC6.

All tests use pure numpy arrays; no model, no database, no I/O.
Each test exercises one analytical boundary condition with a known closed-form
answer so that any formula error is immediately detectable.
"""

import math

import numpy as np
import pytest

from src.metrics_prediction import (
    confidence,
    mutual_information,
    prediction_agreement,
    predictive_entropy,
    predictive_variance,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_N = 30         # MC-Dropout passes (matches config default)
_C = 10         # number of classes (Imagenette)


def _uniform_probs(n: int = _N, c: int = _C) -> np.ndarray:
    """N passes each predicting a uniform distribution over C classes."""
    return np.full((n, c), 1.0 / c, dtype=np.float64)


def _onehot_probs(cls: int, n: int = _N, c: int = _C) -> np.ndarray:
    """N identical passes each placing all mass on class ``cls``."""
    p = np.zeros((n, c), dtype=np.float64)
    p[:, cls] = 1.0
    return p


# ---------------------------------------------------------------------------
# TC1 — one-hot distribution → entropy = 0
# ---------------------------------------------------------------------------

def test_tc1_entropy_onehot() -> None:
    """TC1: one-hot [1, 0, …, 0] → predictive_entropy = 0.0.

    A distribution with all mass on a single class has zero entropy.
    The eps guard prevents 0·log(0) = NaN; the contribution is ~eps·log(eps)
    ≈ 0 for eps = 1e-12, so the result is numerically indistinguishable from 0.

    Input : 30 identical passes, class 0 probability = 1.0.
    Expect: entropy < 1e-10 (effectively zero).
    """
    probs = _onehot_probs(cls=0)
    h = predictive_entropy(probs)
    assert h == pytest.approx(0.0, abs=1e-10), (
        f"One-hot entropy should be ~0, got {h}"
    )


# ---------------------------------------------------------------------------
# TC2 — uniform distribution → entropy = ln(C)
# ---------------------------------------------------------------------------

def test_tc2_entropy_uniform() -> None:
    """TC2: uniform over 10 classes → predictive_entropy = ln(10) ≈ 2.3026.

    The maximum-entropy distribution spreads mass equally across all classes.

    Input : 30 identical uniform passes.
    Expect: entropy = ln(10) ± 1e-6.
    """
    probs = _uniform_probs()
    h = predictive_entropy(probs)
    assert h == pytest.approx(math.log(10), abs=1e-6), (
        f"Uniform entropy should be ln(10)={math.log(10):.6f}, got {h:.6f}"
    )


# ---------------------------------------------------------------------------
# TC3 — identical passes → pred_variance = 0
# ---------------------------------------------------------------------------

def test_tc3_variance_identical_passes() -> None:
    """TC3: N identical softmax distributions → predictive_variance = 0.0.

    When every pass produces the same output, the variance of the predicted-
    class channel is zero.

    Input : 30 identical passes (uniform distribution).
    Expect: pred_variance = 0.0.
    """
    probs = _uniform_probs()
    v = predictive_variance(probs)
    assert v == pytest.approx(0.0, abs=1e-12), (
        f"Variance of identical passes should be 0, got {v}"
    )


# ---------------------------------------------------------------------------
# TC4 — identical passes → mutual_information = 0
# ---------------------------------------------------------------------------

def test_tc4_mi_identical_passes() -> None:
    """TC4: N identical softmax distributions → mutual_information = 0.0.

    When every pass agrees, there is no epistemic uncertainty.
    I = H[p̄] - (1/N)ΣH[p_t] = H[p̄] - H[p̄] = 0.

    Input : 30 identical passes (uniform distribution).
    Expect: mutual_information = 0.0.
    """
    probs = _uniform_probs()
    mi = mutual_information(probs)
    assert mi == pytest.approx(0.0, abs=1e-10), (
        f"MI of identical passes should be 0, got {mi}"
    )


# ---------------------------------------------------------------------------
# TC5 — all passes agree on class 3 → pred_agreement = 1.0
# ---------------------------------------------------------------------------

def test_tc5_agreement_full_consensus() -> None:
    """TC5: all 30 passes predict class 3 → prediction_agreement = 1.0.

    Input : 30 one-hot passes, class 3.
    Expect: pred_agreement = 1.0.
    """
    probs = _onehot_probs(cls=3)
    a = prediction_agreement(probs)
    assert a == pytest.approx(1.0), (
        f"Full consensus should give pred_agreement=1.0, got {a}"
    )


# ---------------------------------------------------------------------------
# TC6 — split vote → pred_agreement = 0.5
# ---------------------------------------------------------------------------

def test_tc6_agreement_split_vote() -> None:
    """TC6: 15 passes predict class 3, 15 predict class 7 → pred_agreement = 0.5.

    Exactly half the passes vote for each of two classes.  The modal class
    is either 3 or 7 (tied); the agreement is 15/30 = 0.5 regardless.

    Input : 15 one-hot class-3 rows + 15 one-hot class-7 rows.
    Expect: pred_agreement = 0.5.
    """
    p3 = _onehot_probs(cls=3, n=15)
    p7 = _onehot_probs(cls=7, n=15)
    probs = np.vstack([p3, p7])        # (30, 10)
    a = prediction_agreement(probs)
    assert a == pytest.approx(0.5), (
        f"50/50 split should give pred_agreement=0.5, got {a}"
    )
