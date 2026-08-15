"""Pure metric functions over MC-Dropout probability arrays.

All functions take a ``(N, num_classes)`` float array and return floats.
No I/O, no model, no database — this is what makes them unit-testable
against analytical values (TC1–TC6).

Notation: N stochastic passes; p_t ∈ R^C is the softmax output of pass t;
p̄ = (1/N) Σ p_t is the mean predictive distribution;
ĉ = argmax_c p̄_c is the predicted class.
"""

from __future__ import annotations

import numpy as np


def confidence(probs: np.ndarray) -> float:
    """Maximum softmax probability of the mean predictive distribution.

    conf = max_c p̄_c

    Args:
        probs: Shape ``(N, num_classes)``.  Each row is one softmax output.

    Returns:
        Float in ``[1/num_classes, 1.0]``.  Column ``confidence``.
    """
    return float(probs.mean(axis=0).max())


def predictive_entropy(probs: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy of the mean predictive distribution (natural log).

    H[p̄] = -Σ_c p̄_c · ln(p̄_c)

    Args:
        probs: Shape ``(N, num_classes)``.
        eps: Numerical guard against ``0 · log(0)`` producing NaN.

    Returns:
        Float in ``[0, ln(num_classes)]``.  Column ``entropy``.
    """
    p = probs.mean(axis=0)
    return float(-(p * np.log(p + eps)).sum())


def mutual_information(probs: np.ndarray, eps: float = 1e-12) -> float:
    """BALD: predictive entropy minus expected per-pass entropy.

    I = H[p̄] - (1/N) Σ_t H[p_t]

    Separates epistemic uncertainty (disagreement between passes) from
    total uncertainty.  Small negative values from floating-point error
    are clamped to 0.0 — mutual information cannot be negative.

    Args:
        probs: Shape ``(N, num_classes)``.
        eps: Numerical guard for the log.

    Returns:
        Float in ``[0, predictive_entropy(probs)]``.
        Column ``mutual_information``.
    """
    mean_p = probs.mean(axis=0)
    total    = float(-(mean_p * np.log(mean_p + eps)).sum())
    expected = float(-(probs * np.log(probs + eps)).sum(axis=1).mean())
    return max(0.0, total - expected)


def predictive_variance(probs: np.ndarray) -> float:
    """Variance of the predicted-class probability across N passes.

    σ² = (1/N) Σ_t (p_{t,ĉ} - p̄_ĉ)²  where ĉ = argmax p̄

    Variance is taken over the **predicted class channel only** (ĉ),
    not aggregated over all classes.

    Args:
        probs: Shape ``(N, num_classes)``.

    Returns:
        Float in ``[0, 0.25]``.  Column ``pred_variance``.
    """
    c_hat = int(probs.mean(axis=0).argmax())
    return float(probs[:, c_hat].var())


def prediction_agreement(probs: np.ndarray) -> float:
    """Fraction of passes that predicted the modal class.

    a = (1/N) · max_c Σ_t 1[argmax p_t = c]

    This is the **control variable** for the central analysis.  The claim
    that explanation instability exists independently of prediction
    instability is tested on the subset where ``pred_agreement == 1.0``.

    Args:
        probs: Shape ``(N, num_classes)``.

    Returns:
        Float in ``[1/N, 1.0]``.  Column ``pred_agreement``.
    """
    per_pass = probs.argmax(axis=1)
    return float(
        np.bincount(per_pass, minlength=probs.shape[1]).max() / len(per_pass)
    )


def compute_all(
    probs: np.ndarray,
    true_label: str | None,
    class_names: list[str],
) -> dict:
    """Compute all prediction-side metrics and return a dict keyed by schema columns.

    The returned dict maps directly to column names in the ``predictions``
    table (``db/schema.sql``).  ``correct`` is ``None`` for OOD and uploaded
    images, which have no ground-truth label in the Imagenette label space.

    Args:
        probs: Shape ``(N, num_classes)``.
        true_label: Ground-truth class name, or ``None`` for OOD / uploaded.
        class_names: Ordered list of class name strings; index matches column
                     index in ``probs``.

    Returns:
        Dict with keys: ``pred_label``, ``correct``, ``confidence``,
        ``entropy``, ``pred_variance``, ``pred_agreement``,
        ``mutual_information``.
    """
    mean_p    = probs.mean(axis=0)
    pred_idx  = int(mean_p.argmax())
    pred_label = class_names[pred_idx]

    correct: int | None = (
        None if true_label is None else int(pred_label == true_label)
    )

    return {
        "pred_label":        pred_label,
        "correct":           correct,
        "confidence":        confidence(probs),
        "entropy":           predictive_entropy(probs),
        "pred_variance":     predictive_variance(probs),
        "pred_agreement":    prediction_agreement(probs),
        "mutual_information": mutual_information(probs),
    }
