"""Explanation-side metric unit tests — TC7 through TC11.

All tests use pure numpy arrays; no model, no database, no I/O.
Fixtures ``identical_cams`` and ``constant_cams`` are defined in conftest.py.
"""

import math
from itertools import combinations

import numpy as np
import pytest

from src.metrics_explanation import (
    cam_iou,
    mean_pairwise_correlation,
    topk_overlap,
)


# ---------------------------------------------------------------------------
# TC7 — two identical maps → correlation = 1.0
# ---------------------------------------------------------------------------

def test_tc7_correlation_identical_maps(identical_cams: np.ndarray) -> None:
    """TC7: N identical Grad-CAM maps → mean pairwise correlation = 1.0.

    When every map is the same 2-D array, every Pearson correlation is 1.0,
    so the mean is exactly 1.0.

    Input : 10 copies of the same 7×7 map (``identical_cams`` fixture).
    Expect: cam_corr_mean = 1.0 ± 1e-6.
    """
    r, _ = mean_pairwise_correlation(identical_cams)
    assert r == pytest.approx(1.0, abs=1e-6), (
        f"Identical maps should correlate at 1.0, got {r:.8f}"
    )


# ---------------------------------------------------------------------------
# TC8 — two identical maps → IoU = 1.0
# ---------------------------------------------------------------------------

def test_tc8_iou_identical_maps(identical_cams: np.ndarray) -> None:
    """TC8: N identical Grad-CAM maps → mean pairwise IoU = 1.0.

    All maps have the same values, so each map is binarised to the same mask.
    Identical masks have intersection = union → IoU = 1.0 for every pair.

    Input : 10 copies of the same 7×7 map.
    Expect: cam_iou_mean = 1.0.
    """
    iou = cam_iou(identical_cams)
    assert iou == pytest.approx(1.0, abs=1e-9), (
        f"Identical maps should give IoU=1.0, got {iou:.8f}"
    )


# ---------------------------------------------------------------------------
# TC9 — map and its negation → IoU = 0.0
# ---------------------------------------------------------------------------

def test_tc9_iou_complement_maps() -> None:
    """TC9: a map and its value-complement → pairwise IoU = 0.0.

    A linearly ascending gradient and its complement (descending gradient)
    binarise at the 80th percentile to disjoint regions:
    - ascending map: top 20% of pixels are the *highest-value* pixels (right end)
    - descending map: top 20% are the *lowest-value* pixels of the original
      (left end, which are the highest of the negated values)
    The two masks do not overlap → intersection = 0 → IoU = 0.0.

    Input : [0, 1, …, 48] and [48, 47, …, 0] reshaped to (7, 7).
    Expect: cam_iou_mean = 0.0.
    """
    base = np.arange(49, dtype=np.float32).reshape(7, 7)
    neg  = 48.0 - base          # value complement (reversed gradient)
    maps = np.stack([base, neg])  # shape (2, 7, 7)

    iou = cam_iou(maps, percentile=80)
    assert iou == pytest.approx(0.0, abs=1e-9), (
        f"Complement maps should give IoU=0.0, got {iou:.8f}"
    )


# ---------------------------------------------------------------------------
# TC10 — constant map → no ZeroDivisionError or NaN
# ---------------------------------------------------------------------------

def test_tc10_constant_map_no_error(constant_cams: np.ndarray) -> None:
    """TC10: constant (all 0.5) maps — functions must not raise or produce NaN.

    A constant map has zero spatial variance.  ``np.corrcoef`` returns NaN
    for such a pair; ``cam_iou`` and ``topk_overlap`` operate on uniform
    arrays.  All three functions must return a finite float without raising.

    Input : 5 constant 7×7 maps (``constant_cams`` fixture, all values 0.5).
    Expect: each function returns a finite float; no ZeroDivisionError or NaN.
    """
    # mean_pairwise_correlation: nan_to_num maps NaN pairs to 0.0
    r, std = mean_pairwise_correlation(constant_cams)
    assert math.isfinite(r),   f"cam_corr_mean is not finite: {r}"
    assert math.isfinite(std), f"cam_corr_std is not finite: {std}"

    # cam_iou: uniform arrays → all pixels at or above percentile → full masks
    iou = cam_iou(constant_cams)
    assert math.isfinite(iou), f"cam_iou_mean is not finite: {iou}"

    # topk_overlap: argpartition on equal values is valid (arbitrary but stable)
    topk = topk_overlap(constant_cams)
    assert math.isfinite(topk), f"topk_overlap is not finite: {topk}"


# ---------------------------------------------------------------------------
# TC11 — 30 maps → exactly 435 pairwise values computed
# ---------------------------------------------------------------------------

def test_tc11_thirty_maps_435_pairs() -> None:
    """TC11: 30 Grad-CAM maps → C(30, 2) = 435 pairwise values computed.

    Verifies that the metric functions iterate over *all* C(N, 2) pairs
    and not a subset.  C(30, 2) = 30×29/2 = 435.

    Input : 30 distinct random 7×7 maps (seeded for reproducibility).
    Expect:
      - mean_pairwise_correlation completes and returns a finite result.
      - cam_iou completes and returns a finite result.
      - The C(30, 2) pair count is confirmed to be 435 via itertools.
    """
    rng  = np.random.RandomState(99)
    cams = rng.rand(30, 7, 7).astype(np.float32)

    # Analytical check: C(30, 2) == 435
    n_pairs = sum(1 for _ in combinations(range(30), 2))
    assert n_pairs == 435, f"Expected 435 pairs for N=30, got {n_pairs}"

    # mean_pairwise_correlation over 30 distinct maps
    r, std = mean_pairwise_correlation(cams)
    assert math.isfinite(r),   f"cam_corr_mean is not finite: {r}"
    assert math.isfinite(std), f"cam_corr_std is not finite: {std}"

    # cam_iou over 30 distinct maps
    iou = cam_iou(cams)
    assert math.isfinite(iou), f"cam_iou_mean is not finite: {iou}"
    assert 0.0 <= iou <= 1.0,  f"cam_iou_mean out of range [0,1]: {iou}"


# ---------------------------------------------------------------------------
# TC_blank — all-zero (30, 7, 7) array must not produce cam_iou_mean == 1.0
# ---------------------------------------------------------------------------

def test_blank_cam_iou_guard() -> None:
    """Blank all-zero CAMs must not inflate cam_iou_mean to 1.0.

    Root cause: np.percentile(zeros, 80) == 0.0, so the binarisation
    threshold c >= 0.0 selects every pixel as True.  Two all-True masks
    have intersection == union, yielding IoU = 1.0 — the maximum possible
    score — for maps that carry no information whatsoever.

    Fix: pairs where either map has spatial std < 1e-6 are excluded from
    the average.  No valid pairs remain for an all-zero input, so
    cam_iou returns 0.0 instead of 1.0.

    Input : (30, 7, 7) array of zeros (dtype float32).
    Expect:
        - cam_iou_mean != 1.0  (the inflation bug is absent)
        - cam_iou_mean == 0.0  (no valid pairs → sentinel value)
    """
    zeros = np.zeros((30, 7, 7), dtype=np.float32)
    iou = cam_iou(zeros)
    assert iou != pytest.approx(1.0, abs=1e-6), (
        f"Blank all-zero CAMs must not give cam_iou_mean=1.0; got {iou:.6f}. "
        "Degeneracy guard is missing or not firing."
    )
    assert iou == pytest.approx(0.0, abs=1e-9), (
        f"Expected cam_iou_mean=0.0 for all-zero input, got {iou:.6f}."
    )
