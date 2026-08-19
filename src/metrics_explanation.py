"""Pure metric functions over Grad-CAM map arrays.

All functions take ``(N, H, W)`` arrays and return floats or ``(float, float)``
tuples.  No I/O, no model, no database — this is what makes them unit-testable
against analytical values (TC7–TC11).

``mean_pairwise_correlation`` is preserved unchanged from Phase 0.  TC16 and
TC17 (the verification gate) depend on its exact signature and nan-handling
behaviour.  See TROUBLESHOOTING.md item 14 before modifying it.
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


def cam_iou(cams: np.ndarray, percentile: float = 80) -> float:
    """Mean pairwise IoU over N binarised Grad-CAM maps.

    Each map is binarised at its own *per-map* percentile — never a fixed
    absolute threshold.  Grad-CAM magnitudes vary across images, so a fixed
    cut is not comparable across samples (CLAUDE.md §1.7).

    Empty union (both masks all-False) returns 0.0.

    Degenerate maps (spatial std < 1e-6, e.g. all-zero outputs after ReLU)
    are excluded from all pairwise comparisons.  A blank map binarised at its
    own 80th percentile (which is 0.0) produces an all-True mask, making every
    IoU = 1.0 — the opposite of the correct interpretation.  Excluding such
    pairs is the conservative, correct choice.  Returns 0.0 when no valid
    pairs remain.

    Args:
        cams: Shape ``(N, H, W)``.
        percentile: Per-map binarisation percentile.  Default 80 (top 20%).

    Returns:
        Mean IoU across valid C(N, 2) pairs.  Range ``[0, 1]``.
        Column ``cam_iou_mean``.
    """
    degenerate = [c.std() < 1e-6 for c in cams]
    masks = [c >= np.percentile(c, percentile) for c in cams]
    vals: list[float] = []
    for a, b in combinations(range(len(masks)), 2):
        if degenerate[a] or degenerate[b]:
            continue
        inter = np.logical_and(masks[a], masks[b]).sum()
        union = np.logical_or(masks[a], masks[b]).sum()
        vals.append(float(inter / union) if union else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def topk_overlap(cams: np.ndarray, k_frac: float = 0.10) -> float:
    """Mean pairwise top-k pixel overlap over N Grad-CAM maps.

    T_t = index set of the top k pixels, k = int(k_frac * H * W).
    Overlap for one pair = |T_a ∩ T_b| / k.

    More permissive than IoU: it ignores region shape and only counts how
    many of the highest-activation pixels are shared.

    Degenerate maps (spatial std < 1e-6) are excluded from all pairwise
    comparisons for the same reason as in ``cam_iou``.  Returns 0.0 when
    no valid pairs remain.

    Args:
        cams: Shape ``(N, H, W)``.
        k_frac: Fraction of pixels to treat as "top".  Default 0.10.

    Returns:
        Mean overlap across valid C(N, 2) pairs.  Range ``[0, 1]``.
        Column ``topk_overlap``.
    """
    k = int(k_frac * cams[0].size)
    degenerate = [c.std() < 1e-6 for c in cams]
    tops = [set(np.argpartition(c.ravel(), -k)[-k:]) for c in cams]
    vals = [
        len(tops[a] & tops[b]) / k
        for a, b in combinations(range(len(tops)), 2)
        if not (degenerate[a] or degenerate[b])
    ]
    return float(np.mean(vals)) if vals else 0.0


def variability_map(cams: np.ndarray) -> np.ndarray:
    """Pixel-wise standard deviation across N Grad-CAM maps.

    Reveals *where* the explanation is spatially unstable, making an
    unstable case legible to a reviewer.  Saved as a heatmap PNG by the
    pipeline; not stored as a scalar in the database.

    Args:
        cams: Shape ``(N, H, W)``.

    Returns:
        Float32 array of shape ``(H, W)``.
    """
    return cams.std(axis=0).astype(np.float32)


def upsample_cams(cams: np.ndarray, size: int = 224) -> np.ndarray:
    """Bilinearly upsample N Grad-CAM maps to ``size × size``.

    Raw layer4 maps are 7×7.  All explanation metrics are computed at
    224×224 to match the input image resolution.  The raw 7×7 form is what
    gets archived to ``.npz`` files (~3 KB each).

    Args:
        cams: Shape ``(N, H, W)``.
        size: Target side length in pixels.

    Returns:
        Float32 array of shape ``(N, size, size)``.
    """
    import cv2
    return np.stack(
        [
            cv2.resize(
                c.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR
            )
            for c in cams
        ],
        axis=0,
    )


def compute_all(cams: np.ndarray, cfg: dict) -> dict:
    """Upsample raw Grad-CAM maps and compute all explanation-side metrics.

    Upsamples to ``gradcam.upsample_size`` first, then computes every
    pairwise metric over the full-resolution maps.  Returns a dict keyed by
    ``explanations`` table column names.

    Filesystem paths (``cam_npz_path``, ``mean_png_path``,
    ``variance_png_path``) are the pipeline's responsibility and are not
    included in this dict.

    Args:
        cams: Shape ``(N, H, W)``, raw 7×7 float32 maps from layer4.
        cfg: Full config dict.  Reads ``gradcam.upsample_size``,
             ``gradcam.iou_percentile``, ``gradcam.topk_k``.

    Returns:
        Dict with keys ``cam_corr_mean``, ``cam_corr_std``,
        ``cam_iou_mean``, ``topk_overlap``.
    """
    size       = cfg["gradcam"]["upsample_size"]
    percentile = cfg["gradcam"]["iou_percentile"]
    topk_k     = cfg["gradcam"]["topk_k"]

    upsampled             = upsample_cams(cams, size=size)
    corr_mean, corr_std   = mean_pairwise_correlation(upsampled)
    iou                   = cam_iou(upsampled, percentile=percentile)
    topk                  = topk_overlap(upsampled, k_frac=topk_k)

    # Count pairs that were excluded from IoU / topk due to degenerate maps.
    # A pair is degenerate if either constituent map has spatial std < 1e-6
    # (e.g. all-zero output after ReLU).  C(N,2) − C(n_clean,2) gives the
    # exact count without an O(N²) loop.
    N       = len(upsampled)
    n_clean = int(sum(1 for c in upsampled if c.std() >= 1e-6))
    n_degenerate_pairs = N * (N - 1) // 2 - n_clean * (n_clean - 1) // 2

    return {
        "cam_corr_mean":      corr_mean,
        "cam_corr_std":       corr_std,
        "cam_iou_mean":       iou,
        "topk_overlap":       topk,
        "n_degenerate_pairs": n_degenerate_pairs,
    }
