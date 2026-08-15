"""Reference baselines for interpreting explanation-side metrics.

Without a reference, "IoU 0.42" is uninterpretable.

| ID | Baseline  | Method                                       | Expected         |
|----|-----------|----------------------------------------------|------------------|
| B1 | Upper     | Dropout off, same image, two maps            | corr 1.0, IoU 1.0|
| B2 | Lower     | 1,000 pairs of Gaussian-smoothed random maps | corr ≈ 0, IoU ≈ 0.2|
| B3 | Cross-img | 1,000 pairs from different images, same class| between B2 and B1|

B1 also doubles as the implementation sanity check: if it is not 1.000,
there is a bug in the Grad-CAM path and no further results should be collected.

B3 supplies the default ``TH_IOU`` for the risk-flag threshold
(``config.risk_flag.iou_threshold``).  It is measured from data rather than
chosen arbitrarily.

All baseline_* functions return ``{"correlation": float, "iou": float}``.
``compute_all`` runs all three and writes rows to the ``baselines`` table.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import gaussian_filter

if TYPE_CHECKING:
    import torch.nn as nn
    from torch.utils.data import Dataset
    from src.repository import Repository

from models.resnet_dropout import get_target_layer
from src.gradcam_generator import generate_single_cam
from src.metrics_explanation import upsample_cams

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise(cam: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Min-max normalise a 2-D array to [0, 1].  Constant maps → all zeros."""
    lo, hi = cam.min(), cam.max()
    return np.zeros_like(cam) if hi - lo < eps else (cam - lo) / (hi - lo)


def _iou_pair(a: np.ndarray, b: np.ndarray, percentile: float = 80) -> float:
    """IoU for a single pair of 2-D maps binarised at their own percentile."""
    ma = a >= np.percentile(a, percentile)
    mb = b >= np.percentile(b, percentile)
    inter = np.logical_and(ma, mb).sum()
    union = np.logical_or(ma, mb).sum()
    return float(inter / union) if union else 0.0


def _corr_pair(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation for a single pair; NaN → 0.0."""
    return float(np.nan_to_num(np.corrcoef(a.ravel(), b.ravel())[0, 1], nan=0.0))


# ---------------------------------------------------------------------------
# B1 — upper bound
# ---------------------------------------------------------------------------

def upper_bound(model, image, target_class: int) -> dict:
    """B1: dropout disabled, same image, two CAMs — should be exactly 1.000.

    This is also the implementation sanity check.  If the result deviates
    from 1.000 by more than 1e-6, a prominent warning is logged — do NOT
    continue collecting results until the cause is identified.

    Args:
        model: MC-Dropout ResNet-18 (any prior state; set to eval here).
        image: Single image tensor of shape ``(3, H, W)``.
        target_class: Class index for attribution.

    Returns:
        ``{"correlation": float, "iou": float}``
    """
    model.eval()
    a = generate_single_cam(model, image, target_class=target_class)
    b = generate_single_cam(model, image, target_class=target_class)

    a_up = upsample_cams(a[np.newaxis], size=224)[0]
    b_up = upsample_cams(b[np.newaxis], size=224)[0]

    corr = _corr_pair(a_up, b_up)
    iou  = _iou_pair(a_up, b_up)

    if abs(corr - 1.0) > 1e-6:
        logger.warning(
            "B1 UPPER-BOUND CORRELATION IS %.6f, NOT 1.000.  "
            "This indicates a bug in the Grad-CAM path or seed handling.  "
            "Do NOT continue collecting results until this is resolved.",
            corr,
        )

    return {"correlation": corr, "iou": iou}


# ---------------------------------------------------------------------------
# B2 — lower bound
# ---------------------------------------------------------------------------

def lower_bound(
    n_pairs: int = 1000,
    size: int = 224,
    sigma: float = 8.0,
    seed: int = 0,
) -> dict:
    """B2: mean pairwise metrics over Gaussian-smoothed random heatmaps.

    Raw white noise has no spatial structure and gives an unrealistically low
    floor.  Gaussian smoothing (sigma ≈ 8 px at 224×224) gives maps whose
    spatial scale roughly matches a real Grad-CAM.

    Args:
        n_pairs: Number of independent pairs to average.
        size: Map side length in pixels.
        sigma: Smoothing kernel sigma in pixels.
        seed: RNG seed for reproducibility.

    Returns:
        ``{"correlation": float, "iou": float}``
    """
    rng = np.random.RandomState(seed)
    corrs: list[float] = []
    ious:  list[float] = []

    for _ in range(n_pairs):
        a = _normalise(gaussian_filter(rng.rand(size, size).astype(np.float32), sigma=sigma))
        b = _normalise(gaussian_filter(rng.rand(size, size).astype(np.float32), sigma=sigma))
        corrs.append(_corr_pair(a, b))
        ious.append(_iou_pair(a, b))

    return {
        "correlation": float(np.mean(corrs)),
        "iou":         float(np.mean(ious)),
    }


# ---------------------------------------------------------------------------
# B3 — cross-image reference
# ---------------------------------------------------------------------------

def cross_image_reference(
    model,
    images: list,
    target_classes: list[int],
    n_pairs: int = 1000,
    seed: int = 0,
) -> dict:
    """B3: mean pairwise metrics between CAMs from different images of the same class.

    Images are grouped by ``target_classes`` (the model's predicted class
    for each image).  Pairs are drawn from within the same predicted class
    so that both CAMs are generated for the same target class, making
    cross-image comparison meaningful.

    B3 is expected to be higher than B2 (random) but lower than B1
    (identical image).  Its IoU value supplies the default ``TH_IOU`` for
    the risk flag.

    Args:
        model: MC-Dropout ResNet-18 in eval mode.
        images: List of ``(3, H, W)`` tensors — one per image.
        target_classes: Predicted class index for each image.
        n_pairs: Number of cross-image pairs to sample.
        seed: RNG seed for reproducibility.

    Returns:
        ``{"correlation": float, "iou": float}``
    """
    model.eval()
    rng = np.random.RandomState(seed)

    # Pre-compute one CAM per image (dropout off) and upsample to 224×224
    cams_up: list[np.ndarray] = []
    for img, tc in zip(images, target_classes):
        raw = generate_single_cam(model, img, target_class=tc)      # (7, 7)
        cams_up.append(upsample_cams(raw[np.newaxis], size=224)[0]) # (224, 224)

    # Build index: predicted class → list of image indices
    class_to_indices: dict[int, list[int]] = {}
    for idx, tc in enumerate(target_classes):
        class_to_indices.setdefault(tc, []).append(idx)

    # Enumerate all valid same-class pairs
    same_class_pairs = [
        (i, j)
        for indices in class_to_indices.values()
        if len(indices) >= 2
        for i, j in combinations(indices, 2)
    ]

    if not same_class_pairs:
        logger.warning(
            "No same-class pairs found for B3 cross-image baseline — "
            "returning zeros.  Provide images from at least two examples "
            "of the same predicted class."
        )
        return {"correlation": 0.0, "iou": 0.0}

    n_sample = min(n_pairs, len(same_class_pairs))
    chosen_idx = rng.choice(len(same_class_pairs), size=n_sample, replace=False)
    chosen = [same_class_pairs[i] for i in chosen_idx]

    corrs: list[float] = []
    ious:  list[float] = []
    for a_idx, b_idx in chosen:
        a, b = cams_up[a_idx], cams_up[b_idx]
        corrs.append(_corr_pair(a, b))
        ious.append(_iou_pair(a, b))

    return {
        "correlation": float(np.mean(corrs)),
        "iou":         float(np.mean(ious)),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def compute_all(model, dataset, cfg: dict, repo, run_id: int) -> dict:
    """Compute B1, B2, B3 and write rows to the ``baselines`` table.

    Collects up to 200 images from the dataset, computes one CAM per image
    (dropout off) and groups them by predicted class for B3.

    Args:
        model: MC-Dropout ResNet-18.
        dataset: PyTorch Dataset yielding ``(image_tensor, label_idx)`` pairs.
        cfg: Full config dict.
        repo: Repository instance (writes to the ``baselines`` table).
        run_id: FK to associate baselines with the current run.

    Returns:
        Nested dict ``{"upper": {...}, "lower": {...}, "cross_image": {...}}``.
    """
    import torch

    sample_size = min(200, len(dataset))
    images:         list = []
    target_classes: list[int] = []

    model.eval()
    with torch.no_grad():
        for i in range(sample_size):
            img, _ = dataset[i]
            logits = model(img.unsqueeze(0))
            tc = int(logits.argmax(dim=1).item())
            images.append(img)
            target_classes.append(tc)

    b1 = upper_bound(model, images[0], target_classes[0])
    b2 = lower_bound(
        n_pairs=1000,
        size=cfg["gradcam"]["upsample_size"],
        seed=cfg["seed"],
    )
    b3 = cross_image_reference(
        model, images, target_classes, n_pairs=1000, seed=cfg["seed"]
    )

    results = {"upper": b1, "lower": b2, "cross_image": b3}

    baseline_rows = [
        {"baseline_type": btype, "metric_name": mname, "value": value}
        for btype, metrics in results.items()
        for mname, value in metrics.items()
    ]
    repo.insert_baselines(run_id, baseline_rows)

    return results
