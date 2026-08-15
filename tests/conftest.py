"""Shared pytest fixtures for the test suite."""

import sqlite3
from pathlib import Path

import numpy as np
import pytest
import torch

from models.resnet_dropout import build_mc_dropout_resnet18


@pytest.fixture(scope="session")
def model() -> torch.nn.Module:
    """ResNet-18 with MC-Dropout, built once and shared across all tests.

    torch.manual_seed(0) before construction so the freshly-initialised fc
    layer has deterministic weights in every pytest session.  Without a fixed
    seed the fc weights vary between sessions, which changes the predicted
    class and — for unlucky initialisations — can produce all-zero Grad-CAM
    maps for class 0, making TC16 flaky.

    Session scope avoids reloading the pretrained backbone weights (≈45 MB)
    for every test module.
    """
    torch.manual_seed(0)
    return build_mc_dropout_resnet18(num_classes=10, p=0.2)


@pytest.fixture
def sample_image() -> torch.Tensor:
    """Synthetic 224×224 image with real spatial structure, ImageNet-normalised.

    A smooth per-channel gradient background plus a bright Gaussian blob
    replaces ``torch.randn`` noise.  Pure noise produces near-zero activations
    in the pretrained ResNet-18 backbone (which was trained on natural images),
    so the weighted-activation sum in Grad-CAM can be uniformly negative,
    collapsing to an all-zero map after ReLU.  np.corrcoef then divides by zero
    and returns nan, poisoning the mean and making TC16 fail non-deterministically.

    A structured image stimulates convolutional feature detectors and produces
    richer, higher-magnitude activations at layer4, greatly reducing the
    probability of all-zero maps.  The construction is fully deterministic —
    no torch seeds are touched; the result is always the same tensor.
    """
    # Coordinate grids, values in [0, 1]
    x = np.linspace(0, 1, 224, dtype=np.float32)
    y = np.linspace(0, 1, 224, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)  # each (224, 224)

    # Per-channel gradient backgrounds — different orientation per channel
    r = 0.30 + 0.40 * xx
    g = 0.30 + 0.40 * yy
    b = 0.30 + 0.20 * (xx + yy)

    # Bright Gaussian blob at (0.25, 0.25) — activates localised detectors
    cx, cy = 0.25, 0.25
    sigma = 0.12
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    blob = np.exp(-dist2 / (2.0 * sigma ** 2)).astype(np.float32)

    r = np.clip(r + 0.50 * blob, 0.0, 1.0)
    g = np.clip(g + 0.40 * blob, 0.0, 1.0)
    b = np.clip(b + 0.30 * blob, 0.0, 1.0)

    img = np.stack([r, g, b], axis=0)  # (3, 224, 224), values in [0, 1]

    # ImageNet normalisation expected by the pretrained backbone
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    img  = (img - mean) / std

    return torch.from_numpy(img)


@pytest.fixture
def temp_db(tmp_path) -> str:
    """Temporary SQLite database initialised from db/schema.sql.

    Each test that requests this fixture receives its own isolated database
    under pytest's tmp_path.  Never points at db/reliability.db.

    Returns:
        Absolute path to the test database as a string.
    """
    path = tmp_path / "test.db"
    conn = sqlite3.connect(path)
    conn.executescript(Path("db/schema.sql").read_text())
    conn.close()
    return str(path)


@pytest.fixture
def identical_cams() -> np.ndarray:
    """Ten identical 7×7 Grad-CAM maps for testing explanation metrics.

    All ten maps share the same random array, so pairwise correlation = 1.0
    and IoU = 1.0.  Useful for the upper-bound cases in TC7, TC8, TC11.

    Returns:
        Float64 array of shape (10, 7, 7) with values in [0, 1).
    """
    base = np.random.RandomState(0).rand(7, 7)
    return np.stack([base] * 10)


@pytest.fixture
def constant_cams() -> np.ndarray:
    """Five identical constant (all-0.5) 7×7 maps for degeneracy testing.

    Every map has zero spatial variance, so np.corrcoef returns nan for every
    pair.  Used by TC10 to verify that metric functions handle the degenerate
    case without raising ZeroDivisionError or propagating nan.

    Returns:
        Float32 array of shape (5, 7, 7), all values = 0.5.
    """
    return np.full((5, 7, 7), 0.5, dtype=np.float32)
