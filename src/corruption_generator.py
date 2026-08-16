"""On-the-fly image corruption transform using the imagecorruptions library.

``CorruptImageTransform`` is a torchvision-compatible PIL→PIL transform that
applies one of the standard ImageNet-C corruption types at a given severity.

It must be inserted **between CenterCrop and ToTensor** in the transform
pipeline so that corruption is applied to the raw uint8 pixel array before
ImageNet normalisation.

Supported corruption types (subset of imagecorruptions.get_corruption_names()):
    gaussian_noise, defocus_blur, brightness, contrast

Reference:
    Hendrycks & Dietterich, "Benchmarking Neural Network Robustness to Common
    Corruptions and Perturbations", ICLR 2019.
    https://github.com/hendrycks/robustness
"""

from __future__ import annotations

import numpy as np
from PIL import Image


# Subset of imagecorruptions types used in Phase 7 E8
SUPPORTED_CORRUPTIONS: tuple[str, ...] = (
    "gaussian_noise",
    "defocus_blur",
    "brightness",
    "contrast",
)


class CorruptImageTransform:
    """Torchvision-compatible PIL -> PIL corruption transform.

    Applies a single imagecorruptions corruption type at a fixed severity level.
    The transform converts the input PIL image to a uint8 HWC numpy array (as
    required by ``imagecorruptions.corrupt``), applies the corruption, then
    converts the result back to a PIL Image so downstream transforms
    (``ToTensor``, ``Normalize``) remain unchanged.

    Args:
        corruption_type: Name of the corruption.  Must be one of
            ``SUPPORTED_CORRUPTIONS``.
        severity: Integer from 1 (weakest) to 5 (strongest).

    Raises:
        ValueError: If *corruption_type* is not in ``SUPPORTED_CORRUPTIONS`` or
            *severity* is outside [1, 5].

    Example::

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            CorruptImageTransform("gaussian_noise", severity=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=..., std=...),
        ])
    """

    def __init__(self, corruption_type: str, severity: int) -> None:
        if corruption_type not in SUPPORTED_CORRUPTIONS:
            raise ValueError(
                f"Unknown corruption {corruption_type!r}. "
                f"Supported: {SUPPORTED_CORRUPTIONS}"
            )
        if not (1 <= severity <= 5):
            raise ValueError(
                f"severity must be in [1, 5], got {severity}."
            )
        self.corruption_type = corruption_type
        self.severity = severity

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply corruption to a PIL Image and return a PIL Image.

        Args:
            img: Input PIL Image (any mode accepted; converted to RGB internally
                 if not already RGB).

        Returns:
            Corrupted PIL Image in RGB mode.
        """
        # Ensure RGB so imagecorruptions receives a 3-channel uint8 array
        if img.mode != "RGB":
            img = img.convert("RGB")

        # PIL -> uint8 HWC numpy (imagecorruptions requirement)
        img_np = np.array(img, dtype=np.uint8)

        # Apply corruption — import deferred so the module loads even if
        # imagecorruptions is not installed (raises ImportError at call time)
        from imagecorruptions import corrupt  # type: ignore[import]

        corrupted_np = corrupt(
            img_np,
            corruption_name=self.corruption_type,
            severity=self.severity,
        )

        # uint8 HWC numpy -> PIL Image
        return Image.fromarray(corrupted_np.astype(np.uint8))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"corruption_type={self.corruption_type!r}, "
            f"severity={self.severity})"
        )
