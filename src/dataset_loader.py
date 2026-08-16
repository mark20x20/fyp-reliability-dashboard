"""Dataset loading with a uniform tensor contract for all sources.

Every loader applies the same 224×224 resize → centre-crop → ImageNet
normalisation pipeline so that downstream code never needs to know which
dataset it is processing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import DTD, ImageFolder

from src.corruption_generator import CorruptImageTransform

# ImageNet normalisation constants — used by the pretrained ResNet-18 backbone
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def _make_transform(input_size: int) -> transforms.Compose:
    """Return the standard evaluation transform for *input_size* × *input_size*."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def _make_corrupt_transform(
    input_size: int,
    corruption_type: str,
    severity: int,
) -> transforms.Compose:
    """Return an evaluation transform with an on-the-fly corruption step.

    Corruption is inserted **between CenterCrop and ToTensor** so that it
    operates on raw uint8 pixel values before ImageNet normalisation.

    Args:
        input_size: Square crop size in pixels (224 for ResNet-18).
        corruption_type: One of the types in
            ``src.corruption_generator.SUPPORTED_CORRUPTIONS``.
        severity: Integer 1–5 passed to imagecorruptions.

    Returns:
        A ``transforms.Compose`` pipeline.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        CorruptImageTransform(corruption_type, severity),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


class _CorruptedSubset(Dataset):
    """A fixed, seeded subset of an ImageFolder dataset with a custom transform.

    The same *n* images (selected once using *seed*) are reused across all
    corruption cells so that severity/type comparisons are fair — every cell
    is evaluated on an identical set of source images.

    Exposes a ``.imgs`` attribute containing ``(path, label_idx)`` pairs in the
    same format as ``ImageFolder.imgs`` so that ``batch_evaluation.run_batch``
    can resolve image paths without modification.

    Args:
        base_dataset: An ``ImageFolder`` (untransformed or with a placeholder
            transform) whose ``.imgs`` and ``.classes`` are used for index
            selection and class name lookup.
        indices: 1-D integer array of selected indices into *base_dataset*.
        transform: The transform to apply when ``__getitem__`` is called.
    """

    def __init__(
        self,
        base_dataset: ImageFolder,
        indices: np.ndarray,
        transform: transforms.Compose,
    ) -> None:
        self._base    = base_dataset
        self._indices = indices
        self.transform = transform

        # Expose the subset's (path, label_idx) pairs so run_batch can read
        # the source image path via dataset.imgs[idx][0]
        self.imgs    = [base_dataset.imgs[i] for i in indices]
        self.classes = base_dataset.classes

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        base_idx = int(self._indices[idx])
        # Load PIL image via the base dataset's loader (bypasses its transform)
        path, label_idx = self._base.imgs[base_idx]
        img = self._base.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label_idx


def get_dataset(
    name: str,
    cfg: dict,
    *,
    corruption_type: str | None = None,
    severity: int | None = None,
    n: int = 250,
    seed: int = 42,
) -> tuple[Dataset, list[str]]:
    """Load a dataset by name and return ``(dataset, class_names)``.

    All datasets pass through the same transform so callers receive a uniform
    ``(3, input_size, input_size)`` float32 tensor regardless of the source.
    Class names are the folder labels for ImageFolder-based datasets and the
    built-in class list for DTD.

    For ``"imagenette-c"`` the extra keyword arguments control the corruption
    and subset selection.  *corruption_type* and *severity* are required.
    The same *n* images (chosen with *seed*) are used across all cells.

    Args:
        name: One of ``"imagenette"``, ``"imagewoof"``, ``"dtd"``,
              ``"imagenette-c"``.
        cfg: Full configuration dict from ``load_config``.  Reads
             ``data.input_size`` and ``data.data_root``.
        corruption_type: Required for ``"imagenette-c"``.  One of the types in
            ``src.corruption_generator.SUPPORTED_CORRUPTIONS``.
        severity: Required for ``"imagenette-c"``.  Integer 1–5.
        n: Number of images to sample from imagenette val for each corruption
           cell.  Default 250 (Phase 7 spec).
        seed: RNG seed for reproducible subset selection.  Default 42.

    Returns:
        ``(dataset, class_names)`` where *class_names* is a list of strings
        in the same order as the integer class labels in the dataset.

    Raises:
        FileNotFoundError: If the data directory for the requested dataset
            does not exist.  Run ``scripts/download_data.py`` to fetch it.
        ValueError: For an unknown dataset name, or if *corruption_type* /
            *severity* are missing when ``name == "imagenette-c"``.
    """
    input_size: int = cfg["data"]["input_size"]
    data_root   = Path(cfg["data"]["data_root"])
    transform   = _make_transform(input_size)

    if name == "imagenette":
        val_root = data_root / "imagenette2-320" / "val"
        if not val_root.exists():
            raise FileNotFoundError(
                f"Imagenette val split not found at {val_root}. "
                "Run scripts/download_data.py to fetch the dataset."
            )
        dataset = ImageFolder(root=str(val_root), transform=transform)
        return dataset, dataset.classes

    if name == "imagewoof":
        val_root = data_root / "imagewoof2-320" / "val"
        if not val_root.exists():
            raise FileNotFoundError(
                f"Imagewoof val split not found at {val_root}. "
                "Run scripts/download_data.py to fetch the dataset."
            )
        dataset = ImageFolder(root=str(val_root), transform=transform)
        return dataset, dataset.classes

    if name == "dtd":
        dtd_root = data_root / "dtd"
        if not dtd_root.exists():
            raise FileNotFoundError(
                f"DTD not found at {dtd_root}. "
                "Run scripts/download_data.py to fetch the dataset."
            )
        dataset = DTD(root=str(data_root), split="test", transform=transform)
        return dataset, dataset.classes

    if name == "imagenette-c":
        if corruption_type is None or severity is None:
            raise ValueError(
                "get_dataset('imagenette-c', ...) requires both "
                "corruption_type and severity keyword arguments."
            )
        val_root = data_root / "imagenette2-320" / "val"
        if not val_root.exists():
            raise FileNotFoundError(
                f"Imagenette val split not found at {val_root}. "
                "Run scripts/download_data.py to fetch the dataset."
            )
        # Load base dataset without any transform (PIL loader only)
        base = ImageFolder(root=str(val_root), transform=None)

        # Fixed, seeded subset — same n images across all corruption cells
        rng     = np.random.default_rng(seed)
        indices = rng.choice(len(base), size=min(n, len(base)), replace=False)
        indices.sort()  # preserve original class distribution order

        corrupt_transform = _make_corrupt_transform(
            input_size, corruption_type, severity
        )
        dataset = _CorruptedSubset(base, indices, corrupt_transform)
        return dataset, base.classes

    raise ValueError(
        f"Unknown dataset {name!r}.  "
        "Supported: imagenette, imagewoof, dtd, imagenette-c."
    )
