"""Dataset loading with a uniform tensor contract for all sources.

Every loader applies the same 224×224 resize → centre-crop → ImageNet
normalisation pipeline so that downstream code never needs to know which
dataset it is processing.
"""

from pathlib import Path

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import DTD, ImageFolder

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


def get_dataset(name: str, cfg: dict) -> tuple[Dataset, list[str]]:
    """Load a dataset by name and return ``(dataset, class_names)``.

    All datasets pass through the same transform so callers receive a uniform
    ``(3, input_size, input_size)`` float32 tensor regardless of the source.
    Class names are the folder labels for ImageFolder-based datasets and the
    built-in class list for DTD.

    Args:
        name: One of ``"imagenette"``, ``"imagewoof"``, ``"dtd"``.
              ``"imagenette-c"`` raises ``NotImplementedError`` (Phase 7).
        cfg: Full configuration dict from ``load_config``.  Reads
             ``data.input_size`` and ``data.data_root``.

    Returns:
        ``(dataset, class_names)`` where *class_names* is a list of strings
        in the same order as the integer class labels in the dataset.

    Raises:
        FileNotFoundError: If the data directory for the requested dataset
            does not exist.  Run ``scripts/download_data.py`` to fetch it.
        NotImplementedError: For ``"imagenette-c"`` (reserved for Phase 7).
        ValueError: For any other unknown dataset name.
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
        raise NotImplementedError(
            "imagenette-c (corrupted Imagenette) is implemented in Phase 7.  "
            "Use src/corruption_generator.py when it is available."
        )

    raise ValueError(
        f"Unknown dataset {name!r}.  Supported: imagenette, imagewoof, dtd."
    )
