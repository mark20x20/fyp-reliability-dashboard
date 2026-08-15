"""Shared utilities: config loading, seeding, directory setup, device selection."""

import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Runtime output directories (gitignored — must exist at runtime)
# ---------------------------------------------------------------------------

_OUTPUT_DIRS = [
    "outputs/runs",
    "outputs/figures",
    "outputs/ablation",
    "outputs/reviews",
    "outputs/reports",
    "outputs/gradcam/raw",
    "outputs/gradcam/png",
    "data",
    "models/checkpoints",
    "db",
]


def load_config(path: str = "configs/config.yaml") -> dict:
    """Load configuration from a YAML file, with optional .env overrides.

    The .env file is loaded into os.environ first.  Two environment variables
    are then applied on top of the parsed YAML dict:

    - ``DATABASE_PATH`` → ``cfg["database"]["path"]``
    - ``DATA_ROOT``     → ``cfg["data"]["data_root"]``

    This allows CI or a separate run environment to redirect the database and
    dataset roots without editing the committed YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Plain dict with the merged configuration.  All callers use this
        dict; no module reads the YAML file again on its own.
    """
    load_dotenv(override=False)   # populate os.environ from .env if present

    with open(path) as f:
        cfg: dict = yaml.safe_load(f)

    # Environment variable overrides
    if os.environ.get("DATABASE_PATH"):
        cfg.setdefault("database", {})["path"] = os.environ["DATABASE_PATH"]
    if os.environ.get("DATA_ROOT"):
        cfg.setdefault("data", {})["data_root"] = os.environ["DATA_ROOT"]

    return cfg


def set_seed(seed: int) -> None:
    """Seed every RNG that affects model or metric results.

    Sets torch (CPU and CUDA), numpy, and the Python stdlib random module.
    Also enables cuDNN deterministic mode and disables the benchmark heuristic
    so that convolution algorithm selection is reproducible across runs.

    Args:
        seed: Integer seed.  Written into the ``runs`` table by the pipeline
              so every reported number is traceable to its exact configuration.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dirs(root: str | Path = ".") -> None:
    """Create all runtime output directories under *root*.  Idempotent.

    These directories are gitignored and must exist at runtime.  Safe to call
    multiple times; existing directories are silently skipped.

    Args:
        root: Project root directory.  Defaults to the current working directory.
    """
    for d in _OUTPUT_DIRS:
        (Path(root) / d).mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    """Return the best available torch device.

    Returns:
        ``torch.device("cuda")`` if a CUDA-capable GPU is available,
        otherwise ``torch.device("cpu")``.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
