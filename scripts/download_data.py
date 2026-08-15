"""Download all datasets used by the reliability pipeline.

Imagenette and Imagewoof are fetched from fast.ai S3 as .tgz archives and
extracted under data/.  DTD is downloaded via torchvision.datasets.DTD.

All three checks are idempotent — if the extracted directory already exists
the script skips that dataset rather than re-downloading ~3 GB.

On network failure the script prints the manual download URL and exits
non-zero without leaving a partial archive on disk.

Usage:
    python scripts/download_data.py
"""

import sys
import tarfile
import urllib.request
from pathlib import Path

# Make src/ importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from src.utils import ensure_dirs


# ---------------------------------------------------------------------------
# Source URLs (from docs/SETUP.md §4)
# ---------------------------------------------------------------------------

_IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
_IMAGEWOOF_URL  = "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-320.tgz"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _ProgressHook:
    """urllib reporthook that updates a tqdm progress bar."""

    def __init__(self, desc: str) -> None:
        self._bar: tqdm | None = None
        self._desc = desc
        self._last = 0

    def __call__(self, block_num: int, block_size: int, total_size: int) -> None:
        if self._bar is None:
            self._bar = tqdm(
                total=max(total_size, 0),
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=self._desc,
            )
        downloaded = block_num * block_size
        self._bar.update(downloaded - self._last)
        self._last = downloaded

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


def _count_images(root: Path) -> dict[str, int]:
    """Return {split: image_count} for train/val splits under *root*."""
    _IMG_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}
    counts: dict[str, int] = {}
    for split in ("train", "val"):
        d = root / split
        if d.exists():
            counts[split] = sum(
                1 for p in d.rglob("*") if p.suffix.lower() in _IMG_EXTS
            )
    return counts


def _dir_size_mb(root: Path) -> float:
    """Return total size of all files under *root* in megabytes."""
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) / 1e6


def _download_tgz(
    name: str,
    url: str,
    archive: Path,
    extract_to: Path,
) -> None:
    """Download *url* to a temp file, extract, then rename to *archive*.

    If the network call or extraction fails the temp file is removed and
    the script exits non-zero.  The final rename is atomic on most
    filesystems so a crash during download never leaves a valid-looking
    but corrupt archive at *archive*.
    """
    tmp = archive.with_suffix(".tmp")
    hook = _ProgressHook(f"Downloading {name}")
    try:
        urllib.request.urlretrieve(url, tmp, reporthook=hook)
        hook.close()
    except Exception as exc:
        hook.close()
        if tmp.exists():
            tmp.unlink()
        print(f"\nERROR: could not download {name}: {exc}", file=sys.stderr)
        print(f"Manual download URL:\n  {url}", file=sys.stderr)
        sys.exit(1)

    print(f"  Extracting {name} ...")
    try:
        with tarfile.open(tmp, "r:gz") as tf:
            tf.extractall(path=extract_to)
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        print(f"\nERROR: extraction failed for {name}: {exc}", file=sys.stderr)
        sys.exit(1)

    tmp.rename(archive)   # keep the archive for reference; it is gitignored


# ---------------------------------------------------------------------------
# Per-dataset downloaders
# ---------------------------------------------------------------------------

def download_imagenette(data_dir: Path) -> None:
    """Fetch imagenette2-320 from fast.ai S3 and extract under data_dir."""
    dest = data_dir / "imagenette2-320"
    if dest.exists():
        counts = _count_images(dest)
        size   = _dir_size_mb(dest)
        print(f"Imagenette already present - skipping.  "
              f"Splits: {counts}  Size: {size:.0f} MB")
        return

    print("Imagenette not found - downloading (~330 MB) ...")
    _download_tgz(
        "Imagenette",
        _IMAGENETTE_URL,
        data_dir / "imagenette2-320.tgz",
        data_dir,
    )
    counts = _count_images(dest)
    size   = _dir_size_mb(dest)
    print(f"Imagenette ready.  Splits: {counts}  Size: {size:.0f} MB")


def download_imagewoof(data_dir: Path) -> None:
    """Fetch imagewoof2-320 from fast.ai S3 and extract under data_dir."""
    dest = data_dir / "imagewoof2-320"
    if dest.exists():
        counts = _count_images(dest)
        size   = _dir_size_mb(dest)
        print(f"Imagewoof already present - skipping.  "
              f"Splits: {counts}  Size: {size:.0f} MB")
        return

    print("Imagewoof not found - downloading (~330 MB) ...")
    _download_tgz(
        "Imagewoof",
        _IMAGEWOOF_URL,
        data_dir / "imagewoof2-320.tgz",
        data_dir,
    )
    counts = _count_images(dest)
    size   = _dir_size_mb(dest)
    print(f"Imagewoof ready.  Splits: {counts}  Size: {size:.0f} MB")


def download_dtd(data_dir: Path) -> None:
    """Download DTD test split via torchvision.datasets.DTD."""
    dest = data_dir / "dtd"
    if dest.exists():
        n    = sum(1 for p in dest.rglob("*.jpg"))
        size = _dir_size_mb(dest)
        print(f"DTD already present - skipping.  "
              f"{n} .jpg files  Size: {size:.0f} MB")
        return

    print("DTD not found - downloading via torchvision ...")
    try:
        from torchvision.datasets import DTD as _DTD
        _DTD(root=str(data_dir), split="test", download=True)
    except Exception as exc:
        print(f"\nERROR: could not download DTD: {exc}", file=sys.stderr)
        print(
            "Manual download:\n"
            "  https://www.robots.ox.ac.uk/~vgg/data/dtd/\n"
            "  Extract so that data/dtd/ contains images/, labels/, etc.",
            file=sys.stderr,
        )
        sys.exit(1)

    n    = sum(1 for p in dest.rglob("*.jpg"))
    size = _dir_size_mb(dest)
    print(f"DTD ready.  {n} .jpg files  Size: {size:.0f} MB")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_dirs()
    data_dir = Path("data")

    print("=" * 60)
    print("Dataset acquisition")
    print("=" * 60)

    download_imagenette(data_dir)
    print()
    download_imagewoof(data_dir)
    print()
    download_dtd(data_dir)

    print()
    print("=" * 60)
    print("All datasets ready.")
    print("=" * 60)


if __name__ == "__main__":
    main()
