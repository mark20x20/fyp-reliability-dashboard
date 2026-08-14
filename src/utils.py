from pathlib import Path

OUTPUT_DIRS = [
    "outputs/runs", "outputs/figures", "outputs/ablation",
    "outputs/reviews", "outputs/reports",
    "outputs/gradcam/raw", "outputs/gradcam/png",
    "data", "models/checkpoints", "db",
]

def ensure_dirs(root: str | Path = ".") -> None:
    """Create all runtime output directories. Idempotent."""
    for d in OUTPUT_DIRS:
        (Path(root) / d).mkdir(parents=True, exist_ok=True)