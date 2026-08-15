"""Fine-tune ResNet-18 with MC-Dropout on Imagenette.

Trains for 15 epochs with AdamW + cosine LR schedule, saves the best-val-
accuracy checkpoint, registers the checkpoint in the database, and returns
a summary dict.  Target: val accuracy >= 95%.

Usage:
    python src/train_model.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# Make project root importable when invoked as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.resnet_dropout import build_mc_dropout_resnet18
from src.repository import Repository
from src.utils import ensure_dirs, get_device, load_config, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_N_EPOCHS = 15


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def _train_transform(input_size: int) -> transforms.Compose:
    """Standard ImageNet augmentation for training."""
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def _val_transform(input_size: int) -> transforms.Compose:
    """Deterministic evaluation transform matching dataset_loader._make_transform."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: dict) -> dict:
    """Fine-tune ResNet-18 on Imagenette and register the checkpoint.

    Reads all hyper-parameters from *cfg* — no magic numbers in the body.
    Saves the best-val-accuracy checkpoint with a filename that encodes
    the architecture, dropout probability, and achieved accuracy.

    Args:
        cfg: Full configuration dict from ``load_config``.  Reads
             ``seed``, ``model.*``, ``data.*``, ``paths.*``, and
             ``database.*``.

    Returns:
        Dict with keys:
        - ``model_id``          (int) — PK in the ``models`` table.
        - ``checkpoint_path``   (str) — absolute path to the .pth file.
        - ``best_val_accuracy`` (float) — best validation accuracy.
        - ``history``           (list[dict]) — per-epoch metrics.
    """
    set_seed(cfg["seed"])
    device = get_device()
    ensure_dirs()

    logger.info("Device: %s", device)

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    input_size = cfg["data"]["input_size"]
    data_root  = Path(cfg["data"]["data_root"])
    batch_size = cfg["data"]["batch_size"]

    train_root = data_root / "imagenette2-320" / "train"
    val_root   = data_root / "imagenette2-320" / "val"

    if not train_root.exists():
        raise FileNotFoundError(
            f"Imagenette train split not found at {train_root}. "
            "Run scripts/download_data.py first."
        )

    train_ds = ImageFolder(root=str(train_root), transform=_train_transform(input_size))
    val_ds   = ImageFolder(root=str(val_root),   transform=_val_transform(input_size))

    # num_workers=4 is safe when called from the __main__ guard (Windows
    # uses spawn; the guard prevents child processes from re-running main).
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )

    logger.info(
        "Imagenette — train: %d images, val: %d images",
        len(train_ds), len(val_ds),
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_mc_dropout_resnet18(
        num_classes=cfg["model"]["num_classes"],
        p=cfg["model"]["dropout_p"],
    ).to(device)

    # ------------------------------------------------------------------
    # Optimiser, scheduler, loss
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=_N_EPOCHS, eta_min=1e-5
    )
    criterion = nn.CrossEntropyLoss()

    # Mixed precision — only on CUDA; does not affect the deterministic
    # eval path because it is used inside no_grad() blocks with model.eval().
    _cuda = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=_cuda)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    history: list[dict] = []
    best_val_acc   = 0.0
    best_ckpt_path = ""

    for epoch in range(1, _N_EPOCHS + 1):

        # ---- train ----
        model.train()
        train_loss_sum = 0.0

        for images, labels in tqdm(
            train_loader,
            desc=f"Epoch {epoch:2d}/{_N_EPOCHS} [train]",
            leave=False,
            unit="batch",
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=_cuda):
                logits = model(images)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * len(images)

        train_loss = train_loss_sum / len(train_ds)
        scheduler.step()

        # ---- val (no dropout, no gradient) ----
        model.eval()
        val_loss_sum = 0.0
        n_correct    = 0

        with torch.no_grad():
            for images, labels in tqdm(
                val_loader,
                desc=f"Epoch {epoch:2d}/{_N_EPOCHS} [val]  ",
                leave=False,
                unit="batch",
            ):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                val_loss_sum += criterion(logits, labels).item() * len(images)
                n_correct    += (logits.argmax(dim=1) == labels).sum().item()

        val_loss = val_loss_sum / len(val_ds)
        val_acc  = n_correct / len(val_ds)

        logger.info(
            "Epoch %2d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f  lr=%.2e",
            epoch, _N_EPOCHS, train_loss, val_loss, val_acc,
            scheduler.get_last_lr()[0],
        )

        history.append({
            "epoch":        epoch,
            "train_loss":   train_loss,
            "val_loss":     val_loss,
            "val_accuracy": val_acc,
        })

        # ---- checkpoint ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            arch    = cfg["model"]["arch"]
            dp      = cfg["model"]["dropout_p"]
            ckpt_dir  = Path(cfg["paths"]["checkpoints"])
            ckpt_path = ckpt_dir / f"{arch}_dp{dp}_acc{val_acc:.4f}.pth"
            torch.save(model.state_dict(), ckpt_path)
            best_ckpt_path = str(ckpt_path.resolve())
            logger.info("  Checkpoint saved: %s", ckpt_path)

    # ------------------------------------------------------------------
    # Post-training summary
    # ------------------------------------------------------------------
    logger.info("Training complete.  Best val accuracy: %.4f", best_val_acc)

    if best_val_acc < 0.95:
        logger.warning(
            "Best val accuracy %.4f is below the 95%% target.  "
            "Per-epoch history follows; report this figure rather than "
            "adjusting the target.",
            best_val_acc,
        )
        for row in history:
            logger.warning(
                "  epoch %2d  val_acc=%.4f", row["epoch"], row["val_accuracy"]
            )

    # ------------------------------------------------------------------
    # Register checkpoint in the database
    # ------------------------------------------------------------------
    repo = Repository(cfg["database"]["path"])
    dropout_layers_str = ",".join(
        layer if isinstance(layer, str) else str(layer)
        for layer in cfg["model"]["dropout_layers"]
    )
    model_id = repo.create_model(
        name=f"{cfg['model']['arch']}_imagenette_dp{cfg['model']['dropout_p']}",
        arch=cfg["model"]["arch"],
        dropout_p=cfg["model"]["dropout_p"],
        dropout_layers=dropout_layers_str,
        checkpoint_path=best_ckpt_path,
        val_accuracy=best_val_acc,
    )
    logger.info("Registered in database: model_id=%d", model_id)

    return {
        "model_id":          model_id,
        "checkpoint_path":   best_ckpt_path,
        "best_val_accuracy": best_val_acc,
        "history":           history,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # The __main__ guard is required on Windows: DataLoader spawns child
    # processes and they must not re-execute this block.
    parser = argparse.ArgumentParser(
        description="Fine-tune ResNet-18 with MC-Dropout on Imagenette"
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to config YAML (default: configs/config.yaml)",
    )
    args = parser.parse_args()

    cfg    = load_config(args.config)
    result = train(cfg)

    print()
    print(f"model_id         : {result['model_id']}")
    print(f"checkpoint_path  : {result['checkpoint_path']}")
    print(f"best_val_accuracy: {result['best_val_accuracy']:.4f}")
