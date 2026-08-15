"""Batched MC-Dropout stochastic forward pass.

Replicates the image N times into one batch so that dropout masks are
sampled independently per sample in a single forward call — roughly 10×
faster than looping N individual passes.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from models.resnet_dropout import enable_mc_dropout


def mc_dropout_forward(
    model: torch.nn.Module,
    image: Tensor,
    n_runs: int,
) -> Tensor:
    """Run N stochastic forward passes and return softmax probabilities.

    Sets the model to MC-Dropout mode internally (eval + Dropout2d in
    train), replicates the image into a batch of n_runs, runs one
    forward call, and returns per-pass softmax probabilities.

    Each sample in the batch receives an independent Dropout2d mask
    because Dropout2d samples channel masks independently per element in
    the batch dimension.  One batched call replaces N sequential passes
    and is an order of magnitude faster at n_runs = 30.

    Args:
        model: MC-Dropout ResNet-18 built by
               ``models.resnet_dropout.build_mc_dropout_resnet18``.
        image: Single image tensor of shape ``(3, H, W)``.  Must already
               be on the same device as the model.
        n_runs: Number of stochastic passes.  Default in config is 30.

    Returns:
        Float32 tensor of shape ``(n_runs, num_classes)`` — one row of
        softmax probabilities per stochastic pass.
    """
    enable_mc_dropout(model)

    # (3, H, W) → (n_runs, 3, H, W): contiguous so CUDA kernels can run
    # on a single contiguous allocation; each copy gets its own mask.
    batch = image.unsqueeze(0).expand(n_runs, -1, -1, -1).contiguous()

    with torch.no_grad():
        logits = model(batch)   # (n_runs, num_classes)

    return F.softmax(logits, dim=1)
