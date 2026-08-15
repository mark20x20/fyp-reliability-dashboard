"""Grad-CAM map generation for MC-Dropout stochastic passes."""

import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from models.resnet_dropout import get_target_layer


def generate_repeated_cams(
    model: torch.nn.Module,
    image: torch.Tensor,
    target_class: int,
    n_runs: int,
) -> np.ndarray:
    """Generate N Grad-CAM maps using a single fixed target class.

    The target class is supplied by the caller (argmax of the mean predictive
    distribution across all MC-Dropout passes) and is held constant for every
    one of the N maps.  Using the per-pass argmax would confound prediction
    instability with explanation instability — see CLAUDE.md §1.3.

    The caller is responsible for setting the model's dropout mode before
    calling this function (enable_mc_dropout for stochastic passes, or
    model.eval() for the deterministic baseline).

    Note: pytorch-grad-cam's BaseCAM.__init__ calls model.eval() internally,
    which would override enable_mc_dropout().  This function saves the
    Dropout2d training states before constructing the GradCAM object and
    restores them inside the context so the caller's intent is preserved.

    Args:
        model: Model whose Dropout2d modules are already configured by the
               caller (MC-Dropout or deterministic).
        image: Input tensor of shape (3, H, W).  Must be on the same device
               as the model.
        target_class: Class index for which attributions are computed.
                      Held fixed across all n_runs passes.  Range: [0, C).
        n_runs: Number of stochastic forward passes.  Each yields one map.

    Returns:
        Float32 NumPy array of shape (n_runs, 7, 7) for a 224×224 input
        through ResNet-18.  Values in [0, 1] after pytorch-grad-cam's
        per-map min-max normalisation.
    """
    target_layers = [get_target_layer(model)]
    # One ClassifierOutputTarget created once — the same class is passed to
    # every call, so the target never drifts between passes.
    targets = [ClassifierOutputTarget(target_class)]

    # Add batch dimension: (3, H, W) → (1, 3, H, W)
    input_tensor = image.unsqueeze(0)

    # pytorch-grad-cam calls model.eval() inside BaseCAM.__init__, which
    # overrides any enable_mc_dropout() the caller set.  Snapshot the
    # training flag for every Dropout2d module before entering the context
    # and restore it immediately after __init__ runs.
    dropout_states: dict[nn.Dropout2d, bool] = {
        m: m.training
        for m in model.modules()
        if isinstance(m, nn.Dropout2d)
    }

    cams: list[np.ndarray] = []
    with GradCAM(model=model, target_layers=target_layers) as cam:
        # Restore the Dropout2d training states the caller intended.
        for module, was_training in dropout_states.items():
            module.training = was_training

        for _ in range(n_runs):
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            # grayscale_cam: (1, H, W) — copy to own memory before the next pass
            cams.append(grayscale_cam[0].copy())

    return np.stack(cams, axis=0)  # (n_runs, H, W)


def generate_single_cam(
    model: torch.nn.Module,
    image: torch.Tensor,
    target_class: int,
) -> np.ndarray:
    """Generate one Grad-CAM map.

    Convenience wrapper around generate_repeated_cams for n_runs=1.
    Used for the B1 (deterministic upper-bound) baseline and TC17.

    Args:
        model: Model in eval mode (dropout off) for deterministic use, or
               MC-Dropout mode for a single stochastic sample.
        image: Input tensor of shape (3, H, W).
        target_class: Class index for attribution.  Not recomputed
                      internally — callers must fix it before calling.

    Returns:
        Float32 NumPy array of shape (7, 7) for a 224×224 input.
        Values in [0, 1].
    """
    return generate_repeated_cams(model, image, target_class, n_runs=1)[0]
