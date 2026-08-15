"""ResNet-18 with MC-Dropout for stochastic inference and uncertainty quantification."""

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_mc_dropout_resnet18(num_classes: int = 10, p: float = 0.2) -> nn.Module:
    """Build ResNet-18 with Dropout2d inserted after layer2 and after layer3.

    Dropout2d is placed BEFORE the Grad-CAM target layer (layer4) so that
    each stochastic forward pass yields a distinct activation map at layer4.
    Placing dropout after layer4 would leave target activations unchanged and
    make explanation-stability measurement impossible — see CLAUDE.md §1.1.

    Args:
        num_classes: Number of output classes.  Default 10 (Imagenette).
        p: Spatial dropout probability applied to each channel.  Default 0.2.

    Returns:
        ResNet-18 with IMAGENET1K_V1 weights, fc replaced by Linear(512,
        num_classes), and Dropout2d(p) appended after layer2 and layer3.
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, num_classes)

    # Wrap layer2 and layer3 — dropout BEFORE layer4 (Grad-CAM target).
    # Never after layer4, after global average pooling, or before the fc layer.
    model.layer2 = nn.Sequential(model.layer2, nn.Dropout2d(p=p))
    model.layer3 = nn.Sequential(model.layer3, nn.Dropout2d(p=p))

    return model


def enable_mc_dropout(model: nn.Module) -> nn.Module:
    """Switch model to MC-Dropout inference mode.

    Sets the entire model to eval() first — keeping BatchNorm in running-
    statistics mode — then sets only the Dropout2d modules back to train()
    so that each forward pass samples a fresh spatial dropout mask.

    Calling model.train() directly would also flip BatchNorm into training
    mode, replacing running statistics with per-batch statistics and
    corrupting predictions.  This function avoids that — see CLAUDE.md §1.2.

    Args:
        model: A model returned by build_mc_dropout_resnet18.

    Returns:
        The same model object with Dropout2d modules in train() and
        everything else (including BatchNorm2d) in eval().
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout2d):
            m.train()
    return model


def get_target_layer(model: nn.Module) -> nn.Module:
    """Return the Grad-CAM target layer (the output of layer4).

    layer4 is not wrapped in an extra Sequential, so it is addressed
    directly.  pytorch-grad-cam registers a forward/backward hook on
    this module; activations and gradients captured here become the
    Grad-CAM heat map.

    Args:
        model: A model returned by build_mc_dropout_resnet18.

    Returns:
        model.layer4 — the nn.Sequential of BasicBlocks whose output
        is the 7×7 feature map for a 224×224 input.
    """
    return model.layer4
