"""
Verification gate — TC16, TC17, TC18.

TC16 and TC17 must pass before any other development proceeds.

  TC16 failure → Dropout2d is misplaced (sits after layer4, the Grad-CAM
                 target); activations at layer4 are never perturbed; every
                 explanation-stability number in the project is meaningless.

  TC17 failure → Bug in the Grad-CAM path or in seed / determinism handling;
                 the B1 baseline cannot be trusted.

Re-run after any change to the model, inference, or Grad-CAM code.
"""

import numpy as np
import pytest
import torch

from models.resnet_dropout import enable_mc_dropout
from src.gradcam_generator import generate_repeated_cams, generate_single_cam
from src.metrics_explanation import mean_pairwise_correlation


def _predicted_class(model: torch.nn.Module, image: torch.Tensor) -> int:
    """Return the model's argmax class for *image* under full eval mode.

    This is the target class used by TC16 and TC17.  Using the model's own
    predicted class (rather than a hard-coded 0) mirrors the production
    design (CLAUDE.md §1.3) and avoids all-zero Grad-CAM maps: the predicted
    class has the highest logit, meaning its gradient signal produces the
    strongest positive contribution to the heat map.
    """
    model.eval()
    with torch.no_grad():
        logits = model(image.unsqueeze(0))
    return int(logits.argmax(dim=1).item())


# ---------------------------------------------------------------------------
# TC16 — maps must vary under active dropout
# ---------------------------------------------------------------------------

def test_tc16_cams_vary_under_dropout(model, sample_image):
    """TC16: Mean pairwise CAM correlation must be < 0.99 with dropout on.

    Setup  : Seeded (torch.manual_seed(42)), Dropout2d modules in train()
             mode, one structured image, 10 forward passes.
             Target class = model's own argmax (see §1.3 and helper above).
    Req    : (a) zero constant maps — correlation undefined for constant maps;
             (b) mean pairwise Pearson r < 0.99.
    Failure (a): fixture image or target class is degenerate — fix the input,
             do not lower the threshold or add nan_to_num in the assertion.
    Failure (b): Dropout2d sits AFTER layer4 (the Grad-CAM target); activations
             are identical across passes.  Do NOT loosen the threshold — fix
             the dropout placement.
    """
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    target_class = _predicted_class(model, sample_image)
    enable_mc_dropout(model)
    cams = generate_repeated_cams(model, sample_image, target_class=target_class, n_runs=10)

    # --- (a) Non-degeneracy guard -------------------------------------------
    # This assertion must come BEFORE computing correlation.
    # mean_pairwise_correlation substitutes nan→0.0 (METRICS.md spec).  If
    # every CAM were blank, that substitution yields mean_r = 0.0, which
    # trivially satisfies r < 0.99 and hides the real defect.
    stds = cams.reshape(len(cams), -1).std(axis=1)
    n_constant = int((stds < 1e-8).sum())
    print(f"\n[TC16] target_class : {target_class}")
    print(f"[TC16] per-CAM std  : {np.round(stds, 3)}")
    print(f"[TC16] constant maps: {n_constant}/{len(cams)}")
    assert n_constant == 0, (
        f"{n_constant}/{len(cams)} Grad-CAM maps are constant (all-zero after "
        f"ReLU).  Correlation is undefined.  This is a degenerate test input, "
        f"not evidence about dropout.  Fix the fixture image or the target class."
    )

    # --- (b) Variation check ------------------------------------------------
    r, _ = mean_pairwise_correlation(cams)
    print(f"[TC16] mean pairwise r: {r:.4f}")
    assert r < 0.99, (
        f"CAMs are not varying under dropout (mean pairwise r={r:.4f}). "
        "Dropout2d must be after layer2 and layer3, BEFORE layer4. "
        "Do not loosen this threshold — fix the dropout placement."
    )


# ---------------------------------------------------------------------------
# TC17 — maps must be identical without dropout (B1 baseline)
# ---------------------------------------------------------------------------

def test_tc17_cams_deterministic_without_dropout(model, sample_image):
    """TC17: Two CAMs with dropout off must correlate at exactly 1.000.

    Setup  : model.eval() (all modules in eval mode, dropout inactive),
             same image, same target class as TC16, cuDNN deterministic mode.
    Req    : non-constant map (so correlation is defined) and Pearson r = 1.000
             (tolerance 1e-6).
    Failure: Non-deterministic op in the Grad-CAM path, or dropout still active
             (enable_mc_dropout was not reversed by model.eval()).
             Also serves as the B1 upper-bound baseline — a value below 1.000
             means B1 cannot be reported.
    """
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model.eval()  # puts ALL modules — including Dropout2d — into eval mode
    target_class = _predicted_class(model, sample_image)

    a = generate_single_cam(model, sample_image, target_class=target_class)
    b = generate_single_cam(model, sample_image, target_class=target_class)

    # Non-degeneracy guard before computing correlation (avoids nan)
    a_std = float(a.ravel().std())
    print(f"\n[TC17] target_class: {target_class}")
    print(f"[TC17] map std     : {a_std:.4f}")
    assert a_std > 1e-8, (
        f"Deterministic Grad-CAM map is constant (std={a_std:.2e}).  "
        f"Correlation is undefined for target_class={target_class}.  "
        f"Fix the model fixture or the target class."
    )

    r = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    print(f"[TC17] deterministic r: {r:.8f}")
    assert r == pytest.approx(1.0, abs=1e-6), (
        f"Deterministic CAMs differ (r={r:.8f}). "
        "Check for non-deterministic ops or that model.eval() is not "
        "being overridden by a residual enable_mc_dropout call."
    )


# ---------------------------------------------------------------------------
# TC18 — all N maps generated against the same fixed target class
# ---------------------------------------------------------------------------

def test_tc18_cams_use_fixed_target_class(model, sample_image):
    """TC18: generate_repeated_cams uses the caller-supplied target class
    for every one of the N passes.

    Setup  : MC-Dropout mode, target_class=3, n_runs=5.
    Req    : Exactly n_runs maps returned; all attributed to target_class=3.
             Per-pass recomputation of the target class would confound
             prediction instability with explanation instability, which is
             what the study design is built to separate — see CLAUDE.md §1.3.

    Verification strategy: monkeypatch ClassifierOutputTarget.__init__ to
    record every class argument passed during the call.  All recorded values
    must equal the fixed target_class.
    """
    from unittest.mock import patch

    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    enable_mc_dropout(model)
    target_class = 3
    n_runs = 5
    recorded: list[int] = []

    original_init = ClassifierOutputTarget.__init__

    def recording_init(self, category: int) -> None:  # type: ignore[override]
        recorded.append(category)
        original_init(self, category)

    with patch.object(ClassifierOutputTarget, "__init__", recording_init):
        cams = generate_repeated_cams(
            model, sample_image, target_class=target_class, n_runs=n_runs
        )

    # Shape contract: (n_runs, H, W)
    assert cams.shape[0] == n_runs, (
        f"Expected {n_runs} maps, got {cams.shape[0]}"
    )
    assert cams.ndim == 3, (
        f"Expected 3-D array (n_runs, H, W), got shape {cams.shape}"
    )

    # Every ClassifierOutputTarget instantiated must use the fixed class.
    assert len(recorded) >= 1, "ClassifierOutputTarget was never instantiated"
    wrong = [c for c in recorded if c != target_class]
    assert not wrong, (
        f"Target class drifted across passes. "
        f"Expected all={target_class}, found: {recorded}"
    )
