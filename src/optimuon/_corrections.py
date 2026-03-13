"""Optional update corrections for the Muon optimizer.

- MARS: gradient correction using consecutive gradient differences.
- Cautious: masks update components that disagree in sign with the gradient.
- Clipping: gradient-norm and update-norm clipping (foreach-friendly).
"""

import torch
from torch import Tensor

__all__ = [
    "apply_cautious_mask",
    "apply_mars_correction",
    "clip_grad_norm_foreach",
    "clip_update_norm_foreach",
]


def apply_mars_correction(
    grad: Tensor,
    prev_grad: Tensor | None,
    beta: float,
    gamma: float,
) -> Tensor:
    """Apply MARS gradient correction.

    Incorporates curvature from consecutive gradient differences, scaled by
    gamma * beta / (1 - beta). Returns the unmodified gradient on the first step.
    """
    if prev_grad is None:
        return grad
    correction = gamma * beta / (1.0 - beta) * (grad - prev_grad)
    return grad + correction


def apply_cautious_mask(update: Tensor, grad: Tensor) -> Tensor:
    """Mask update components that disagree in sign with the gradient.

    Surviving components are rescaled to preserve the original update norm.
    """
    mask = (update * grad > 0).to(update.dtype)
    masked = update * mask
    n_surviving = mask.sum()
    if n_surviving > 0:
        masked = masked * (mask.numel() / n_surviving)
    return masked


def clip_grad_norm_foreach(grads: list[Tensor], max_norm: float) -> float:
    """Clip global gradient norm across a list of tensors, in-place.

    Returns the total norm before clipping.
    """
    if not grads:
        return 0.0
    norms = torch._foreach_norm(grads)  # type: ignore[attr-defined]
    total_norm_sq = sum(n.item() ** 2 for n in norms)
    total_norm = total_norm_sq**0.5
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        torch._foreach_mul_(grads, clip_coef)  # type: ignore[attr-defined]
    return total_norm


def clip_update_norm_foreach(updates: list[Tensor], max_norm: float) -> float:
    """Clip global update norm across a list of tensors, in-place.

    Semantic alias for :func:`clip_grad_norm_foreach`, applied to optimizer
    updates rather than raw gradients. The operation is identical.

    Returns the total norm before clipping.
    """
    return clip_grad_norm_foreach(updates, max_norm)
