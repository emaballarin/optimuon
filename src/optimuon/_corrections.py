"""Optional update corrections for the Muon optimizer.

- MARS: gradient correction using consecutive gradient differences.
- Cautious: masks update components that disagree in sign with the gradient.
- Clipping: gradient-norm and update-norm clipping (foreach-friendly).
- NorMuon: adaptive per-dimension variance reduction.
- Cautious weight decay: WD masked by update–parameter sign alignment.
- Weight normalization: Frobenius-norm clamping to sqrt(fan_out).
"""

import torch
from torch import Tensor

__all__ = [
    "apply_cautious_mask",
    "apply_cautious_weight_decay",
    "apply_mars_correction",
    "apply_normuon_rescale",
    "apply_weight_norm",
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


def apply_normuon_rescale(
    update: Tensor,
    second_momentum_buffer: Tensor,
    beta2: float,
    red_dim: int,
) -> Tensor:
    """NorMuon: rescale per-dimension variance while preserving total norm.

    Applies adaptive variance reduction via a second-moment EMA, rescaling each
    row (or column) of the update to equalize variance across dimensions while
    keeping the overall Frobenius norm unchanged. Mutates buffer in-place.
    """
    v_mean = update.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = update.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()

    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)

    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()

    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    return update * final_scale.to(update.dtype)


def apply_weight_norm(param: Tensor) -> None:
    """Normalize param Frobenius norm to sqrt(fan_out). Modifies param in-place."""
    target_norm = param.shape[0] ** 0.5
    current_norm = param.norm()
    if current_norm > 0:
        param.mul_(target_norm / current_norm)


def apply_cautious_weight_decay(param: Tensor, update: Tensor, lr: float, wd: float) -> None:
    """Weight decay only where update and param agree in sign. Modifies param in-place."""
    mask = (update * param >= 0).to(param.dtype)
    param.sub_(lr * wd * param * mask)


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
