"""Newton-Schulz iterations for matrix orthogonalization.

The Newton-Schulz iteration approximates the polar factor of a matrix G—the
nearest semi-orthogonal matrix. This is the core computational primitive of Muon.

References:
    Keller Jordan et al., "Muon: An optimizer for hidden layers"
    https://kellerjordan.github.io/posts/muon/
    Amsel et al., "The Polar Express" (arXiv:2505.16932) — per-step coefficients.
"""

from collections import defaultdict

import torch
from torch import Tensor

__all__ = [
    "NS_COEFFICIENTS_DEFAULT",
    "NS_COEFFICIENTS_POLAR_EXPRESS",
    "NS_DTYPE_DEFAULT",
    "NS_EPS_DEFAULT",
    "NS_NORM_SCALE_DEFAULT",
    "NS_NORM_SCALE_POLAR_EXPRESS",
    "NS_STEPS_DEFAULT",
    "NSCoefficients",
    "newton_schulz",
    "newton_schulz_batched",
]

NSCoefficients = tuple[float, float, float] | tuple[tuple[float, float, float], ...]

# Quintic coefficients selected to maximize slope at zero.
NS_COEFFICIENTS_DEFAULT: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)

# Per-step coefficients from Polar Express (autoresearch).
NS_COEFFICIENTS_POLAR_EXPRESS: tuple[tuple[float, float, float], ...] = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)

NS_STEPS_DEFAULT: int = 5
NS_EPS_DEFAULT: float = 1e-6
NS_DTYPE_DEFAULT: torch.dtype | None = torch.bfloat16
NS_NORM_SCALE_DEFAULT: float = 1.0
NS_NORM_SCALE_POLAR_EXPRESS: float = 1.02


def newton_schulz(
    G: Tensor,
    ns_steps: int = NS_STEPS_DEFAULT,
    ns_coefficients: NSCoefficients = NS_COEFFICIENTS_DEFAULT,
    eps: float = NS_EPS_DEFAULT,
    ns_dtype: torch.dtype | None = NS_DTYPE_DEFAULT,
    ns_norm_scale: float = NS_NORM_SCALE_DEFAULT,
) -> Tensor:
    """Orthogonalize a single matrix via Newton-Schulz iteration.

    For G with SVD = USV^T, approximates US'V^T where S' ~ I.
    Uses a quintic polynomial for fast convergence. Supports per-step
    coefficients (Polar Express) and norm contraction.

    Args:
        G: Input tensor with ndim >= 2.
        ns_steps: Number of NS iterations.
        ns_coefficients: Quintic polynomial coefficients ``(a, b, c)`` or a tuple
            of per-step ``(a, b, c)`` triples (Polar Express).
        eps: Stability constant for Frobenius-norm normalization.
        ns_dtype: Dtype to use for NS iterations. ``None`` preserves input dtype.
        ns_norm_scale: Frobenius-norm contraction factor. 1.0 = no contraction,
            1.02 = Polar Express default.

    Returns:
        Orthogonalized tensor with the same shape and dtype as G.
    """
    if G.ndim < 2:
        raise ValueError(f"newton_schulz requires ndim >= 2, got {G.ndim}")
    original_shape = G.shape
    G = G.reshape(G.shape[0], -1)
    rows, cols = G.shape
    tall = rows > cols

    original_dtype = G.dtype
    if ns_dtype is not None and G.dtype != ns_dtype:
        G = G.to(ns_dtype)

    G = G / (G.norm() * ns_norm_scale + eps)

    # Determine per-step coefficients
    coeffs_list: list[tuple[float, float, float]]
    if isinstance(ns_coefficients[0], (int, float)):
        coeffs_list = [ns_coefficients] * ns_steps  # type: ignore[list-item]
    else:
        if len(ns_coefficients) < ns_steps:
            raise ValueError(f"Per-step coefficients has {len(ns_coefficients)} entries but ns_steps={ns_steps}")
        coeffs_list = list(ns_coefficients[:ns_steps])  # type: ignore[arg-type]

    for a, b, c in coeffs_list:
        if tall:
            A = G.mT @ G
            B = b * A + c * (A @ A)
            G = a * G + G @ B
        else:
            A = G @ G.mT
            B = b * A + c * (A @ A)
            G = a * G + B @ G

    if G.dtype != original_dtype:
        G = G.to(original_dtype)

    return G.reshape(original_shape)


def newton_schulz_batched(
    tensors: list[Tensor],
    ns_steps: int = NS_STEPS_DEFAULT,
    ns_coefficients: NSCoefficients = NS_COEFFICIENTS_DEFAULT,
    eps: float = NS_EPS_DEFAULT,
    ns_dtype: torch.dtype | None = NS_DTYPE_DEFAULT,
    ns_norm_scale: float = NS_NORM_SCALE_DEFAULT,
) -> list[Tensor]:
    """Orthogonalize a batch of matrices, grouped by shape for efficient matmul.

    Groups matrices by their flattened 2D shape, stacks each group into a batch
    tensor, and runs Newton-Schulz as a single batched matmul per group. Supports
    per-step coefficients (Polar Express) and norm contraction.

    Args:
        tensors: List of ≥2D tensors to orthogonalize.
        ns_steps: Number of NS iterations.
        ns_coefficients: Quintic polynomial coefficients ``(a, b, c)`` or a tuple
            of per-step ``(a, b, c)`` triples (Polar Express).
        eps: Stability constant for normalization.
        ns_dtype: Dtype to use for NS iterations. ``None`` preserves input dtype.
        ns_norm_scale: Frobenius-norm contraction factor.

    Returns:
        List of orthogonalized tensors in the same order as input.
    """
    if not tensors:
        return []
    if len(tensors) == 1:
        return [newton_schulz(tensors[0], ns_steps, ns_coefficients, eps, ns_dtype, ns_norm_scale)]

    # Determine per-step coefficients
    coeffs_list: list[tuple[float, float, float]]
    if isinstance(ns_coefficients[0], (int, float)):
        coeffs_list = [ns_coefficients] * ns_steps  # type: ignore[list-item]
    else:
        if len(ns_coefficients) < ns_steps:
            raise ValueError(f"Per-step coefficients has {len(ns_coefficients)} entries but ns_steps={ns_steps}")
        coeffs_list = list(ns_coefficients[:ns_steps])  # type: ignore[arg-type]

    # Prepare: flatten to 2D
    records: list[tuple[int, torch.Size, tuple[int, int], Tensor]] = []
    for i, t in enumerate(tensors):
        original_shape = t.shape
        t2d = t.reshape(t.shape[0], -1)
        records.append((i, original_shape, (t2d.shape[0], t2d.shape[1]), t2d))

    # Group by 2D shape
    shape_groups: dict[tuple[int, int], list[tuple[int, torch.Size, Tensor]]] = defaultdict(list)
    for idx, orig_shape, shape_2d, t2d in records:
        shape_groups[shape_2d].append((idx, orig_shape, t2d))

    results: list[tuple[int, Tensor]] = []

    for shape_2d, group in shape_groups.items():
        indices, orig_shapes, t2ds = zip(*group, strict=True)
        rows, cols = shape_2d
        tall = rows > cols

        G_batch = torch.stack(list(t2ds))  # (batch, rows, cols)

        original_dtype = G_batch.dtype
        if ns_dtype is not None and G_batch.dtype != ns_dtype:
            G_batch = G_batch.to(ns_dtype)

        norms = G_batch.flatten(1).norm(dim=1, keepdim=True).unsqueeze(-1) * ns_norm_scale + eps
        G_batch = G_batch / norms

        for a, b, c in coeffs_list:
            if tall:
                A = G_batch.mT @ G_batch
                B = b * A + c * (A @ A)
                G_batch = a * G_batch + G_batch @ B
            else:
                A = G_batch @ G_batch.mT
                B = b * A + c * (A @ A)
                G_batch = a * G_batch + B @ G_batch

        if G_batch.dtype != original_dtype:
            G_batch = G_batch.to(original_dtype)

        for j, (idx, orig_shape) in enumerate(zip(indices, orig_shapes, strict=True)):
            results.append((idx, G_batch[j].reshape(orig_shape)))

    results.sort(key=lambda x: x[0])
    return [t for _, t in results]
