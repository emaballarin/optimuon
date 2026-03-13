"""Newton-Schulz iterations for matrix orthogonalization.

The Newton-Schulz iteration approximates the polar factor of a matrix G—the
nearest semi-orthogonal matrix. This is the core computational primitive of Muon.

Reference coefficients from Keller Jordan's Muon:
https://kellerjordan.github.io/posts/muon/
"""

from collections import defaultdict

import torch
from torch import Tensor

__all__ = [
    "NS_COEFFICIENTS_DEFAULT",
    "NS_EPS_DEFAULT",
    "NS_STEPS_DEFAULT",
    "newton_schulz",
    "newton_schulz_batched",
]

# Quintic coefficients selected to maximize slope at zero.
NS_COEFFICIENTS_DEFAULT: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)
NS_STEPS_DEFAULT: int = 5
NS_EPS_DEFAULT: float = 1e-7


def newton_schulz(
    G: Tensor,
    ns_steps: int = NS_STEPS_DEFAULT,
    ns_coefficients: tuple[float, float, float] = NS_COEFFICIENTS_DEFAULT,
    eps: float = NS_EPS_DEFAULT,
) -> Tensor:
    """Orthogonalize a single matrix via Newton-Schulz iteration.

    For G with SVD = USV^T, approximates US'V^T where S' ~ I.
    Uses a quintic polynomial for fast convergence.

    Args:
        G: Input tensor with ndim >= 2.
        ns_steps: Number of NS iterations.
        ns_coefficients: Quintic polynomial coefficients (a, b, c).
        eps: Stability constant for Frobenius-norm normalization.

    Returns:
        Orthogonalized tensor with the same shape as G.
    """
    if G.ndim < 2:
        raise ValueError(f"newton_schulz requires ndim >= 2, got {G.ndim}")
    original_shape = G.shape
    G = G.reshape(G.shape[0], -1)
    rows, cols = G.shape
    transposed = rows > cols
    if transposed:
        G = G.T

    G = G / (G.norm() + eps)

    a, b, c = ns_coefficients
    for _ in range(ns_steps):
        A = G @ G.T
        B = b * A + c * A @ A
        G = a * G + B @ G

    if transposed:
        G = G.T
    return G.reshape(original_shape)


def newton_schulz_batched(
    tensors: list[Tensor],
    ns_steps: int = NS_STEPS_DEFAULT,
    ns_coefficients: tuple[float, float, float] = NS_COEFFICIENTS_DEFAULT,
    eps: float = NS_EPS_DEFAULT,
) -> list[Tensor]:
    """Orthogonalize a batch of matrices, grouped by shape for efficient matmul.

    Groups matrices by their flattened 2D shape, stacks each group into a batch
    tensor, and runs Newton-Schulz as a single batched matmul per group.

    Args:
        tensors: List of ≥2D tensors to orthogonalize.
        ns_steps: Number of NS iterations.
        ns_coefficients: Quintic polynomial coefficients (a, b, c).
        eps: Stability constant for normalization.

    Returns:
        List of orthogonalized tensors in the same order as input.
    """
    if not tensors:
        return []
    if len(tensors) == 1:
        return [newton_schulz(tensors[0], ns_steps, ns_coefficients, eps)]

    # Prepare: flatten to 2D, ensure tall orientation
    records: list[tuple[int, torch.Size, tuple[int, int], bool, Tensor]] = []
    for i, t in enumerate(tensors):
        original_shape = t.shape
        t2d = t.reshape(t.shape[0], -1)
        rows, cols = t2d.shape
        transposed = rows > cols
        if transposed:
            t2d = t2d.T
        records.append((i, original_shape, t2d.shape, transposed, t2d))

    # Group by 2D shape
    shape_groups: dict[tuple[int, int], list[tuple[int, torch.Size, bool, Tensor]]] = defaultdict(list)
    for idx, orig_shape, shape_2d, transp, t2d in records:
        shape_groups[shape_2d].append((idx, orig_shape, transp, t2d))

    results: list[tuple[int, Tensor]] = []
    a, b, c = ns_coefficients

    for group in shape_groups.values():
        indices, orig_shapes, transposeds, t2ds = zip(*group, strict=True)

        G_batch = torch.stack(list(t2ds))  # (batch, rows, cols)
        norms = G_batch.flatten(1).norm(dim=1, keepdim=True).unsqueeze(-1) + eps
        G_batch = G_batch / norms

        for _ in range(ns_steps):
            A = G_batch @ G_batch.transpose(-2, -1)
            B = b * A + c * A @ A
            G_batch = a * G_batch + B @ G_batch

        for j, (idx, orig_shape, transp) in enumerate(zip(indices, orig_shapes, transposeds, strict=True)):
            out = G_batch[j]
            if transp:
                out = out.T
            results.append((idx, out.reshape(orig_shape)))

    results.sort(key=lambda x: x[0])
    return [t for _, t in results]
