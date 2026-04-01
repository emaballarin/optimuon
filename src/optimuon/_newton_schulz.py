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
    "GNS_DTYPE_DEFAULT",
    "GNS_RESTART_AFTER_DEFAULT",
    "NS_COEFFICIENTS_DEFAULT",
    "NS_COEFFICIENTS_POLAR_EXPRESS",
    "NS_DTYPE_DEFAULT",
    "NS_EPS_DEFAULT",
    "NS_NORM_SCALE_DEFAULT",
    "NS_NORM_SCALE_POLAR_EXPRESS",
    "NS_STEPS_DEFAULT",
    "NSCoefficients",
    "gram_newton_schulz",
    "gram_newton_schulz_batched",
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

GNS_DTYPE_DEFAULT: torch.dtype | None = torch.float16
GNS_RESTART_AFTER_DEFAULT: int = 2


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


def _resolve_coeffs(ns_coefficients: NSCoefficients, ns_steps: int) -> list[tuple[float, float, float]]:
    """Expand coefficients to a per-step list."""
    if isinstance(ns_coefficients[0], (int, float)):
        return [ns_coefficients] * ns_steps  # type: ignore[list-item]
    if len(ns_coefficients) < ns_steps:
        raise ValueError(f"Per-step coefficients has {len(ns_coefficients)} entries but ns_steps={ns_steps}")
    return list(ns_coefficients[:ns_steps])  # type: ignore[arg-type]


def gram_newton_schulz(
    G: Tensor,
    ns_steps: int = NS_STEPS_DEFAULT,
    ns_coefficients: NSCoefficients = NS_COEFFICIENTS_POLAR_EXPRESS,
    eps: float = NS_EPS_DEFAULT,
    ns_dtype: torch.dtype | None = GNS_DTYPE_DEFAULT,
    ns_norm_scale: float = NS_NORM_SCALE_POLAR_EXPRESS,
    restart_after: int | None = GNS_RESTART_AFTER_DEFAULT,
) -> Tensor:
    """Orthogonalize a single matrix via Gram Newton-Schulz iteration.

    Reformulation of Newton-Schulz that iterates on the smaller symmetric Gram
    matrix instead of the full rectangular matrix, reducing FLOPs by 40-60% for
    rectangular matrices. Falls back to standard Newton-Schulz for square matrices.

    For G with SVD = USV^T, approximates UV^T (the polar factor).

    Args:
        G: Input tensor with ndim >= 2.
        ns_steps: Number of NS iterations.
        ns_coefficients: Quintic polynomial coefficients ``(a, b, c)`` or a tuple
            of per-step ``(a, b, c)`` triples (Polar Express).
        eps: Stability constant for Frobenius-norm normalization.
        ns_dtype: Dtype to use for iterations. Defaults to ``torch.float16`` for
            Gram NS (better precision than bfloat16 for the Gram matrix).
            ``None`` preserves input dtype.
        ns_norm_scale: Frobenius-norm contraction factor.
        restart_after: Re-materialize rectangular product after this many iterations
            to prevent half-precision divergence. ``None`` disables restarting.
            Default 2 is tuned for Polar Express coefficients.

    Returns:
        Orthogonalized tensor with the same shape and dtype as G.

    References:
        Zhang et al., "Gram Newton-Schulz" (Dao-AILab, 2026).
    """
    if G.ndim < 2:
        raise ValueError(f"gram_newton_schulz requires ndim >= 2, got {G.ndim}")

    original_shape = G.shape
    G = G.reshape(G.shape[0], -1)
    rows, cols = G.shape

    # No Gram benefit for square matrices — fall back to standard NS.
    if rows == cols:
        result = newton_schulz(G, ns_steps, ns_coefficients, eps, ns_dtype, ns_norm_scale)
        return result.reshape(original_shape)

    tall = rows > cols

    original_dtype = G.dtype
    if ns_dtype is not None and G.dtype != ns_dtype:
        G = G.to(ns_dtype)

    G = G / (G.norm() * ns_norm_scale + eps)

    # Compute initial Gram matrix (always on the smaller dimension).
    if tall:
        R = G.mT @ G  # (cols, cols)
    else:
        R = G @ G.mT  # (rows, rows)

    coeffs_list = _resolve_coeffs(ns_coefficients, ns_steps)

    # Q = None is a sentinel representing the identity matrix.
    # On the first iteration (and after each restart), we avoid a redundant
    # matmul against identity by materializing Q = Z directly.
    Q: Tensor | None = None

    for t, (a, b, c) in enumerate(coeffs_list):
        # Z̃ = b*R + c*R² (the polynomial without the identity term)
        Z_tilde = b * R + c * (R @ R)

        # --- Q update (accumulate transform) ---
        if Q is None:
            # Q was identity: Q_new = Z̃ + a*I
            Q = Z_tilde.clone()
            Q.diagonal().add_(a)
        elif tall:
            # Tall: G_{t+1} = G_t @ Z_t, so Q_{t+1} = Q_t @ Z_t
            Q = torch.addmm(Q, Q, Z_tilde, beta=a, alpha=1.0)
        else:
            # Non-tall: G_{t+1} = Z_t @ G_t, so Q_{t+1} = Z_t @ Q_t
            Q = torch.addmm(Q, Z_tilde, Q, beta=a, alpha=1.0)

        # --- R update: R_new = Z @ R @ Z ---
        # RZ = R @ Z = R @ Z̃ + a*R
        RZ = torch.addmm(R, R, Z_tilde, beta=a, alpha=1.0)
        # R_new = Z @ RZ = Z̃ @ RZ + a*RZ
        R = torch.addmm(RZ, Z_tilde, RZ, beta=a, alpha=1.0)

        # --- Restart: flush spurious negative eigenvalues ---
        if restart_after is not None and t == restart_after and t < len(coeffs_list) - 1:
            if tall:
                G = G @ Q
                R = G.mT @ G
            else:
                G = Q @ G
                R = G @ G.mT
            Q = None

    # Final: apply accumulated Q to G.
    if Q is not None:
        G = G @ Q if tall else Q @ G

    if G.dtype != original_dtype:
        G = G.to(original_dtype)

    return G.reshape(original_shape)


def gram_newton_schulz_batched(
    tensors: list[Tensor],
    ns_steps: int = NS_STEPS_DEFAULT,
    ns_coefficients: NSCoefficients = NS_COEFFICIENTS_POLAR_EXPRESS,
    eps: float = NS_EPS_DEFAULT,
    ns_dtype: torch.dtype | None = GNS_DTYPE_DEFAULT,
    ns_norm_scale: float = NS_NORM_SCALE_POLAR_EXPRESS,
    restart_after: int | None = GNS_RESTART_AFTER_DEFAULT,
) -> list[Tensor]:
    """Orthogonalize a batch of matrices via Gram Newton-Schulz, grouped by shape.

    Groups matrices by their flattened 2D shape, stacks each group, and runs
    batched Gram Newton-Schulz. Square-shape groups fall back to standard
    Newton-Schulz (no Gram benefit). Supports per-step coefficients and restart.

    Args:
        tensors: List of >=2D tensors to orthogonalize.
        ns_steps: Number of NS iterations.
        ns_coefficients: Quintic polynomial coefficients.
        eps: Stability constant for normalization.
        ns_dtype: Dtype for iterations. Defaults to ``torch.float16``.
        ns_norm_scale: Frobenius-norm contraction factor.
        restart_after: Re-materialize after this many iterations. ``None`` disables.

    Returns:
        List of orthogonalized tensors in the same order as input.
    """
    if not tensors:
        return []
    if len(tensors) == 1:
        return [gram_newton_schulz(tensors[0], ns_steps, ns_coefficients, eps, ns_dtype, ns_norm_scale, restart_after)]

    coeffs_list = _resolve_coeffs(ns_coefficients, ns_steps)

    # Prepare: flatten to 2D.
    records: list[tuple[int, torch.Size, tuple[int, int], Tensor]] = []
    for i, t in enumerate(tensors):
        original_shape = t.shape
        t2d = t.reshape(t.shape[0], -1)
        records.append((i, original_shape, (t2d.shape[0], t2d.shape[1]), t2d))

    # Group by 2D shape.
    shape_groups: dict[tuple[int, int], list[tuple[int, torch.Size, Tensor]]] = defaultdict(list)
    for idx, orig_shape, shape_2d, t2d in records:
        shape_groups[shape_2d].append((idx, orig_shape, t2d))

    results: list[tuple[int, Tensor]] = []

    for shape_2d, group in shape_groups.items():
        indices, orig_shapes, t2ds = zip(*group, strict=True)
        rows, cols = shape_2d
        tall = rows > cols
        is_square = rows == cols

        G_batch = torch.stack(list(t2ds))  # (batch, rows, cols)

        original_dtype = G_batch.dtype
        if ns_dtype is not None and G_batch.dtype != ns_dtype:
            G_batch = G_batch.to(ns_dtype)

        norms = G_batch.flatten(1).norm(dim=1, keepdim=True).unsqueeze(-1) * ns_norm_scale + eps
        G_batch = G_batch / norms

        if is_square:
            # Standard NS for square matrices (no Gram benefit).
            for a, b, c in coeffs_list:
                A = G_batch @ G_batch.mT
                B = b * A + c * (A @ A)
                G_batch = a * G_batch + B @ G_batch
        else:
            # Gram NS for rectangular matrices.
            if tall:
                R = G_batch.mT @ G_batch  # (batch, cols, cols)
            else:
                R = G_batch @ G_batch.mT  # (batch, rows, rows)

            Q: Tensor | None = None

            for t, (a, b, c) in enumerate(coeffs_list):
                Z_tilde = b * R + c * (R @ R)

                if Q is None:
                    Q = Z_tilde.clone()
                    Q.diagonal(dim1=-2, dim2=-1).add_(a)
                elif tall:
                    Q = torch.baddbmm(Q, Q, Z_tilde, beta=a, alpha=1.0)
                else:
                    Q = torch.baddbmm(Q, Z_tilde, Q, beta=a, alpha=1.0)

                RZ = torch.baddbmm(R, R, Z_tilde, beta=a, alpha=1.0)
                R = torch.baddbmm(RZ, Z_tilde, RZ, beta=a, alpha=1.0)

                if restart_after is not None and t == restart_after and t < len(coeffs_list) - 1:
                    if tall:
                        G_batch = G_batch @ Q
                        R = G_batch.mT @ G_batch
                    else:
                        G_batch = Q @ G_batch
                        R = G_batch @ G_batch.mT
                    Q = None

            if Q is not None:
                G_batch = G_batch @ Q if tall else Q @ G_batch

        if G_batch.dtype != original_dtype:
            G_batch = G_batch.to(original_dtype)

        for j, (idx, orig_shape) in enumerate(zip(indices, orig_shapes, strict=True)):
            results.append((idx, G_batch[j].reshape(orig_shape)))

    results.sort(key=lambda x: x[0])
    return [t for _, t in results]
