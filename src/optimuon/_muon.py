"""Core Muon optimizer with foreach support, LR scaling modes, and corrections.

Muon applies Nesterov momentum followed by Newton-Schulz orthogonalization,
yielding the nearest semi-orthogonal update matrix—equivalent to steepest
descent under the spectral norm.

References:
    Keller Jordan et al., "Muon: An optimizer for hidden layers"
    https://kellerjordan.github.io/posts/muon/
    Moonshot AI, "Muon is Scalable for LLM Training" (arXiv:2502.16982)
"""

import math
import warnings
from collections.abc import Callable
from collections.abc import Iterable
from typing import Any
from typing import Literal

import torch
import torch.distributed as dist
from torch import Tensor
from torch.optim.optimizer import Optimizer

from ._corrections import apply_cautious_mask
from ._corrections import apply_cautious_weight_decay
from ._corrections import apply_mars_correction
from ._corrections import apply_normuon_rescale
from ._corrections import apply_weight_norm
from ._corrections import clip_grad_norm_foreach
from ._corrections import clip_update_norm_foreach
from ._newton_schulz import newton_schulz
from ._newton_schulz import newton_schulz_batched
from ._newton_schulz import NS_COEFFICIENTS_POLAR_EXPRESS
from ._newton_schulz import NS_DTYPE_DEFAULT
from ._newton_schulz import NS_EPS_DEFAULT
from ._newton_schulz import NS_NORM_SCALE_POLAR_EXPRESS
from ._newton_schulz import NS_STEPS_DEFAULT
from ._newton_schulz import NSCoefficients

__all__ = ["Muon"]

AdjustLrMode = Literal["original", "match_rms_adamw", "none"]
MomentumType = Literal["ema", "classical"]


def _match_rms_adamw_scale(param: Tensor) -> float:
    """Compute Moonshot's match_rms_adamw LR multiplier: 0.2 * sqrt(max(rows, cols))."""
    shape_2d = (param.shape[0], param[0].numel()) if param.ndim > 2 else tuple(param.shape)
    return 0.2 * math.sqrt(max(shape_2d[0], shape_2d[1]))


def _aspect_ratio_scale(param: Tensor) -> float:
    """Compute KJ's aspect-ratio scaling: sqrt(max(1, rows/cols))."""
    shape_2d = (param.shape[0], param[0].numel()) if param.ndim > 2 else tuple(param.shape)
    return max(1, shape_2d[0] / shape_2d[1]) ** 0.5


class Muon(Optimizer):
    """Muon optimizer for >=2D hidden-layer weight matrices.

    Args:
        params: Parameters or parameter groups. All must be >=2D.
        lr: Learning rate. 0.02 for ``"original"`` mode, ~2e-4 for ``"match_rms_adamw"``.
        momentum: SGD momentum coefficient.
        nesterov: Use Nesterov momentum.
        weight_decay: Decoupled weight decay (AdamW-style).
        ns_steps: Newton-Schulz iterations.
        ns_coefficients: Quintic polynomial coefficients ``(a, b, c)`` or per-step
            coefficients (Polar Express). Defaults to Polar Express.
        ns_eps: Stability constant for NS normalization.
        ns_norm_scale: Frobenius-norm contraction factor for NS. 1.0 = no contraction,
            1.02 = Polar Express default.
        ns_dtype: Dtype for NS iterations. Defaults to ``torch.bfloat16`` for speed.
            ``None`` preserves input dtype (original behavior).
        adjust_lr: LR scaling mode. ``"original"`` applies KJ aspect-ratio scaling
            ``sqrt(max(1, rows/cols))``. ``"match_rms_adamw"`` applies Moonshot's
            per-parameter scaling ``0.2 * sqrt(max(rows, cols))``.
            ``"none"`` applies no scaling (manual LR tuning).
        momentum_type: Momentum convention. ``"ema"`` uses ``m = beta*m + (1-beta)*g``
            (dominant convention). ``"classical"`` uses ``m = beta*m + g``.
        momentum_dtype: Dtype for momentum buffers. ``None`` inherits param dtype
            (default). Set to a lower-precision dtype (e.g., ``torch.bfloat16``) to
            reduce memory.
        foreach: Use ``torch._foreach_*`` for momentum/WD/update steps.
        batched_ns: Batch matrices by shape for Newton-Schulz.
        mars: Enable MARS gradient correction.
        mars_gamma: MARS correction strength.
        cautious: Enable cautious update masking.
        cautious_wd: Enable cautious weight decay (WD only where update and param
            agree in sign).
        normuon: Enable NorMuon adaptive variance reduction.
        normuon_beta2: Second-moment EMA coefficient for NorMuon.
        weight_norm: Normalize each weight matrix's Frobenius norm to ``sqrt(fan_out)``
            before each update. Prevents weight scale drift since Muon produces
            direction-only (orthogonalized) updates.
        grad_clip: Clip global gradient norm to this value.
        update_clip: Clip global update norm to this value (not supported with
            ``distributed=True``).
        distributed: Enable ``all_gather``-based gradient sharding.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        ns_steps: int = NS_STEPS_DEFAULT,
        ns_coefficients: NSCoefficients = NS_COEFFICIENTS_POLAR_EXPRESS,
        ns_eps: float = NS_EPS_DEFAULT,
        ns_norm_scale: float = NS_NORM_SCALE_POLAR_EXPRESS,
        ns_dtype: torch.dtype | None = NS_DTYPE_DEFAULT,
        adjust_lr: AdjustLrMode = "original",
        momentum_type: MomentumType = "ema",
        momentum_dtype: torch.dtype | None = None,
        foreach: bool = True,
        batched_ns: bool = True,
        mars: bool = False,
        mars_gamma: float = 0.025,
        cautious: bool = False,
        cautious_wd: bool = True,
        normuon: bool = True,
        normuon_beta2: float = 0.95,
        weight_norm: bool = False,
        grad_clip: float | None = None,
        update_clip: float | None = None,
        distributed: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if ns_steps < 1:
            raise ValueError(f"Invalid ns_steps: {ns_steps}")
        if adjust_lr not in ("original", "match_rms_adamw", "none"):
            raise ValueError(f"Invalid adjust_lr: {adjust_lr!r}")
        if momentum_type not in ("ema", "classical"):
            raise ValueError(f"Invalid momentum_type: {momentum_type!r}")
        if mars_gamma <= 0.0:
            raise ValueError(f"Invalid mars_gamma: {mars_gamma}")
        if grad_clip is not None and grad_clip <= 0.0:
            raise ValueError(f"Invalid grad_clip: {grad_clip}")
        if update_clip is not None and update_clip <= 0.0:
            raise ValueError(f"Invalid update_clip: {update_clip}")
        if ns_eps <= 0.0:
            raise ValueError(f"Invalid ns_eps: {ns_eps}")
        if ns_norm_scale <= 0.0:
            raise ValueError(f"Invalid ns_norm_scale: {ns_norm_scale}")
        if ns_dtype is not None and not isinstance(ns_dtype, torch.dtype):
            raise TypeError(f"ns_dtype must be a torch.dtype or None, got {type(ns_dtype)}")
        if momentum_dtype is not None and not isinstance(momentum_dtype, torch.dtype):
            raise TypeError(f"momentum_dtype must be a torch.dtype or None, got {type(momentum_dtype)}")
        if not 0.0 < normuon_beta2 < 1.0:
            raise ValueError(f"Invalid normuon_beta2: {normuon_beta2}")

        if update_clip is not None and distributed:
            warnings.warn(
                "update_clip is not supported with distributed=True and will be ignored. Use grad_clip instead.",
                stacklevel=2,
            )

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "weight_decay": weight_decay,
            "ns_steps": ns_steps,
            "ns_coefficients": ns_coefficients,
            "ns_eps": ns_eps,
            "ns_norm_scale": ns_norm_scale,
            "ns_dtype": ns_dtype,
            "adjust_lr": adjust_lr,
            "momentum_type": momentum_type,
            "momentum_dtype": momentum_dtype,
            "foreach": foreach,
            "batched_ns": batched_ns,
            "mars": mars,
            "mars_gamma": mars_gamma,
            "cautious": cautious,
            "cautious_wd": cautious_wd,
            "normuon": normuon,
            "normuon_beta2": normuon_beta2,
            "weight_norm": weight_norm,
            "grad_clip": grad_clip,
            "update_clip": update_clip,
            "distributed": distributed,
        }
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim < 2:
                    raise ValueError(
                        f"Muon requires >=2D parameters, got ndim={p.ndim} "
                        f"shape={tuple(p.shape)}. Route 1D/scalar params "
                        f"to an auxiliary optimizer (see CompositeMuon)."
                    )

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self._step_group(group)

        return loss

    def _step_group(self, group: dict[str, Any]) -> None:
        """Process a single parameter group."""
        params_with_grad: list[Tensor] = []
        grads: list[Tensor] = []
        momentum_bufs: list[Tensor] = []
        prev_grads: list[Tensor | None] = []
        normuon_bufs: list[Tensor] = []

        beta = group["momentum"]
        use_mars = group["mars"]
        use_normuon = group["normuon"]
        momentum_type = group["momentum_type"]

        for p in group["params"]:
            if p.grad is None:
                continue
            params_with_grad.append(p)
            grads.append(p.grad)

            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["momentum_buffer"] = torch.zeros_like(p, dtype=group["momentum_dtype"] or p.dtype)
                if use_mars:
                    state["prev_grad"] = None
            state["step"] += 1

            if use_normuon and "normuon_buffer" not in state:
                shape_2d = (p.shape[0], p[0].numel()) if p.ndim > 2 else tuple(p.shape)
                rows, cols = shape_2d
                buf_shape = (rows, 1) if rows >= cols else (1, cols)
                state["normuon_buffer"] = torch.zeros(buf_shape, device=p.device, dtype=p.dtype)

            momentum_bufs.append(state["momentum_buffer"])
            if use_mars:
                prev_grads.append(state.get("prev_grad"))
            if use_normuon:
                normuon_bufs.append(state["normuon_buffer"])

        if not params_with_grad:
            return

        # --- Gradient clipping ---
        if group["grad_clip"] is not None:
            clip_grad_norm_foreach(grads, group["grad_clip"])

        # --- Distributed path ---
        if group["distributed"] and dist.is_initialized() and dist.get_world_size() > 1:
            self._distributed_step(params_with_grad, grads, momentum_bufs, prev_grads, group)
            return

        # --- Save original grads for MARS prev_grad storage ---
        original_grads = [g.clone() for g in grads] if use_mars else None

        # --- MARS correction ---
        if use_mars:
            grads = [
                apply_mars_correction(g, prev_grads[i] if i < len(prev_grads) else None, beta, group["mars_gamma"])
                for i, g in enumerate(grads)
            ]

        # --- Momentum update ---
        use_foreach = group["foreach"]
        _needs_cast = (
            group["momentum_dtype"] is not None and len(grads) > 0 and grads[0].dtype != momentum_bufs[0].dtype
        )
        if _needs_cast and use_foreach:
            _grads_for_momentum = [g.to(momentum_bufs[0].dtype) for g in grads]
        else:
            _grads_for_momentum = grads

        if momentum_type == "ema":
            if use_foreach:
                torch._foreach_mul_(momentum_bufs, beta)  # type: ignore[attr-defined]
                torch._foreach_add_(momentum_bufs, _grads_for_momentum, alpha=1.0 - beta)  # type: ignore[attr-defined]
            else:
                for mb, g in zip(momentum_bufs, grads):
                    mb.mul_(beta).add_(g, alpha=1.0 - beta)
        else:  # classical
            if use_foreach:
                torch._foreach_mul_(momentum_bufs, beta)  # type: ignore[attr-defined]
                torch._foreach_add_(momentum_bufs, _grads_for_momentum)  # type: ignore[attr-defined]
            else:
                for mb, g in zip(momentum_bufs, grads):
                    mb.mul_(beta).add_(g)

        # --- Nesterov extrapolation ---
        if _needs_cast:
            _mbufs = [mb.to(grads[0].dtype) for mb in momentum_bufs]
        else:
            _mbufs = momentum_bufs

        if group["nesterov"]:
            if momentum_type == "ema":
                if use_foreach:
                    updates = torch._foreach_mul(grads, 1.0 - beta)  # type: ignore[attr-defined]
                    bufs_scaled = torch._foreach_mul(_mbufs, beta)  # type: ignore[attr-defined]
                    torch._foreach_add_(updates, bufs_scaled)  # type: ignore[attr-defined]
                else:
                    updates = [(1.0 - beta) * g + beta * mb for g, mb in zip(grads, _mbufs)]
            else:  # classical
                if use_foreach:
                    updates = torch._foreach_mul(_mbufs, beta)  # type: ignore[attr-defined]
                    torch._foreach_add_(updates, grads)  # type: ignore[attr-defined]
                else:
                    updates = [mb * beta + g for mb, g in zip(_mbufs, grads)]
        else:
            updates = list(_mbufs)

        # --- Newton-Schulz orthogonalization ---
        ns_steps = group["ns_steps"]
        ns_coeff = group["ns_coefficients"]
        ns_eps = group["ns_eps"]
        ns_dtype = group["ns_dtype"]
        ns_norm_scale = group["ns_norm_scale"]
        if group["batched_ns"] and len(updates) > 1:
            updates = newton_schulz_batched(updates, ns_steps, ns_coeff, ns_eps, ns_dtype, ns_norm_scale)
        else:
            updates = [newton_schulz(u, ns_steps, ns_coeff, ns_eps, ns_dtype, ns_norm_scale) for u in updates]

        # --- NorMuon adaptive variance reduction ---
        if use_normuon:
            for i, (u, p) in enumerate(zip(updates, params_with_grad)):
                shape_2d = (p.shape[0], p[0].numel()) if p.ndim > 2 else tuple(p.shape)
                rows, cols = shape_2d
                red_dim = -1 if rows >= cols else -2
                u_2d = u.reshape(rows, cols)
                u_2d = apply_normuon_rescale(u_2d, normuon_bufs[i], group["normuon_beta2"], red_dim)
                updates[i] = u_2d.reshape(u.shape)

        # --- Aspect-ratio scaling for "original" mode ---
        adjust_lr = group["adjust_lr"]
        if adjust_lr == "original":
            for i, p in enumerate(params_with_grad):
                ratio_scale = _aspect_ratio_scale(p)
                if ratio_scale != 1.0:
                    updates[i].mul_(ratio_scale)

        # --- Cautious masking ---
        if group["cautious"]:
            updates = [apply_cautious_mask(u, g) for u, g in zip(updates, grads)]

        # --- Update clipping ---
        if group["update_clip"] is not None:
            clip_update_norm_foreach(updates, group["update_clip"])

        # --- Weight normalization ---
        if group["weight_norm"]:
            for p in params_with_grad:
                apply_weight_norm(p)

        # --- Weight decay + parameter update ---
        lr = group["lr"]
        wd = group["weight_decay"]

        if group["cautious_wd"] and wd > 0:
            if adjust_lr == "match_rms_adamw":
                # Per-param LR — must loop for both WD and update
                for p, u in zip(params_with_grad, updates):
                    effective_lr = lr * _match_rms_adamw_scale(p)
                    apply_cautious_weight_decay(p, u, effective_lr, wd)
                    p.add_(u.reshape(p.shape), alpha=-effective_lr)
            else:
                # Uniform LR — loop for cautious WD, foreach for update
                for p, u in zip(params_with_grad, updates):
                    apply_cautious_weight_decay(p, u, lr, wd)
                if use_foreach:
                    torch._foreach_add_(params_with_grad, updates, alpha=-lr)  # type: ignore[attr-defined]
                else:
                    for p, u in zip(params_with_grad, updates):
                        p.add_(u.reshape(p.shape), alpha=-lr)
        elif use_foreach and adjust_lr in ("original", "none"):
            # Uniform LR -> single foreach call
            if wd > 0:
                torch._foreach_mul_(params_with_grad, 1.0 - lr * wd)  # type: ignore[attr-defined]
            torch._foreach_add_(params_with_grad, updates, alpha=-lr)  # type: ignore[attr-defined]
        else:
            # Per-parameter LR (match_rms_adamw) or non-foreach fallback
            for p, u in zip(params_with_grad, updates):
                effective_lr = lr * _match_rms_adamw_scale(p) if adjust_lr == "match_rms_adamw" else lr
                if wd > 0:
                    p.mul_(1.0 - effective_lr * wd)
                p.add_(u.reshape(p.shape), alpha=-effective_lr)

        # --- Store prev_grad for MARS (original, uncorrected gradients) ---
        if use_mars:
            for i, og in enumerate(original_grads):  # type: ignore[arg-type]
                self.state[params_with_grad[i]]["prev_grad"] = og

    def _distributed_step(
        self,
        params: list[Tensor],
        grads: list[Tensor],
        momentum_bufs: list[Tensor],
        prev_grads: list[Tensor | None],
        group: dict[str, Any],
    ) -> None:
        """Distributed step: shard NS computation across ranks, all_gather results.

        Follows KellerJordan/Muon's pattern: sort by descending size, pad to
        world_size multiple, each rank handles its shard.
        """
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        beta = group["momentum"]
        momentum_type = group["momentum_type"]
        ns_steps = group["ns_steps"]
        ns_coeff = group["ns_coefficients"]
        ns_eps = group["ns_eps"]
        ns_norm_scale = group["ns_norm_scale"]
        ns_dtype = group["ns_dtype"]
        lr = group["lr"]
        wd = group["weight_decay"]
        adjust_lr = group["adjust_lr"]
        use_mars = group["mars"]
        use_normuon = group["normuon"]
        use_weight_norm = group["weight_norm"]
        use_cautious_wd = group["cautious_wd"]

        # Sort by descending numel for load balancing
        order = sorted(range(len(params)), key=lambda i: params[i].numel(), reverse=True)
        n = len(params)

        # Pad to world_size multiple
        pad_n = (world_size - n % world_size) % world_size
        padded_params = [params[order[i]] if i < n else torch.zeros_like(params[order[-1]]) for i in range(n + pad_n)]

        for base_i in range(0, n + pad_n, world_size):
            my_idx = base_i + rank
            orig_idx = order[my_idx] if my_idx < n else -1

            if orig_idx >= 0:
                p = params[orig_idx]
                g = grads[orig_idx]
                mb = momentum_bufs[orig_idx]

                # Save original grad for MARS prev_grad storage
                original_g = g.clone() if use_mars else None

                # MARS
                if use_mars and orig_idx < len(prev_grads):
                    g = apply_mars_correction(g, prev_grads[orig_idx], beta, group["mars_gamma"])

                # Momentum
                if momentum_type == "ema":
                    mb.mul_(beta).add_(g, alpha=1.0 - beta)
                else:  # classical
                    mb.mul_(beta).add_(g)

                # Nesterov
                if group["nesterov"]:
                    if momentum_type == "ema":
                        update = (1.0 - beta) * g + beta * mb
                    else:  # classical
                        update = mb * beta + g
                else:
                    update = mb.to(p.dtype) if mb.dtype != p.dtype else mb
                if update.dtype != p.dtype:
                    update = update.to(p.dtype)

                # Newton-Schulz
                update = newton_schulz(update, ns_steps, ns_coeff, ns_eps, ns_dtype, ns_norm_scale)

                # NorMuon
                if use_normuon:
                    state = self.state[p]
                    if "normuon_buffer" not in state:
                        shape_2d = (p.shape[0], p[0].numel()) if p.ndim > 2 else tuple(p.shape)
                        rows, cols = shape_2d
                        buf_shape = (rows, 1) if rows >= cols else (1, cols)
                        state["normuon_buffer"] = torch.zeros(buf_shape, device=p.device, dtype=p.dtype)
                    shape_2d = (p.shape[0], p[0].numel()) if p.ndim > 2 else tuple(p.shape)
                    rows, cols = shape_2d
                    red_dim = -1 if rows >= cols else -2
                    u_2d = update.reshape(rows, cols)
                    u_2d = apply_normuon_rescale(u_2d, state["normuon_buffer"], group["normuon_beta2"], red_dim)
                    update = u_2d.reshape(update.shape)

                # Aspect-ratio scaling for "original" mode
                if adjust_lr == "original":
                    ratio_scale = _aspect_ratio_scale(p)
                    if ratio_scale != 1.0:
                        update.mul_(ratio_scale)

                # Cautious
                if group["cautious"]:
                    update = apply_cautious_mask(update, g)

                # Weight normalization
                if use_weight_norm:
                    apply_weight_norm(p)

                # Weight decay + update
                effective_lr = lr * _match_rms_adamw_scale(p) if adjust_lr == "match_rms_adamw" else lr
                if use_cautious_wd and wd > 0:
                    apply_cautious_weight_decay(p, update, effective_lr, wd)
                elif wd > 0:
                    p.mul_(1.0 - effective_lr * wd)
                p.add_(update.reshape(p.shape), alpha=-effective_lr)

                # Store prev_grad (original, uncorrected gradient)
                if use_mars:
                    self.state[p]["prev_grad"] = original_g

            # All-gather: synchronize updated params across ranks
            chunk = padded_params[base_i : base_i + world_size]
            dist.all_gather(chunk, chunk[rank])
