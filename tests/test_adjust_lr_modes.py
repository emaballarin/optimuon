"""Tests for the aspect-ratio LR scaling modes: original (KJ), mup, interp."""

import math

import pytest
import torch
from optimuon import AdjustLrMode
from optimuon import Muon
from optimuon._muon import _spectral_floor
from optimuon._muon import _spectral_scale

# (rows, cols) = (d_out, d_in): fan-out, fan-in, square.
FAN_OUT = (3072, 768)
FAN_IN = (768, 3072)
SQUARE = (768, 768)
CONV = (64, 32, 3, 3)  # ndim > 2; flattens to (64, 288).


def _conv_mup_scale() -> float:
    """Closed-form MuP scale for the CONV shape: sqrt(d_out/d_in)."""
    return (CONV[0] / (CONV[1] * CONV[2] * CONV[3])) ** 0.5


# --------------------------------------------------------------------------- #
# Scale values (closed form, via the private helpers)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("adjust_lr", "tau", "shape", "expected"),
    [
        # original (floor = 1.0): KJ, truncates fan-in to 1.0.
        ("original", 1.0, FAN_OUT, 2.0),
        ("original", 1.0, FAN_IN, 1.0),
        ("original", 1.0, SQUARE, 1.0),
        ("original", 1.0, CONV, 1.0),
        # mup (floor = 0.0): no truncation.
        ("mup", 1.0, FAN_OUT, 2.0),
        ("mup", 1.0, FAN_IN, 0.5),
        ("mup", 1.0, SQUARE, 1.0),
        ("mup", 1.0, CONV, _conv_mup_scale()),
        # interp at the endpoints reproduces original (tau=1) and mup (tau=0).
        ("interp", 1.0, FAN_OUT, 2.0),
        ("interp", 1.0, FAN_IN, 1.0),
        ("interp", 1.0, SQUARE, 1.0),
        ("interp", 0.0, FAN_OUT, 2.0),
        ("interp", 0.0, FAN_IN, 0.5),
        ("interp", 0.0, SQUARE, 1.0),
        # interp mid-range (floor = 0.5).
        ("interp", 0.5, FAN_OUT, 2.0),
        ("interp", 0.5, FAN_IN, 0.5),
        ("interp", 0.5, SQUARE, 1.0),
        ("interp", 0.5, CONV, 0.5),  # max(0.5, ~0.471) = 0.5
    ],
)
def test_scale_values(adjust_lr: str, tau: float, shape: tuple[int, ...], expected: float) -> None:
    """Aspect-ratio scale matches the closed form, not swapping rows/cols."""
    floor = _spectral_floor(adjust_lr, tau)
    scale = _spectral_scale(torch.zeros(shape), floor)
    assert scale == pytest.approx(expected, rel=1e-12)


def test_floor_selection() -> None:
    """Floor is 0 for mup, tau for interp, 1 for every other mode."""
    assert _spectral_floor("mup", 0.7) == 0.0
    assert _spectral_floor("interp", 0.3) == 0.3
    assert _spectral_floor("interp", 0.0) == 0.0
    assert _spectral_floor("original", 0.3) == 1.0
    assert _spectral_floor("none", 0.3) == 1.0
    assert _spectral_floor("match_rms_adamw", 0.3) == 1.0


def test_fan_in_out_distinct() -> None:
    """MuP gives distinct values on transposed shapes (no d_in/d_out swap)."""
    floor = _spectral_floor("mup", 0.0)
    assert _spectral_scale(torch.zeros(FAN_OUT), floor) == 2.0
    assert _spectral_scale(torch.zeros(FAN_IN), floor) == 0.5


# --------------------------------------------------------------------------- #
# End-to-end equivalence and live-tau behaviour
# --------------------------------------------------------------------------- #

_E2E_SHAPES = [(48, 16), (16, 48), (32, 32)]  # fan-out, fan-in, square


def _step_once(mode: AdjustLrMode, base: list[torch.Tensor], grads: list[torch.Tensor], **kwargs) -> list[torch.Tensor]:
    """Run one Muon step over fresh clones of ``base`` with fixed ``grads``."""
    params = [b.clone().requires_grad_(True) for b in base]
    for p, g in zip(params, grads):
        p.grad = g.clone()
    opt = Muon(params, lr=0.1, adjust_lr=mode, ns_dtype=torch.float32, **kwargs)
    opt.step()
    return [p.detach().clone() for p in params]


def test_interp_endpoints_match_original_and_mup() -> None:
    """interp(tau=1) == original and interp(tau=0) == mup after one step."""
    gen = torch.Generator().manual_seed(0)
    base = [torch.randn(s, generator=gen) for s in _E2E_SHAPES]
    grads = [torch.randn(s, generator=gen) for s in _E2E_SHAPES]

    original = _step_once("original", base, grads)
    interp_kj = _step_once("interp", base, grads, tau=1.0)
    for a, b in zip(original, interp_kj):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-5)

    mup = _step_once("mup", base, grads)
    interp_mup = _step_once("interp", base, grads, tau=0.0)
    for a, b in zip(mup, interp_mup):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-5)


def test_tau_is_read_per_step() -> None:
    """Mutating group['tau'] between steps changes the applied scale immediately."""
    # Fan-in param with sqrt(d_out/d_in) = 0.25 < 0.5, so the floor binds for both
    # tau=1.0 (scale 1.0) and tau=0.5 (scale 0.5): a clean 2x difference.
    shape = (8, 128)
    gen = torch.Generator().manual_seed(0)
    base = torch.randn(shape, generator=gen)
    grad = torch.randn(shape, generator=gen)
    lr = 0.1

    def run(mutate_tau: float | None) -> torch.Tensor:
        p = base.clone().requires_grad_(True)
        opt = Muon([p], lr=lr, adjust_lr="interp", tau=1.0, weight_decay=0.0, ns_dtype=torch.float32)
        # Step 1 (tau=1.0) establishes momentum identically for both branches.
        p.grad = grad.clone()
        opt.step()
        before = p.detach().clone()
        if mutate_tau is not None:
            opt.param_groups[0]["tau"] = mutate_tau
        # Step 2 with the same grad; delta isolates the applied scale.
        p.grad = grad.clone()
        opt.step()
        return before - p.detach()  # = lr * scale * Phi_2

    delta_kj = run(None)  # tau stays 1.0 -> scale 1.0
    delta_mup = run(0.5)  # tau -> 0.5 -> scale 0.5
    # If tau were baked in at construction, delta_mup would equal delta_kj.
    assert not torch.allclose(delta_mup, delta_kj)
    assert torch.allclose(delta_mup, 0.5 * delta_kj, atol=1e-6, rtol=1e-5)


# --------------------------------------------------------------------------- #
# Weight-decay independence: MuP/interp scale the update, not decoupled WD
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ["original", "mup", "interp"])
def test_wd_independent_of_scale(mode: AdjustLrMode) -> None:
    """Decoupled WD shrink is 1 - lr*wd for the original family, on every shape."""
    shape = (16, 64)  # fan-in, non-square
    lr, wd = 0.1, 0.1
    gen = torch.Generator().manual_seed(0)
    p = torch.randn(shape, generator=gen).requires_grad_(True)
    before = p.detach().clone()
    opt = Muon([p], lr=lr, weight_decay=wd, adjust_lr=mode, tau=0.5, cautious_wd=False, ns_dtype=torch.float32)
    p.grad = torch.zeros_like(p)  # zero grad -> zero orthogonalised update
    opt.step()
    ratio = p.detach() / before
    assert torch.allclose(ratio, torch.full_like(ratio, 1.0 - lr * wd), atol=1e-6)


def test_match_rms_adamw_scales_wd() -> None:
    """match_rms_adamw folds the scale into the LR, so WD shrink differs."""
    shape = (16, 64)  # non-square: 0.2*sqrt(64) = 1.6 != 1
    lr, wd = 0.1, 0.1
    gen = torch.Generator().manual_seed(0)
    p = torch.randn(shape, generator=gen).requires_grad_(True)
    before = p.detach().clone()
    opt = Muon([p], lr=lr, weight_decay=wd, adjust_lr="match_rms_adamw", cautious_wd=False, ns_dtype=torch.float32)
    p.grad = torch.zeros_like(p)
    opt.step()
    ratio = p.detach() / before
    effective_lr = lr * 0.2 * math.sqrt(max(shape))
    assert torch.allclose(ratio, torch.full_like(ratio, 1.0 - effective_lr * wd), atol=1e-6)
    assert not torch.allclose(ratio, torch.full_like(ratio, 1.0 - lr * wd), atol=1e-4)


# --------------------------------------------------------------------------- #
# Smoke: every new mode runs cleanly across shapes / foreach / wd
# --------------------------------------------------------------------------- #

_SMOKE_SHAPES = [(48, 16), (16, 48), (32, 32), (16, 8, 3, 3)]


@pytest.mark.parametrize("mode", ["mup", "interp"])
@pytest.mark.parametrize("foreach", [True, False])
@pytest.mark.parametrize("wd", [0.0, 0.1])
def test_smoke(mode: AdjustLrMode, foreach: bool, wd: float) -> None:
    """A few steps over mixed shapes stay finite for the new modes."""
    gen = torch.Generator().manual_seed(0)
    params = [torch.randn(s, generator=gen).requires_grad_(True) for s in _SMOKE_SHAPES]
    opt = Muon(params, lr=0.05, weight_decay=wd, adjust_lr=mode, tau=0.5, foreach=foreach, ns_dtype=torch.float32)
    for _ in range(3):
        for p in params:
            p.grad = torch.randn(p.shape, generator=gen)
        opt.step()
    for p in params:
        assert torch.isfinite(p).all()
