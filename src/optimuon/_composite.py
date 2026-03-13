"""Composite optimizer combining Muon with an arbitrary auxiliary optimizer.

Generalizes KellerJordan's ``MuonWithAuxAdam``: a single optimizer object
that routes ≥2D hidden weight matrices to Muon and everything else to
a user-specified auxiliary (e.g. AdamW, Lion, Prodigy, …).

Two API styles for the auxiliary optimizer:

1. **Factory callable** (most flexible)::

       CompositeMuon(model, aux_optimizer_factory=lambda pg: AdamW(pg, lr=3e-4))

2. **Class + kwargs** (convenience)::

       CompositeMuon(model, aux_optimizer_class=AdamW, aux_optimizer_kwargs={"lr": 3e-4})
"""

import warnings
from collections.abc import Callable
from collections.abc import Iterator
from typing import Any

import torch
from torch import nn
from torch import Tensor
from torch.optim.optimizer import Optimizer

from ._muon import Muon
from ._routing import partition_params
from ._routing import PartitionResult

__all__ = ["CompositeMuon"]


class CompositeMuon:
    """Composite optimizer: Muon for eligible parameters, arbitrary aux for the rest.

    Not a subclass of ``torch.optim.Optimizer``—it coordinates a :class:`Muon`
    and an auxiliary optimizer behind a unified interface.

    Args:
        named_params: An ``nn.Module`` or iterator of ``(name, param)`` pairs.
        muon_lr: Learning rate for Muon.
        muon_kwargs: Extra kwargs forwarded to :class:`Muon`.
        aux_optimizer_factory: ``(list[dict]) -> Optimizer``. Mutually exclusive
            with ``aux_optimizer_class``.
        aux_optimizer_class: Optimizer class for auxiliary parameters.
        aux_optimizer_kwargs: Kwargs for the auxiliary optimizer class.
        exclude_patterns: Regex patterns to exclude from Muon routing.
        auto_route: Auto-partition parameters. If False, provide ``muon_params``
            and ``aux_params`` explicitly.
        muon_params: Explicit Muon parameters (when ``auto_route=False``).
        aux_params: Explicit aux parameters or param groups (when ``auto_route=False``).
        verbose: Print routing decisions.
    """

    muon: Muon
    """The internal Muon optimizer."""
    aux: Optimizer
    """The internal auxiliary optimizer."""
    partition: PartitionResult | None
    """Partition result from auto-routing, if used."""

    def __init__(
        self,
        named_params: nn.Module | Iterator[tuple[str, Tensor]] | None = None,
        *,
        muon_lr: float = 0.02,
        muon_kwargs: dict[str, Any] | None = None,
        aux_optimizer_factory: Callable[[list[dict[str, Any]]], Optimizer] | None = None,
        aux_optimizer_class: type[Optimizer] | None = None,
        aux_optimizer_kwargs: dict[str, Any] | None = None,
        exclude_patterns: tuple[str, ...] | None = None,
        auto_route: bool = True,
        muon_params: list[Tensor] | None = None,
        aux_params: list[Tensor] | list[dict[str, Any]] | None = None,
        verbose: bool = False,
    ) -> None:
        if aux_optimizer_factory is not None and aux_optimizer_class is not None:
            raise ValueError("Specify either aux_optimizer_factory or aux_optimizer_class, not both.")
        if aux_optimizer_factory is None and aux_optimizer_class is None:
            aux_optimizer_class = torch.optim.AdamW
            aux_optimizer_kwargs = aux_optimizer_kwargs or {"lr": 3e-4}

        muon_kwargs = muon_kwargs or {}

        # --- Partition ---
        if auto_route:
            if named_params is None:
                raise ValueError("named_params required when auto_route=True.")
            routing_kw: dict[str, Any] = {}
            if exclude_patterns is not None:
                routing_kw["exclude_patterns"] = exclude_patterns
            self.partition = partition_params(named_params, **routing_kw)
            _muon_params = self.partition.muon_params
            _aux_params: list[Tensor] | list[dict[str, Any]] = self.partition.aux_params
            if verbose:
                print(f"[CompositeMuon] {len(self.partition.muon_names)} → Muon, {len(self.partition.aux_names)} → aux")
                for n in self.partition.muon_names:
                    print(f"  [Muon] {n}")
                for n in self.partition.aux_names:
                    print(f"  [Aux]  {n}")
        else:
            self.partition = None
            if muon_params is None or aux_params is None:
                raise ValueError("muon_params and aux_params required when auto_route=False.")
            _muon_params = muon_params
            _aux_params = aux_params

        # --- Build Muon ---
        if _muon_params:
            self.muon = Muon(_muon_params, lr=muon_lr, **muon_kwargs)
        else:
            warnings.warn("No parameters routed to Muon. All will use the auxiliary optimizer.", stacklevel=2)
            # Muon.__init__ validates that all params are >=2D, so we cannot pass an empty
            # list directly. Create a throwaway 2x2 tensor to satisfy the constructor,
            # then immediately clear the param group so no dummy updates occur.
            self.muon = Muon([torch.zeros(2, 2, requires_grad=True)], lr=muon_lr, **muon_kwargs)
            self.muon.param_groups[0]["params"] = []

        # --- Build auxiliary ---
        if isinstance(_aux_params, list) and _aux_params and isinstance(_aux_params[0], dict):
            aux_groups: list[dict[str, Any]] = _aux_params  # type: ignore[assignment]
        else:
            aux_groups = [{"params": _aux_params}]

        if aux_optimizer_factory is not None:
            self.aux = aux_optimizer_factory(aux_groups)
        else:
            assert aux_optimizer_class is not None
            self.aux = aux_optimizer_class(aux_groups, **(aux_optimizer_kwargs or {}))

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Step both optimizers."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.muon.step()
        self.aux.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients for all managed parameters."""
        self.muon.zero_grad(set_to_none=set_to_none)
        self.aux.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        """Serialize both optimizer states."""
        return {"muon": self.muon.state_dict(), "aux": self.aux.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load both optimizer states."""
        self.muon.load_state_dict(state_dict["muon"])
        self.aux.load_state_dict(state_dict["aux"])

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        """Combined param groups (Muon groups first, then aux)."""
        return self.muon.param_groups + self.aux.param_groups

    def __repr__(self) -> str:
        mn = sum(len(g["params"]) for g in self.muon.param_groups)
        an = sum(len(g["params"]) for g in self.aux.param_groups)
        return f"CompositeMuon(\n  muon=Muon(params={mn}),\n  aux={type(self.aux).__name__}(params={an}),\n)"
