"""Auto-parameter routing for composite Muon optimizers.

Partitions model parameters into Muon-eligible (≥2D hidden weights) and
auxiliary (embeddings, heads, norms, biases, scalars).
"""

import re
from collections.abc import Iterator
from typing import NamedTuple

from torch import nn
from torch import Tensor

__all__ = [
    "PartitionResult",
    "is_muon_eligible",
    "partition_params",
]

# Patterns are matched as substrings (via re.search) against lowercased param names.
# E.g. r"norm" catches "layernorm", "rmsnorm", etc.; r"cls" catches "classifier".
_DEFAULT_EXCLUDE_PATTERNS: tuple[str, ...] = (
    r"embed",
    r"(?:^|[._])head(?:$|[._])",  # word-boundary: matches lm_head, model.head, NOT overhead/header
    r"cls",
    r"norm",
    r"ln_",
    r"bias",
)


class PartitionResult(NamedTuple):
    """Result of partitioning model parameters."""

    muon_params: list[Tensor]
    """Parameters eligible for Muon (≥2D hidden weight matrices)."""
    aux_params: list[Tensor]
    """Parameters to be optimized by the auxiliary optimizer."""
    muon_names: list[str]
    """Names of Muon-eligible parameters."""
    aux_names: list[str]
    """Names of auxiliary parameters."""


def is_muon_eligible(
    name: str,
    param: Tensor,
    exclude_patterns: tuple[str, ...] = _DEFAULT_EXCLUDE_PATTERNS,
    min_ndim: int = 2,
) -> bool:
    """Check whether a named parameter is eligible for Muon.

    Eligible if: requires_grad, ndim >= min_ndim, name does not match
    any exclude pattern (case-insensitive).
    """
    if not param.requires_grad:
        return False
    if param.ndim < min_ndim:
        return False
    name_lower = name.lower()
    return not any(re.search(pat, name_lower) for pat in exclude_patterns)


def partition_params(
    named_params: Iterator[tuple[str, Tensor]] | nn.Module,
    exclude_patterns: tuple[str, ...] = _DEFAULT_EXCLUDE_PATTERNS,
    min_ndim: int = 2,
    include_frozen: bool = False,
) -> PartitionResult:
    """Partition model parameters into Muon-eligible and auxiliary groups.

    Args:
        named_params: A module or iterator of (name, param) pairs.
        exclude_patterns: Regex patterns to exclude from Muon.
        min_ndim: Minimum ndim for Muon eligibility.
        include_frozen: Include non-grad params in the auxiliary group.

    Returns:
        PartitionResult with separate lists for each group.
    """
    if isinstance(named_params, nn.Module):
        named_params = named_params.named_parameters()

    muon_params: list[Tensor] = []
    muon_names: list[str] = []
    aux_params: list[Tensor] = []
    aux_names: list[str] = []

    for name, param in named_params:
        if not param.requires_grad and not include_frozen:
            continue
        if is_muon_eligible(name, param, exclude_patterns, min_ndim):
            muon_params.append(param)
            muon_names.append(name)
        elif param.requires_grad or include_frozen:
            aux_params.append(param)
            aux_names.append(name)

    return PartitionResult(muon_params, aux_params, muon_names, aux_names)
