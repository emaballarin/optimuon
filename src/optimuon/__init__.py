"""optimuon — A performance-optimized Muon optimizer for PyTorch.

Foreach-native Muon with auto-parameter routing, composite optimizer patterns,
and optional corrections (MARS, cautious, clipping).
"""

from ._composite import CompositeMuon
from ._corrections import apply_cautious_mask
from ._corrections import apply_mars_correction
from ._corrections import clip_grad_norm_foreach
from ._corrections import clip_update_norm_foreach
from ._muon import AdjustLrMode
from ._muon import MomentumType
from ._muon import Muon
from ._newton_schulz import newton_schulz
from ._newton_schulz import newton_schulz_batched
from ._newton_schulz import NS_COEFFICIENTS_DEFAULT
from ._newton_schulz import NS_EPS_DEFAULT
from ._newton_schulz import NS_STEPS_DEFAULT
from ._routing import is_muon_eligible
from ._routing import partition_params
from ._routing import PartitionResult

__all__ = [
    "NS_COEFFICIENTS_DEFAULT",
    "NS_EPS_DEFAULT",
    "NS_STEPS_DEFAULT",
    "AdjustLrMode",
    "CompositeMuon",
    "MomentumType",
    "Muon",
    "PartitionResult",
    "apply_cautious_mask",
    "apply_mars_correction",
    "clip_grad_norm_foreach",
    "clip_update_norm_foreach",
    "is_muon_eligible",
    "newton_schulz",
    "newton_schulz_batched",
    "partition_params",
]

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"
