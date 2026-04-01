"""optimuon — A performance-optimized Muon optimizer for PyTorch.

Foreach-native Muon with auto-parameter routing, composite optimizer patterns,
and optional corrections (MARS, cautious, clipping, NorMuon, weight normalization, Polar Express).
"""

from ._composite import CompositeMuon
from ._corrections import apply_cautious_mask
from ._corrections import apply_cautious_weight_decay
from ._corrections import apply_mars_correction
from ._corrections import apply_normuon_rescale
from ._corrections import apply_weight_norm
from ._corrections import clip_grad_norm_foreach
from ._corrections import clip_update_norm_foreach
from ._muon import AdjustLrMode
from ._muon import MomentumType
from ._muon import Muon
from ._newton_schulz import GNS_DTYPE_DEFAULT
from ._newton_schulz import GNS_RESTART_AFTER_DEFAULT
from ._newton_schulz import gram_newton_schulz
from ._newton_schulz import gram_newton_schulz_batched
from ._newton_schulz import newton_schulz
from ._newton_schulz import newton_schulz_batched
from ._newton_schulz import NS_COEFFICIENTS_DEFAULT
from ._newton_schulz import NS_COEFFICIENTS_POLAR_EXPRESS
from ._newton_schulz import NS_DTYPE_DEFAULT
from ._newton_schulz import NS_EPS_DEFAULT
from ._newton_schulz import NS_NORM_SCALE_DEFAULT
from ._newton_schulz import NS_NORM_SCALE_POLAR_EXPRESS
from ._newton_schulz import NS_STEPS_DEFAULT
from ._newton_schulz import NSCoefficients
from ._routing import is_muon_eligible
from ._routing import partition_params
from ._routing import PartitionResult

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
    "AdjustLrMode",
    "CompositeMuon",
    "MomentumType",
    "Muon",
    "NSCoefficients",
    "PartitionResult",
    "apply_cautious_mask",
    "apply_cautious_weight_decay",
    "apply_mars_correction",
    "apply_normuon_rescale",
    "apply_weight_norm",
    "clip_grad_norm_foreach",
    "clip_update_norm_foreach",
    "gram_newton_schulz",
    "gram_newton_schulz_batched",
    "is_muon_eligible",
    "newton_schulz",
    "newton_schulz_batched",
    "partition_params",
]

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"
