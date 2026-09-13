"""DRAFTKV -- KV-cache compression as a pure speed knob.

Compress the cache, draft with it, verify against the full cache.  Output is
identical to full-KV decoding, so the compression setting no longer trades
against quality; it only trades against speed, and can therefore be tuned
online from the acceptance rate the verify pass hands you for free.
"""

from .compress import CompressedKVCache, FullKVCache, quantize
from .config import CompressionConfig, LayerPlan, default_arms
from .controller import (
    DepthThompsonController,
    FixedPolicy,
    FlatBandit,
    OraclePolicy,
    ThompsonController,
)
from .engine import DraftKVEngine, GenerationResult
from .model import TinyTransformer
from .layers import SensitivityProfile, allocate, measure_alpha, profile_layers
from .throughput import (
    CostModel,
    best_gamma,
    best_gamma_profile,
    expected_tokens,
    expected_tokens_profile,
)

__all__ = [
    "CompressedKVCache",
    "CompressionConfig",
    "CostModel",
    "DepthThompsonController",
    "DraftKVEngine",
    "FixedPolicy",
    "FlatBandit",
    "FullKVCache",
    "GenerationResult",
    "LayerPlan",
    "OraclePolicy",
    "SensitivityProfile",
    "ThompsonController",
    "TinyTransformer",
    "allocate",
    "best_gamma",
    "best_gamma_profile",
    "default_arms",
    "expected_tokens",
    "expected_tokens_profile",
    "measure_alpha",
    "profile_layers",
    "quantize",
]
__version__ = "0.1.0"
