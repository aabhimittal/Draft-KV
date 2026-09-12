"""DRAFTKV -- KV-cache compression as a pure speed knob.

Compress the cache, draft with it, verify against the full cache.  Output is
identical to full-KV decoding, so the compression setting no longer trades
against quality; it only trades against speed, and can therefore be tuned
online from the acceptance rate the verify pass hands you for free.
"""

from .compress import CompressedKVCache, FullKVCache, quantize
from .config import CompressionConfig, default_arms
from .controller import FixedPolicy, FlatBandit, OraclePolicy, ThompsonController
from .engine import DraftKVEngine, GenerationResult
from .model import TinyTransformer
from .throughput import CostModel, best_gamma, expected_tokens

__all__ = [
    "CompressedKVCache",
    "CompressionConfig",
    "CostModel",
    "DraftKVEngine",
    "FixedPolicy",
    "FlatBandit",
    "FullKVCache",
    "GenerationResult",
    "OraclePolicy",
    "ThompsonController",
    "TinyTransformer",
    "best_gamma",
    "default_arms",
    "expected_tokens",
    "quantize",
]
__version__ = "0.1.0"
