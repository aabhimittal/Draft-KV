"""Compression configuration and the arm space the controller searches over."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence


@dataclass(frozen=True, order=True)
class CompressionConfig:
    """How aggressively the *drafter's* view of the KV cache is compressed.

    Because DRAFTKV verifies every drafted token against the full-precision
    cache, none of these knobs can change the emitted text.  They only change
    how often the drafter guesses right (acceptance rate) and how fast a draft
    step is (bytes moved).

    bits:      quantization width for K and V entries (16 == no quantization).
    keep_frac: fraction of context tokens the drafter is allowed to keep.
    sink:      number of leading tokens always kept (attention sinks).
    recent:    number of trailing tokens always kept (local window).
    """

    bits: int = 16
    keep_frac: float = 1.0
    sink: int = 4
    recent: int = 32

    def __post_init__(self) -> None:
        if self.bits not in (2, 3, 4, 8, 16):
            raise ValueError(f"unsupported bits={self.bits}")
        if not 0.0 < self.keep_frac <= 1.0:
            raise ValueError(f"keep_frac must be in (0, 1], got {self.keep_frac}")

    @property
    def lossless(self) -> bool:
        return self.bits == 16 and self.keep_frac >= 1.0

    def bytes_per_entry(self, d_head: int, group: int = 64) -> float:
        """Bytes to store one (token, head) key or value vector.

        Quantized storage is `bits` per element plus one fp16 scale and one
        fp16 zero-point per group of `group` elements.
        """
        if self.bits == 16:
            return 2.0 * d_head
        groups = max(1, -(-d_head // group))
        return d_head * self.bits / 8.0 + 4.0 * groups

    def label(self) -> str:
        b = "fp16" if self.bits == 16 else f"{self.bits}b"
        return f"{b}/keep{self.keep_frac:g}"


def default_arms(
    bits: Sequence[int] = (2, 3, 4, 8, 16),
    keep_fracs: Sequence[float] = (0.25, 0.5, 1.0),
    sink: int = 4,
    recent: int = 32,
) -> list[CompressionConfig]:
    """Cartesian arm space, deduplicated (fp16/keep1.0 is the only lossless arm)."""
    seen: dict[tuple, CompressionConfig] = {}
    for b in bits:
        for k in keep_fracs:
            c = CompressionConfig(bits=b, keep_frac=k, sink=sink, recent=recent)
            seen.setdefault((b, k), c)
    return sorted(seen.values())


def gamma_range(gamma_max: int = 8) -> Iterator[int]:
    return iter(range(1, gamma_max + 1))


@dataclass(frozen=True)
class LayerPlan:
    """A per-layer compression assignment.

    One global config is a strong assumption: measurement on the reference
    model shows identical 2-bit damage costs 0.62 acceptance in one layer and
    0.33 in another.  Spending the same bits everywhere therefore overpays in
    tolerant layers and starves sensitive ones.  A plan lets the byte budget be
    distributed where it buys the most acceptance.
    """

    cfgs: tuple[CompressionConfig, ...]

    def __len__(self) -> int:
        return len(self.cfgs)

    def __getitem__(self, i: int) -> CompressionConfig:
        return self.cfgs[i]

    def __iter__(self):
        return iter(self.cfgs)

    def per_layer_bytes(self, ctx: int, n_kv_heads: int, d_head: int) -> float:
        """Exact K+V bytes for the whole plan -- summed per layer, not averaged.

        Averaging `keep_frac` and `bytes_per_entry` separately would misprice a
        mixed plan, since it is their product that matters per layer.
        """
        total = 0.0
        for c in self.cfgs:
            kept = min(ctx, max(c.sink + c.recent, int(round(c.keep_frac * ctx))))
            total += 2.0 * kept * n_kv_heads * c.bytes_per_entry(d_head)
        return total

    def label(self) -> str:
        uniq = sorted({c.label() for c in self.cfgs})
        if len(uniq) == 1:
            return f"plan[{uniq[0]}]"
        counts: dict[str, int] = {}
        for c in self.cfgs:
            counts[c.label()] = counts.get(c.label(), 0) + 1
        return "plan[" + ",".join(f"{k}x{v}" for k, v in sorted(counts.items())) + "]"

    @classmethod
    def uniform(cls, cfg: CompressionConfig, n_layers: int) -> "LayerPlan":
        return cls(tuple(cfg for _ in range(n_layers)))
