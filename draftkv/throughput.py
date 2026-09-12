"""The arithmetic DRAFTKV optimizes.

Two pieces: `expected_tokens` (how many tokens a verify pass yields) and
`CostModel` (what a draft step and a verify pass actually cost).

The cost model is deliberately built from memory traffic rather than from
fudge factors, because that is what exposes the ceiling.  Both drafting and
verification run the *same* model, so both stream the full weights.  Over one
round, weight traffic is (gamma + 1) * W either way -- identical to plain
decoding.  Every byte DRAFTKV saves is therefore a KV byte.  The speedup is
bounded by `CostModel.ceiling()`, and no acceptance rate can exceed it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import CompressionConfig


def expected_tokens(alpha: float, gamma: int) -> float:
    """E(alpha, gamma) = (1 - alpha^(gamma+1)) / (1 - alpha).

    Tokens emitted per verify pass when each drafted token is independently
    accepted with probability `alpha`: the accepted run, plus one bonus token
    the verify pass produces for free.  Bounded above by gamma + 1, and
    saturating at 1 / (1 - alpha) as gamma grows -- which is why gamma has a
    finite optimum even when drafting is nearly free.
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if gamma < 0:
        raise ValueError("gamma must be >= 0")
    if alpha >= 1.0 - 1e-12:
        return float(gamma + 1)
    return float((1.0 - alpha ** (gamma + 1)) / (1.0 - alpha))


def expected_tokens_profile(accept: Sequence[float]) -> float:
    """E for a depth-indexed acceptance profile: a_i = P(accept at depth i | reached i).

    E = 1 + sum_k prod_{i<k} a_i, over k = 1..len(accept).

    Why this exists: `expected_tokens` assumes one constant alpha, but measured
    profiles are not flat -- acceptance *rises* with depth, because reaching
    depth i is itself evidence that the current region is easy to draft.  The
    i.i.d. model therefore understates E and picks gamma too small.  With a
    constant profile this reduces exactly to `expected_tokens`.
    """
    total, run = 1.0, 1.0
    for a in accept:
        run *= float(np.clip(a, 0.0, 1.0))
        total += run
    return total


@dataclass
class CostModel:
    """Wall-clock model of batch-1 decoding, parameterized by memory traffic.

    Defaults describe a Llama-3-8B-shaped model (GQA, 8 KV heads) on an
    H100-class device with the full cache resident in HBM.  Set
    `full_kv_bandwidth` to a PCIe figure to model an offloaded full cache --
    which is the regime where this technique actually pays.
    """

    n_layers: int = 32
    n_kv_heads: int = 8
    d_head: int = 128
    weight_bytes: float = 16.0e9        # streamed once per forward pass, draft or verify
    bandwidth: float = 2.0e12           # HBM bytes/s (weights, compressed cache)
    full_kv_bandwidth: float | None = None  # where the full cache lives; None -> HBM
    batch: int = 1                      # KV traffic scales with batch, weight traffic does not
    hbm_kv_budget: float | None = None  # bytes of HBM the drafter's cache may occupy; None = unlimited
    compute_per_token: float = 1.0e-6   # FLOP-bound term, small at batch 1
    fixed_draft: float = 5.0e-6         # launch / sampling overhead per draft step
    fixed_verify: float = 1.0e-5        # launch overhead of a verify pass
    resync_per_token: float = 2.0e-6    # rewriting one accepted position into the drafter's cache
    controller_overhead: float = 1.0e-6  # per decision

    def __post_init__(self) -> None:
        if self.full_kv_bandwidth is None:
            self.full_kv_bandwidth = self.bandwidth

    # ---------------------------------------------------------------- bytes
    def full_kv_bytes(self, ctx: int) -> float:
        """K and V, fp16, all layers, whole batch."""
        return 2.0 * ctx * self.n_layers * self.n_kv_heads * self.d_head * 2.0 * self.batch

    def draft_kv_bytes(self, cfg, ctx: int) -> float:
        """Bytes the drafter streams per token. Accepts a config or a LayerPlan."""
        if hasattr(cfg, "per_layer_bytes"):
            return cfg.per_layer_bytes(ctx, self.n_kv_heads, self.d_head) * self.batch
        kept = min(ctx, max(cfg.sink + cfg.recent, int(round(cfg.keep_frac * ctx))))
        per = cfg.bytes_per_entry(self.d_head)
        return 2.0 * kept * self.n_layers * self.n_kv_heads * per * self.batch

    # ---------------------------------------------------------------- times
    def _kv_read_time(self, nbytes: float) -> float:
        """Time to stream the drafter's cache, honoring the HBM budget.

        This is where the memory/bandwidth trade the whole scheme rests on gets
        priced.  Whatever fits in `hbm_kv_budget` is read at HBM speed; the
        overflow is read wherever the full cache lives (host or remote), at
        that link's bandwidth.  Without this term an "fp16, keep everything"
        drafter looks free, and the controller correctly concludes that
        compression is pointless -- correctly, for a machine that does not
        exist.
        """
        if self.hbm_kv_budget is None:
            return nbytes / self.bandwidth
        resident = min(nbytes, self.hbm_kv_budget)
        spilled = max(0.0, nbytes - self.hbm_kv_budget)
        return resident / self.bandwidth + spilled / self.full_kv_bandwidth

    def draft_kv_resident_frac(self, cfg: CompressionConfig, ctx: int) -> float:
        n = self.draft_kv_bytes(cfg, ctx)
        if self.hbm_kv_budget is None or n <= 0:
            return 1.0
        return min(1.0, self.hbm_kv_budget / n)

    def t_draft(self, cfg: CompressionConfig, ctx: int) -> float:
        """One drafted token: full weights + the compressed cache."""
        return (
            self.weight_bytes / self.bandwidth
            + self._kv_read_time(self.draft_kv_bytes(cfg, ctx))
            + self.compute_per_token
            + self.fixed_draft
        )

    def t_verify(self, ctx: int, gamma: int) -> float:
        """One verify pass over gamma + 1 positions.

        The full cache is read *once* for the whole block (a single batched
        attention over gamma + 1 queries), which is the second, often
        overlooked, source of savings.
        """
        return (
            self.weight_bytes / self.bandwidth
            + self.full_kv_bytes(ctx) / self.full_kv_bandwidth
            + (gamma + 1) * self.compute_per_token
            + self.fixed_verify
        )

    def t_baseline(self, ctx: int) -> float:
        """Per-token cost of plain full-KV autoregressive decoding."""
        return (
            self.weight_bytes / self.bandwidth
            + self.full_kv_bytes(ctx) / self.full_kv_bandwidth
            + self.compute_per_token
            + self.fixed_verify
        )

    # ----------------------------------------------------------- objectives
    def rebuild_cost(self, ctx: int) -> float:
        """Cost of re-deriving the drafter's whole cache under a new config.

        Switching compression config is not free: every cached position has to
        be re-quantized, so it is O(ctx). A controller that ignores this will
        thrash between near-equal arms and lose more than it gains.
        """
        return ctx * self.resync_per_token

    def round_cost_profile(
        self, cfg: CompressionConfig, accept: Sequence[float], ctx: int, extra_cost: float = 0.0
    ) -> float:
        gamma = len(accept)
        return (
            gamma * self.t_draft(cfg, ctx)
            + self.t_verify(ctx, gamma)
            + expected_tokens_profile(accept) * self.resync_per_token
            + self.controller_overhead
            + extra_cost
        )

    def throughput_profile(
        self, cfg: CompressionConfig, accept: Sequence[float], ctx: int, extra_cost: float = 0.0
    ) -> float:
        """Throughput under a depth-indexed acceptance profile."""
        if not accept:
            return 1.0 / self.t_baseline(ctx)
        return expected_tokens_profile(accept) / self.round_cost_profile(cfg, accept, ctx, extra_cost)

    def round_cost(
        self, cfg: CompressionConfig, gamma: int, alpha: float, ctx: int, extra_cost: float = 0.0
    ) -> float:
        return (
            gamma * self.t_draft(cfg, ctx)
            + self.t_verify(ctx, gamma)
            + expected_tokens(alpha, gamma) * self.resync_per_token
            + self.controller_overhead
            + extra_cost
        )

    def throughput(
        self, cfg: CompressionConfig, gamma: int, alpha: float, ctx: int, extra_cost: float = 0.0
    ) -> float:
        """T(c, gamma) = E(alpha(c), gamma) / (gamma * t_draft(c) + t_verify).

        `extra_cost` carries per-round charges that are not intrinsic to the
        arm -- currently the amortized cost of switching to it.
        """
        if gamma == 0:
            return 1.0 / self.t_baseline(ctx)
        return expected_tokens(alpha, gamma) / self.round_cost(cfg, gamma, alpha, ctx, extra_cost)

    def speedup(self, cfg: CompressionConfig, gamma: int, alpha: float, ctx: int) -> float:
        return self.throughput(cfg, gamma, alpha, ctx) * self.t_baseline(ctx)

    def ceiling(self, cfg: CompressionConfig, ctx: int) -> float:
        """Speedup as alpha -> 1 and gamma -> infinity: the ratio of per-token
        memory traffic between a baseline token and a drafted token.

        Nothing in the controller can exceed this.  If it is close to 1.0 for
        your context length, DRAFTKV cannot help and should stay off.
        """
        base = self.weight_bytes / self.bandwidth + self.full_kv_bytes(ctx) / self.full_kv_bandwidth
        draft = self.weight_bytes / self.bandwidth + self._kv_read_time(self.draft_kv_bytes(cfg, ctx))
        return (base + self.compute_per_token + self.fixed_verify) / (
            draft + self.compute_per_token + self.fixed_draft + self.resync_per_token
        )


def best_gamma_profile(
    cost: CostModel,
    cfg: CompressionConfig,
    accept: Sequence[float],
    ctx: int,
    extra_cost: float = 0.0,
) -> tuple[int, float]:
    """Argmax over gamma given a depth profile; gamma indexes into `accept`.

    Still a scan, but no longer unimodal-by-construction: a profile that rises
    with depth can make a longer draft pay off after a shorter one did not, so
    the whole range is evaluated rather than stopping at the first decline.
    """
    best, best_t = 0, cost.throughput_profile(cfg, (), ctx)
    for g in range(1, len(accept) + 1):
        t = cost.throughput_profile(cfg, accept[:g], ctx, extra_cost)
        if t > best_t:
            best, best_t = g, t
    return best, best_t


def best_gamma(
    cost: CostModel,
    cfg: CompressionConfig,
    alpha: float,
    ctx: int,
    gamma_max: int = 8,
    extra_cost: float = 0.0,
) -> tuple[int, float]:
    """Argmax over gamma of T(c, gamma), with gamma = 0 meaning "don't draft".

    T is unimodal in gamma -- the numerator saturates at 1/(1-alpha) while the
    denominator grows linearly -- so a scan over a small range is both exact
    and cheaper than any solver.
    """
    best, best_t = 0, cost.throughput(cfg, 0, alpha, ctx)
    for g in range(1, gamma_max + 1):
        t = cost.throughput(cfg, g, alpha, ctx, extra_cost)
        if t > best_t:
            best, best_t = g, t
    return best, best_t
