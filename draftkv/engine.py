"""The draft/verify loop.

Invariant maintained everywhere below:

    caches hold K/V for positions 0 .. len(tokens) - 2
    `pending` is tokens[-1], not yet written to any cache

Keeping the last token pending (rather than caching it and holding its logits)
is what lets the verify pass recompute the drafter's very first step under the
full cache.  Without it, the first drafted token would be trivially accepted
and the acceptance statistic would be biased upward.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .compress import CompressedKVCache, FullKVCache
from .config import CompressionConfig
from .controller import Controller, FixedPolicy


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def _layer_cfgs(cfg, n_layers: int) -> list:
    """Per-layer configs for either a LayerPlan or a single config."""
    return list(cfg.cfgs) if hasattr(cfg, "cfgs") else [cfg] * n_layers


def _probs(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0.0:
        p = np.zeros_like(logits, dtype=np.float64)
        p[int(np.argmax(logits))] = 1.0
        return p
    return _softmax(logits.astype(np.float64) / temperature)


@dataclass
class RoundLog:
    cfg: CompressionConfig
    gamma: int
    accepted: int
    emitted: int
    ctx: int


@dataclass
class GenerationResult:
    tokens: list[int]
    rounds: list[RoundLog] = field(default_factory=list)
    draft_steps: int = 0
    verify_passes: int = 0
    cache_rebuilds: int = 0
    layers_rebuilt: int = 0

    @property
    def new_tokens(self) -> int:
        return sum(r.emitted for r in self.rounds)

    @property
    def acceptance_rate(self) -> float:
        drafted = sum(r.gamma for r in self.rounds)
        return sum(r.accepted for r in self.rounds) / drafted if drafted else float("nan")

    @property
    def tokens_per_verify(self) -> float:
        return self.new_tokens / self.verify_passes if self.verify_passes else float("nan")


class DraftKVEngine:
    """Self-speculative decoding where the drafter is the target model reading
    a compressed copy of its own KV cache.

    Output is identical to full-KV decoding: exactly so under greedy decoding,
    and in distribution under sampling (standard speculative-sampling
    rejection rule).  Compression therefore only moves speed, never quality --
    which is the premise that makes an online controller legitimate.
    """

    def __init__(
        self,
        model,
        controller: Controller | None = None,
        seed: int = 0,
        full_cache_factory=None,
    ) -> None:
        self.model = model
        self.controller = controller or FixedPolicy(CompressionConfig(4, 0.5), 4)
        self.rng = np.random.default_rng(seed)
        # swappable so the authoritative cache can be a paged pool instead of a
        # contiguous array; the rollback contract has to hold for both
        self.full_cache_factory = full_cache_factory or (lambda: FullKVCache(model.n_layers))

    # ------------------------------------------------------------------ util
    def _sample(self, p: np.ndarray) -> int:
        p = np.clip(p, 0.0, None)
        s = p.sum()
        if s <= 0:
            return int(self.rng.integers(len(p)))
        return int(self.rng.choice(len(p), p=p / s))

    # -------------------------------------------------------------- baseline
    def generate_baseline(
        self, prompt: list[int], max_new_tokens: int, temperature: float = 0.0
    ) -> list[int]:
        """Plain full-KV autoregressive decoding -- the reference output."""
        tokens = list(prompt)
        cache = self.full_cache_factory()
        if len(tokens) > 1:
            self.model.forward(np.array(tokens[:-1]), cache, 0)
        for _ in range(max_new_tokens):
            logits = self.model.forward(np.array([tokens[-1]]), cache, len(tokens) - 1)[-1]
            tokens.append(self._sample(_probs(logits, temperature)))
        return tokens

    # --------------------------------------------------------------- draftkv
    def generate(
        self,
        prompt: list[int],
        max_new_tokens: int,
        temperature: float = 0.0,
        context_key: str | None = None,
    ) -> GenerationResult:
        model = self.model
        tokens = list(prompt)
        full = self.full_cache_factory()
        if len(tokens) > 1:
            model.forward(np.array(tokens[:-1]), full, 0)

        comp: CompressedKVCache | None = None
        comp_cfg: CompressionConfig | None = None
        res = GenerationResult(tokens=tokens)

        while len(tokens) - len(prompt) < max_new_tokens:
            n = len(tokens)          # caches cover 0 .. n-2; tokens[-1] is pending
            ctx = n
            budget = max_new_tokens - (n - len(prompt))
            cfg, gamma = self.controller.select(ctx, context_key)
            gamma = min(gamma, max(budget - 1, 0))

            if gamma <= 0:
                logits = model.forward(np.array([tokens[-1]]), full, n - 1)[-1]
                tokens.append(self._sample(_probs(logits, temperature)))
                res.verify_passes += 1
                res.rounds.append(RoundLog(cfg, 0, 0, 1, ctx))
                if comp is not None:
                    comp.sync_from(full, n - 1)
                continue

            # -- drafter's cache: rebuild only the layers that actually changed --
            if comp is None:
                comp = CompressedKVCache(model.n_layers, cfg)
                comp.sync_from(full, 0)     # O(n_layers * ctx), once
                comp_cfg = cfg
                res.cache_rebuilds += 1
            elif comp_cfg != cfg:
                old = _layer_cfgs(comp_cfg, model.n_layers)
                new = _layer_cfgs(cfg, model.n_layers)
                changed = [L for L in range(model.n_layers) if old[L] != new[L]]
                comp.rebuild_layers(full, changed, new)
                comp_cfg = cfg
                res.cache_rebuilds += 1
                res.layers_rebuilt += len(changed)

            # ------------------------------------------------------ draft
            draft_tokens: list[int] = []
            draft_probs: list[np.ndarray] = []
            x = tokens[-1]
            for i in range(gamma):
                logits = model.forward(np.array([x]), comp, n - 1 + i)[-1]
                q = _probs(logits, temperature)
                x = self._sample(q)
                draft_tokens.append(x)
                draft_probs.append(q)
            res.draft_steps += gamma

            # ----------------------------------------------------- verify
            block = np.array([tokens[-1]] + draft_tokens)
            logits = model.forward(block, full, n - 1)   # (gamma+1, V)
            res.verify_passes += 1

            accepted = 0
            for i in range(gamma):
                p = _probs(logits[i], temperature)
                if temperature <= 0.0:
                    ok = int(np.argmax(logits[i])) == draft_tokens[i]
                else:
                    q = draft_probs[i]
                    d = draft_tokens[i]
                    ratio = p[d] / q[d] if q[d] > 0 else 0.0
                    ok = self.rng.random() < min(1.0, ratio)
                if not ok:
                    break
                accepted += 1

            emitted = list(draft_tokens[:accepted])
            if accepted == gamma:                      # all good: free bonus token
                nxt = self._sample(_probs(logits[gamma], temperature))
            else:                                      # rejected: repair the distribution
                p = _probs(logits[accepted], temperature)
                if temperature <= 0.0:
                    nxt = int(np.argmax(logits[accepted]))
                else:
                    resid = np.clip(p - draft_probs[accepted], 0.0, None)
                    nxt = self._sample(resid if resid.sum() > 0 else p)
            emitted.append(nxt)
            tokens.extend(emitted)

            # -------------------------------------------- roll back + resync
            full.truncate(n + accepted)
            comp.sync_from(full, n - 1)

            self.controller.update(cfg, gamma, accepted, ctx, context_key)
            res.rounds.append(RoundLog(cfg, gamma, accepted, len(emitted), ctx))

        res.tokens = tokens
        return res
