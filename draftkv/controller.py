"""Online selection of (compression config, draft length).

The feedback signal is free and needs no labels: every verify pass reports how
many of its gamma drafted tokens survived.  That is a truncated-geometric
observation -- `accepted` successes, plus one failure iff the run was cut
short -- which is conjugate to a Beta prior on the per-token acceptance rate.

Design choice worth stating: alpha is modeled as a property of the compression
config alone, *not* of (config, gamma).  Under the i.i.d. acceptance model that
E(alpha, gamma) already assumes, gamma cannot change alpha -- it only changes
how much of the geometric tail you observe.  So gamma is solved analytically
from the sampled alpha instead of being explored, which shrinks the arm space
from |C| x |gamma| to |C| and makes the bandit converge several times faster.
`FlatBandit` keeps the naive product formulation as a baseline to measure that
claim against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from .config import CompressionConfig, default_arms
from .throughput import CostModel, best_gamma, best_gamma_profile, expected_tokens


class Controller(Protocol):
    def select(self, ctx: int, key: str | None = None) -> tuple[CompressionConfig, int]: ...
    def update(
        self, cfg: CompressionConfig, gamma: int, accepted: int, ctx: int, key: str | None = None
    ) -> None: ...


@dataclass
class FixedPolicy:
    """Static config and draft length -- what you ship without a controller."""

    cfg: CompressionConfig
    gamma: int

    def select(self, ctx: int, key: str | None = None) -> tuple[CompressionConfig, int]:
        return self.cfg, self.gamma

    def update(self, cfg, gamma, accepted, ctx, key=None) -> None:  # noqa: D102
        pass


@dataclass
class _Beta:
    a: float = 1.0
    b: float = 1.0

    def observe(self, accepted: int, gamma: int, decay: float) -> None:
        # decay toward the uniform prior so the posterior tracks content shifts
        self.a = 1.0 + (self.a - 1.0) * decay + accepted
        self.b = 1.0 + (self.b - 1.0) * decay + (1.0 if accepted < gamma else 0.0)

    @property
    def mean(self) -> float:
        return self.a / (self.a + self.b)

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.beta(self.a, self.b))


class ThompsonController:
    """Thompson sampling over compression configs, with gamma solved in closed
    form from the sampled acceptance rate.

    Per decision: sample alpha~ from each arm's posterior, ask the cost model
    for that arm's best gamma and throughput, take the argmax.  gamma = 0 is
    always in the running, so the controller gates itself off automatically
    whenever drafting cannot pay -- short contexts, mostly.
    """

    def __init__(
        self,
        cost: CostModel,
        arms: Sequence[CompressionConfig] | None = None,
        gamma_max: int = 8,
        decay: float = 0.995,
        min_speedup: float = 1.05,
        switch_amortize: int = 32,
        commit_rounds: int = 1,
        seed: int = 0,
        prior: tuple[float, float] = (2.0, 1.0),
    ) -> None:
        self.cost = cost
        self.arms = list(arms or default_arms())
        self.gamma_max = gamma_max
        self.decay = decay
        self.min_speedup = min_speedup
        self.switch_amortize = max(1, switch_amortize)
        self.commit_rounds = max(1, commit_rounds)
        self._held: CompressionConfig | None = None
        self._since_switch = 0
        self.switches = 0
        self.rng = np.random.default_rng(seed)
        self.prior = prior
        self.post: dict[tuple[str | None, CompressionConfig], _Beta] = {}
        self.n_decisions = 0

    def _beta(self, cfg: CompressionConfig, key: str | None) -> _Beta:
        return self.post.setdefault((key, cfg), _Beta(*self.prior))

    # ------------------------------------------------------------------ api
    def select(self, ctx: int, key: str | None = None) -> tuple[CompressionConfig, int]:
        """Pick a config and a draft length.

        gamma is re-solved every round -- it is just a loop bound, free to
        change. A challenger config, by contrast, is charged the rebuild cost
        amortized over `switch_amortize` rounds, so near-ties do not cause
        thrashing while a genuinely better arm still wins immediately.

        `commit_rounds` (a hard hold) is available but defaults to off.
        Measured on the simulator it usually loses: Thompson sampling already
        stops switching once its posteriors concentrate (~6% of rounds), so a
        hard hold buys little and costs a lot of learning speed. It earns its
        keep only when rebuild cost is large relative to a round -- very long
        contexts, or a cache that must be re-fetched over a slow link.
        """
        self.n_decisions += 1
        self._since_switch += 1
        base = 1.0 / self.cost.t_baseline(ctx)

        held_t = -1.0
        if self._held is not None:
            alpha = self._beta(self._held, key).sample(self.rng)
            held_g, held_t = best_gamma(self.cost, self._held, alpha, ctx, self.gamma_max)
            if self._since_switch < self.commit_rounds:
                if held_g > 0 and held_t >= base * self.min_speedup:
                    return self._held, held_g
                if held_g == 0:
                    return self._held, 0

        switch_penalty = self.cost.rebuild_cost(ctx) / self.switch_amortize
        best_cfg, best_g, best_t = self.arms[0], 0, base
        for cfg in self.arms:
            alpha = self._beta(cfg, key).sample(self.rng)
            extra = 0.0 if cfg == self._held else switch_penalty
            g, t = best_gamma(self.cost, cfg, alpha, ctx, self.gamma_max, extra)
            if g > 0 and t > best_t:
                best_cfg, best_g, best_t = cfg, g, t
        if best_g > 0 and best_cfg != self._held:
            self._held, self._since_switch = best_cfg, 0
            self.switches += 1
        if best_g > 0 and best_t < base * self.min_speedup:
            # The predicted win is inside the noise of our own alpha estimate,
            # and a wrong draft costs real time. Decline to draft.
            best_g = 0
        return best_cfg, best_g

    def update(self, cfg, gamma, accepted, ctx, key=None) -> None:
        if gamma > 0:
            self._beta(cfg, key).observe(accepted, gamma, self.decay)

    # ---------------------------------------------------------- diagnostics
    def alpha_estimates(self, key: str | None = None) -> dict[CompressionConfig, float]:
        return {c: self._beta(c, key).mean for c in self.arms}

    def best_known(self, ctx: int, key: str | None = None) -> tuple[CompressionConfig, int, float]:
        rows = []
        for cfg in self.arms:
            a = self._beta(cfg, key).mean
            g, t = best_gamma(self.cost, cfg, a, ctx, self.gamma_max)
            rows.append((t, cfg, g))
        t, cfg, g = max(rows)
        return cfg, g, t


class FlatBandit:
    """Baseline: treat every (config, gamma) pair as an independent arm.

    This is the formulation the problem statement suggests literally.  It works,
    but it has |C| x |gamma| arms, learns nothing about gamma=7 from pulls of
    gamma=3, and needs far more rounds to converge.  Kept so the comparison is
    measurable rather than asserted.
    """

    def __init__(
        self,
        cost: CostModel,
        arms: Sequence[CompressionConfig] | None = None,
        gamma_max: int = 8,
        decay: float = 0.995,
        seed: int = 0,
    ) -> None:
        self.cost = cost
        self.cfgs = list(arms or default_arms())
        self.gamma_max = gamma_max
        self.decay = decay
        self.rng = np.random.default_rng(seed)
        self.post: dict[tuple[CompressionConfig, int], _Beta] = {
            (c, g): _Beta(2.0, 1.0) for c in self.cfgs for g in range(1, gamma_max + 1)
        }

    def select(self, ctx: int, key: str | None = None) -> tuple[CompressionConfig, int]:
        best, best_t = (self.cfgs[0], 1), -1.0
        for (cfg, g), post in self.post.items():
            alpha = post.sample(self.rng)
            t = self.cost.throughput(cfg, g, alpha, ctx)
            if t > best_t:
                best, best_t = (cfg, g), t
        return best

    def update(self, cfg, gamma, accepted, ctx, key=None) -> None:
        if gamma > 0:
            self.post[(cfg, gamma)].observe(accepted, gamma, self.decay)


class OraclePolicy:
    """Upper bound: knows every arm's true alpha and plays the exact argmax."""

    def __init__(self, cost: CostModel, alphas: dict[CompressionConfig, float], gamma_max: int = 8):
        self.cost = cost
        self.alphas = alphas
        self.gamma_max = gamma_max

    def select(self, ctx: int, key: str | None = None) -> tuple[CompressionConfig, int]:
        rows = []
        for cfg, a in self.alphas.items():
            g, t = best_gamma(self.cost, cfg, a, ctx, self.gamma_max)
            rows.append((t, cfg, g))
        t, cfg, g = max(rows)
        return cfg, g

    def update(self, cfg, gamma, accepted, ctx, key=None) -> None:
        pass


class DepthThompsonController:
    """Thompson sampling over configs with a *depth-indexed* acceptance model.

    `ThompsonController` estimates one alpha per config, which is what the
    geometric E(alpha, gamma) assumes.  Measurement says that assumption is
    wrong in a specific, exploitable direction: acceptance rises with draft
    depth, because reaching depth i is evidence the current region drafts
    easily.  Fitting one alpha to that data anchors it near the depth-0 rate
    and truncates gamma far too early.

    Here each (config, depth) keeps its own Beta.  The data is identical -- a
    round with k accepted out of gamma reports successes at depths 0..k-1 and
    one failure at depth k -- it is only indexed differently.  Depths never
    observed borrow the deepest depth that has data, rather than falling back
    on an optimistic prior.
    """

    def __init__(
        self,
        cost: CostModel,
        arms: Sequence[CompressionConfig] | None = None,
        gamma_max: int = 12,
        decay: float = 0.995,
        min_speedup: float = 1.05,
        switch_amortize: int = 32,
        seed: int = 0,
        prior: tuple[float, float] = (2.0, 1.0),
        min_obs: float = 1.0,
    ) -> None:
        self.cost = cost
        self.arms = list(arms or default_arms())
        self.gamma_max = gamma_max
        self.decay = decay
        self.min_speedup = min_speedup
        self.switch_amortize = max(1, switch_amortize)
        self.rng = np.random.default_rng(seed)
        self.prior = prior
        self.min_obs = min_obs
        self.post: dict[tuple[str | None, CompressionConfig], list[_Beta]] = {}
        self.obs: dict[tuple[str | None, CompressionConfig], list[float]] = {}
        self._held: CompressionConfig | None = None
        self.switches = 0

    def _profile_post(self, cfg: CompressionConfig, key: str | None) -> list[_Beta]:
        k = (key, cfg)
        if k not in self.post:
            self.post[k] = [_Beta(*self.prior) for _ in range(self.gamma_max)]
            self.obs[k] = [0.0] * self.gamma_max
        return self.post[k]

    def sample_profile(self, cfg: CompressionConfig, key: str | None = None) -> list[float]:
        """One Thompson draw of the whole depth profile."""
        post = self._profile_post(cfg, key)
        obs = self.obs[(key, cfg)]
        out: list[float] = []
        last = None
        for d in range(self.gamma_max):
            if obs[d] >= self.min_obs:
                last = post[d].sample(self.rng)
                out.append(last)
            else:
                out.append(last if last is not None else post[d].sample(self.rng))
        return out

    def mean_profile(self, cfg: CompressionConfig, key: str | None = None) -> list[float]:
        post = self._profile_post(cfg, key)
        obs = self.obs[(key, cfg)]
        out, last = [], None
        for d in range(self.gamma_max):
            if obs[d] >= self.min_obs:
                last = post[d].mean
            out.append(last if last is not None else post[d].mean)
        return out

    # ------------------------------------------------------------------ api
    def select(self, ctx: int, key: str | None = None) -> tuple[CompressionConfig, int]:
        base = 1.0 / self.cost.t_baseline(ctx)
        penalty = self.cost.rebuild_cost(ctx) / self.switch_amortize
        best_cfg, best_g, best_t = self.arms[0], 0, base
        for cfg in self.arms:
            prof = self.sample_profile(cfg, key)
            extra = 0.0 if cfg == self._held else penalty
            g, t = best_gamma_profile(self.cost, cfg, prof, ctx, extra)
            if g > 0 and t > best_t:
                best_cfg, best_g, best_t = cfg, g, t
        if best_g > 0 and best_cfg != self._held:
            self._held = best_cfg
            self.switches += 1
        if best_g > 0 and best_t < base * self.min_speedup:
            best_g = 0
        return best_cfg, best_g

    def update(self, cfg, gamma, accepted, ctx, key=None) -> None:
        if gamma <= 0:
            return
        post = self._profile_post(cfg, key)
        obs = self.obs[(key, cfg)]
        for d in range(min(accepted, self.gamma_max)):
            post[d].observe(1, 1, self.decay)
            obs[d] += 1
        if accepted < gamma and accepted < self.gamma_max:
            post[accepted].observe(0, 1, self.decay)
            obs[accepted] += 1
