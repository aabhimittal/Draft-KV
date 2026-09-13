"""Cheap simulator: a bandit environment with known ground-truth alpha(c).

Running thousands of controller decisions against the real transformer would
take minutes and tell you nothing extra about the controller -- the model only
matters for proving losslessness.  Here alpha(c) is stipulated, so regret
against the true optimum is exactly computable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import CompressionConfig
from .throughput import CostModel, best_gamma, best_gamma_profile, expected_tokens


def plausible_alpha(cfg: CompressionConfig, difficulty: float = 1.0) -> float:
    """Stylized alpha(c): degradation grows as bits fall and tokens are dropped.

    Shape, not calibration -- real values are content dependent, which is
    exactly why the controller has to measure them instead of assuming them.
    `difficulty` scales the whole penalty (prose tolerant, code brittle).
    """
    bit_pen = {16: 0.0, 8: 0.02, 4: 0.12, 3: 0.25, 2: 0.45}[cfg.bits]
    drop_pen = 0.55 * (1.0 - cfg.keep_frac) ** 0.8
    return float(np.clip(1.0 - difficulty * (bit_pen + drop_pen), 0.01, 0.995))


def rising_profile(alpha0: float, depth: int = 12, gain: float = 0.45) -> list[float]:
    """Depth profile of the shape measured on the reference model.

    Acceptance starts at `alpha0` and climbs toward 1 with depth -- survivorship,
    not drift: a draft that has already survived i tokens is in a stretch the
    compressed cache happens to model well.  `gain` sets how much of the gap to
    1.0 is closed by depth ~3.
    """
    return [float(np.clip(1.0 - (1.0 - alpha0) * (1.0 - gain) ** d, 0.0, 1.0)) for d in range(depth)]


@dataclass
class BanditEnv:
    """Draws (accepted | gamma) from the acceptance model.

    `alphas` gives one constant acceptance rate per config.  `profiles`, when
    supplied, overrides it with a depth-indexed profile per config -- the shape
    the reference model actually exhibits.
    """

    alphas: dict[CompressionConfig, float]
    ctx: int = 32768
    seed: int = 0
    profiles: dict[CompressionConfig, Sequence[float]] | None = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def accept_profile(self, cfg: CompressionConfig, depth: int) -> list[float]:
        if self.profiles is not None:
            p = list(self.profiles[cfg])
            return (p + [p[-1]] * depth)[:depth]
        return [self.alphas[cfg]] * depth

    def pull(self, cfg: CompressionConfig, gamma: int) -> int:
        prof = self.accept_profile(cfg, gamma)
        k = 0
        while k < gamma and self.rng.random() < prof[k]:
            k += 1
        return k

    def true_best(self, cost: CostModel, gamma_max: int = 8) -> tuple[CompressionConfig, int, float]:
        rows = []
        for cfg in self.alphas:
            g, t = best_gamma_profile(cost, cfg, self.accept_profile(cfg, gamma_max), self.ctx)
            rows.append((t, cfg, g))
        t, cfg, g = max(rows)
        return cfg, g, t


def run_bandit(
    controller,
    env: BanditEnv,
    cost: CostModel,
    rounds: int = 2000,
    key: str | None = None,
) -> dict:
    """Play `rounds` decisions; score each by its *true* expected throughput.

    Changing config is charged `cost.rebuild_cost(ctx)` on the round it
    happens.  Omitting that charge is the easy way to make any bandit look
    good: it rewards thrashing between near-equal arms, which on real hardware
    means requantizing the whole drafter cache every few tokens.
    """
    _, _, opt_t = env.true_best(cost, 12)
    total_t = 0.0
    regret = []
    picks = []
    prev: CompressionConfig | None = None
    switches = 0
    for _ in range(rounds):
        cfg, gamma = controller.select(env.ctx, key)
        if gamma > 0:
            k = env.pull(cfg, gamma)
            extra = 0.0 if (prev is None or cfg == prev) else cost.rebuild_cost(env.ctx)
            switches += extra > 0
            # scored against the environment's TRUE profile, whatever model the
            # controller used to choose -- so a mis-specified model is penalized
            t = cost.throughput_profile(cfg, env.accept_profile(cfg, gamma), env.ctx, extra)
            prev = cfg
        else:
            k = 0
            t = 1.0 / cost.t_baseline(env.ctx)
        controller.update(cfg, gamma, k, env.ctx, key)
        total_t += t
        regret.append(opt_t - t)
        picks.append((cfg, gamma))
    cum = np.cumsum(regret)
    return dict(
        mean_throughput=total_t / rounds,
        optimal_throughput=opt_t,
        cumulative_regret=float(cum[-1]),
        normalized_regret=float(cum[-1] / (opt_t * rounds)),
        switches=switches,
        final_pick=picks[-1],
        picks=picks,
    )


@dataclass
class LayerEnv:
    """Simulated environment with a known per-layer sensitivity, for testing
    the adaptive allocator without paying for model runs.

    Acceptance for a plan is the reference rate discounted by each layer's
    damage at the config it was given, composed log-additively (which the
    real-model measurements support once the profile is averaged properly).
    `shift()` moves the sensitivity to a different layer, which is the case a
    static offline profile cannot handle and an online one should.
    """

    damage: dict[int, float]            # layer -> damage at the cheapest config
    ranked: Sequence[CompressionConfig]  # cheapest first
    alpha_ref: float = 0.95
    ctx: int = 32768
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def alpha(self, plan) -> float:
        total = 0.0
        for L, c in enumerate(plan):
            notches = len(self.ranked) - 1 - list(self.ranked).index(c)
            total += self.damage.get(L, 0.0) * notches / max(1, len(self.ranked) - 1)
        return float(np.clip(self.alpha_ref * np.exp(-total), 0.01, 0.999))

    def pull(self, plan, gamma: int) -> int:
        a = self.alpha(plan)
        k = 0
        while k < gamma and self.rng.random() < a:
            k += 1
        return k

    def shift(self, damage: dict[int, float]) -> None:
        self.damage = damage
