"""Cheap simulator: a bandit environment with known ground-truth alpha(c).

Running thousands of controller decisions against the real transformer would
take minutes and tell you nothing extra about the controller -- the model only
matters for proving losslessness.  Here alpha(c) is stipulated, so regret
against the true optimum is exactly computable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CompressionConfig
from .throughput import CostModel, best_gamma, expected_tokens


def plausible_alpha(cfg: CompressionConfig, difficulty: float = 1.0) -> float:
    """Stylized alpha(c): degradation grows as bits fall and tokens are dropped.

    Shape, not calibration -- real values are content dependent, which is
    exactly why the controller has to measure them instead of assuming them.
    `difficulty` scales the whole penalty (prose tolerant, code brittle).
    """
    bit_pen = {16: 0.0, 8: 0.02, 4: 0.12, 3: 0.25, 2: 0.45}[cfg.bits]
    drop_pen = 0.55 * (1.0 - cfg.keep_frac) ** 0.8
    return float(np.clip(1.0 - difficulty * (bit_pen + drop_pen), 0.01, 0.995))


@dataclass
class BanditEnv:
    """Draws (accepted | gamma) from the truncated geometric implied by alpha."""

    alphas: dict[CompressionConfig, float]
    ctx: int = 32768
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def pull(self, cfg: CompressionConfig, gamma: int) -> int:
        a = self.alphas[cfg]
        k = 0
        while k < gamma and self.rng.random() < a:
            k += 1
        return k

    def true_best(self, cost: CostModel, gamma_max: int = 8) -> tuple[CompressionConfig, int, float]:
        rows = []
        for cfg, a in self.alphas.items():
            g, t = best_gamma(cost, cfg, a, self.ctx, gamma_max)
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
    _, _, opt_t = env.true_best(cost)
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
            t = cost.throughput(cfg, gamma, env.alphas[cfg], env.ctx, extra)
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
