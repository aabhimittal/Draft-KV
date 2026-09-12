"""Per-layer bit allocation.

Compression is normally applied uniformly, which implicitly assumes every
layer's KV cache is equally worth its bytes.  On the reference model that is
measurably false: the same 2-bit/keep-0.5 damage applied to one layer at a time
drops acceptance to 0.62 in the most tolerant layer and 0.33 in the most
sensitive one.  This module measures that curve and spends a byte budget where
it buys the most acceptance.

The method is deliberately simple, because the expensive part is measurement,
not optimization:

1. profile  -- for each (layer, candidate config), measure acceptance with only
               that layer degraded.  Convert to a damage in log-acceptance.
2. allocate -- greedy knapsack: repeatedly take the upgrade with the best
               damage-reduction per extra byte until the budget is spent.

Step 2 assumes per-layer damages add in log space.  That is an assumption, not
a theorem, so `predicted_alpha` exposes the prediction and
`tests/test_layers.py` checks it against a measured mixed plan rather than
taking it on faith.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .config import CompressionConfig, LayerPlan
from .controller import FixedPolicy
from .engine import DraftKVEngine


def measure_alpha_detail(
    model,
    prompt: Sequence[int],
    plan: LayerPlan | CompressionConfig,
    gamma: int = 6,
    n_new: int = 96,
    seed: int = 0,
) -> tuple[float, int]:
    """Observed acceptance rate for a plan, plus the number of drafted tokens.

    The trial count is returned because it is the thing that decides whether a
    profile is usable.  Acceptance differences between neighbouring configs are
    often ~0.05, and a short run resolves them no better than a coin flip --
    which silently corrupts the allocation downstream.
    """
    eng = DraftKVEngine(model, FixedPolicy(plan, gamma), seed=seed)
    r = eng.generate(list(prompt), n_new, 0.0)
    return float(r.acceptance_rate), int(sum(x.gamma for x in r.rounds))


def measure_alpha(
    model,
    prompt: Sequence[int],
    plan: LayerPlan | CompressionConfig,
    gamma: int = 6,
    n_new: int = 96,
    seed: int = 0,
) -> float:
    return measure_alpha_detail(model, prompt, plan, gamma, n_new, seed)[0]


@dataclass
class SensitivityProfile:
    """Measured cost, in log-acceptance, of degrading one layer at a time."""

    reference: CompressionConfig
    alpha_ref: float
    candidates: tuple[CompressionConfig, ...]
    alpha: dict[tuple[int, CompressionConfig], float] = field(default_factory=dict)
    trials: dict[tuple[int, CompressionConfig], int] = field(default_factory=dict)

    @property
    def min_trials(self) -> int:
        return min(self.trials.values()) if self.trials else 0

    def damage(self, layer: int, cfg: CompressionConfig) -> float:
        """log(alpha_ref) - log(alpha with only `layer` set to `cfg`); >= 0."""
        a = self.alpha.get((layer, cfg), self.alpha_ref)
        return max(0.0, float(np.log(max(self.alpha_ref, 1e-6)) - np.log(max(a, 1e-6))))

    def predicted_alpha(self, plan: LayerPlan) -> float:
        """Log-additive prediction for a mixed plan."""
        total = sum(self.damage(L, c) for L, c in enumerate(plan))
        return float(np.clip(self.alpha_ref * np.exp(-total), 0.0, 1.0))


def profile_layers(
    model,
    prompt: Sequence[int],
    candidates: Sequence[CompressionConfig],
    reference: CompressionConfig = CompressionConfig(16, 1.0),
    gamma: int = 6,
    n_new: int = 96,
    seed: int = 0,
) -> SensitivityProfile:
    """One run per (layer, candidate).  Offline calibration, not a hot path."""
    n = model.n_layers
    alpha_ref, _ = measure_alpha_detail(
        model, prompt, LayerPlan.uniform(reference, n), gamma, n_new, seed
    )
    prof = SensitivityProfile(reference, alpha_ref, tuple(candidates))
    for L in range(n):
        for c in candidates:
            cfgs = [reference] * n
            cfgs[L] = c
            a, t = measure_alpha_detail(
                model, prompt, LayerPlan(tuple(cfgs)), gamma, n_new, seed
            )
            prof.alpha[(L, c)], prof.trials[(L, c)] = a, t
    return prof


def _costs(prof: SensitivityProfile, ctx: int, n_kv_heads: int, d_head: int):
    cost_of = lambda c: LayerPlan((c,)).per_layer_bytes(ctx, n_kv_heads, d_head)
    ranked = sorted(prof.candidates, key=cost_of)
    return ranked, [cost_of(c) for c in ranked]


def allocate_greedy(
    prof: SensitivityProfile,
    budget_bytes: float,
    ctx: int,
    n_kv_heads: int,
    d_head: int,
    n_layers: int,
) -> LayerPlan:
    """Greedy over adjacent per-layer upgrades, best damage-drop per byte first.

    Kept as a baseline because it is the obvious thing to write and it is
    measurably wrong.  Greedy is optimal for *fractional* knapsack; here the
    choices are discrete, so it will happily sink the whole budget into lifting
    two layers to the top config while a third stays at 2-bit -- a plan its own
    damage model scores worse than plain uniform 4-bit.  `allocate` solves the
    same objective exactly instead.
    """
    ranked, costs = _costs(prof, ctx, n_kv_heads, d_head)
    idx = [0] * n_layers
    plan = lambda: LayerPlan(tuple(ranked[i] for i in idx))
    spent = sum(costs[i] for i in idx)
    while True:
        best = None
        for L in range(n_layers):
            if idx[L] + 1 >= len(ranked):
                continue
            cur, nxt = ranked[idx[L]], ranked[idx[L] + 1]
            d_bytes = costs[idx[L] + 1] - costs[idx[L]]
            gain = (prof.damage(L, cur) - prof.damage(L, nxt)) / d_bytes if d_bytes > 0 else np.inf
            if spent + d_bytes <= budget_bytes and gain > 0:
                if best is None or gain > best[0]:
                    best = (gain, L, d_bytes)
        if best is None:
            return plan()
        _, L, d_bytes = best
        idx[L] += 1
        spent += d_bytes


def allocate(
    prof: SensitivityProfile,
    budget_bytes: float,
    ctx: int,
    n_kv_heads: int,
    d_head: int,
    n_layers: int,
    max_states: int = 20000,
) -> LayerPlan:
    """Exact multiple-choice knapsack: minimize total damage under a byte budget.

    Each layer picks exactly one candidate config; damages are summed (the
    log-additive assumption) and the exact byte sum must fit.

    Solved by carrying a Pareto frontier of (cost, damage) states layer by layer
    and pruning any state that costs more and damages more than another.  No
    byte axis, so no quantization: an earlier version bucketed the budget and
    lost exactly-affordable plans, because three layers each rounding up by a
    third of a bucket read as one bucket over budget, and the DP then returned
    something strictly worse than plain uniform compression.  Rounding down is
    no better -- it returns plans that do not fit.  Working in exact costs
    removes the whole failure mode.

    The frontier stays small in practice (it is bounded by the number of
    distinct achievable cost sums); `max_states` caps it for very deep models by
    keeping the lowest-damage states.

    Returns the all-cheapest plan when even that will not fit.
    """
    ranked, costs = _costs(prof, ctx, n_kv_heads, d_head)
    cheapest = LayerPlan(tuple(ranked[0] for _ in range(n_layers)))
    if costs[0] * n_layers > budget_bytes + 1e-9:
        return cheapest

    frontier: list[tuple[float, float]] = [(0.0, 0.0)]
    back: list[list[tuple[int, int]]] = []
    for L in range(n_layers):
        states: list[tuple[float, float, int, int]] = []
        # every layer after this one still needs at least the cheapest config
        reserve = costs[0] * (n_layers - L - 1)
        for pi, (c0, d0) in enumerate(frontier):
            for j, cfg in enumerate(ranked):
                c = c0 + costs[j]
                if c + reserve > budget_bytes + 1e-9:
                    continue
                states.append((c, d0 + prof.damage(L, cfg), j, pi))
        if not states:
            return cheapest
        states.sort(key=lambda s: (s[0], s[1]))
        pruned, best_d = [], float("inf")
        for st in states:
            if st[1] < best_d - 1e-12:
                pruned.append(st)
                best_d = st[1]
        if len(pruned) > max_states:
            pruned = sorted(pruned, key=lambda s: s[1])[:max_states]
            pruned.sort(key=lambda s: (s[0], s[1]))
        frontier = [(c, d) for c, d, _, _ in pruned]
        back.append([(j, pi) for _, _, j, pi in pruned])

    end = min(range(len(frontier)), key=lambda i: frontier[i][1])
    picks: list[int] = []
    b = end
    for L in range(n_layers - 1, -1, -1):
        j, parent = back[L][b]
        picks.append(j)
        b = parent
    return LayerPlan(tuple(ranked[j] for j in reversed(picks)))
