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

Step 2 needs a rule for how per-layer damages *compose* into the damage of a
mixed plan.  Log-additivity (equivalently, acceptance multiplies across layers)
is the obvious guess and it is measurably wrong: it over-predicts damage badly
once several layers are degraded at once, because the errors they introduce
overlap rather than stack.

`SensitivityProfile.p` generalizes it to a power mean,

    predicted damage = (sum_L D_L ** p) ** (1/p)

with p = 1 recovering log-additivity.  p > 1 interpolates toward "only the
worst layer matters" (damages overlap); p < 1 toward super-additive (damages
amplify each other).  Both directions occur -- which one shows up is a property
of the model, not a constant -- so `fit_composition` searches across p = 1
rather than assuming a side.

The optimizer does not change: minimizing (sum D**p)**(1/p) is the same as
minimizing sum D**p, which is still separable, so the same exact knapsack DP
runs against `objective()` instead of `damage()`.
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
    spread: dict[tuple[int, CompressionConfig], float] = field(default_factory=dict)
    p: float = 1.0          # composition exponent; 1.0 == log-additive
    n_prompts: int = 1

    @property
    def min_trials(self) -> int:
        return min(self.trials.values()) if self.trials else 0

    @property
    def noise(self) -> float:
        """Typical sd of log-acceptance for one cell across prompts.

        This is the number that decides whether a profile means anything.
        Measured on distilgpt2 it is ~1.2 -- larger than most of the damages
        being estimated -- so a single-prompt profile is not a measurement of
        the layer, it is a measurement of the prompt.
        """
        vals = [v for v in self.spread.values() if v == v]
        return float(np.mean(vals)) if vals else float("nan")

    def damage(self, layer: int, cfg: CompressionConfig) -> float:
        """log(alpha_ref) - log(alpha with only `layer` set to `cfg`); >= 0."""
        a = self.alpha.get((layer, cfg), self.alpha_ref)
        return max(0.0, float(np.log(max(self.alpha_ref, 1e-6)) - np.log(max(a, 1e-6))))

    def objective(self, layer: int, cfg: CompressionConfig) -> float:
        """The separable per-layer term the allocator minimizes: D ** p."""
        return self.damage(layer, cfg) ** self.p

    def composed_damage(self, plan: LayerPlan) -> float:
        """Power mean of per-layer damages, the p-norm of the damage vector."""
        total = sum(self.objective(L, c) for L, c in enumerate(plan))
        return float(total ** (1.0 / self.p)) if total > 0 else 0.0

    def predicted_alpha(self, plan: LayerPlan) -> float:
        """Predicted acceptance for a mixed plan under the composition rule."""
        return float(np.clip(self.alpha_ref * np.exp(-self.composed_damage(plan)), 0.0, 1.0))


def _as_prompts(prompt) -> list[Sequence[int]]:
    """Accept a single prompt or a list of them."""
    if len(prompt) and isinstance(prompt[0], (list, tuple, np.ndarray)):
        return list(prompt)
    return [prompt]


def profile_layers(
    model,
    prompt: Sequence[int] | Sequence[Sequence[int]],
    candidates: Sequence[CompressionConfig],
    reference: CompressionConfig = CompressionConfig(16, 1.0),
    gamma: int = 6,
    n_new: int = 96,
    seed: int = 0,
) -> SensitivityProfile:
    """One run per (layer, candidate, prompt).  Offline calibration, not a hot path.

    `prompt` may be a list of prompts, and should be: acceptance measured on a
    single prompt is dominated by that prompt's content, not by the layer under
    test.  Cells are aggregated as a geometric mean (the mean of log-alpha,
    which is the space damages live in) and `spread` records the sd across
    prompts so the caller can see whether a damage estimate outruns its own
    noise.
    """
    prompts = _as_prompts(prompt)
    n = model.n_layers

    def cell(plan: LayerPlan) -> tuple[float, int, float]:
        logs, trials = [], 0
        for i, pr in enumerate(prompts):
            a, t = measure_alpha_detail(model, pr, plan, gamma, n_new, seed + i)
            logs.append(np.log(max(a, 1e-4)))
            trials += t
        return float(np.exp(np.mean(logs))), trials, float(np.std(logs))

    alpha_ref, _, _ = cell(LayerPlan.uniform(reference, n))
    prof = SensitivityProfile(reference, alpha_ref, tuple(candidates), n_prompts=len(prompts))
    for L in range(n):
        for c in candidates:
            cfgs = [reference] * n
            cfgs[L] = c
            a, t, sd = cell(LayerPlan(tuple(cfgs)))
            prof.alpha[(L, c)], prof.trials[(L, c)], prof.spread[(L, c)] = a, t, sd
    return prof


def fit_composition(
    prof: SensitivityProfile,
    observations: Sequence[tuple[LayerPlan, float]],
    grid: Sequence[float] = tuple(np.round(np.concatenate([
        np.arange(0.2, 1.0, 0.05), np.arange(1.0, 6.01, 0.05)]), 2)),
    require_cv: bool = True,
) -> dict:
    """Fit the composition exponent `p` to measured mixed plans -- if it helps.

    Candidates are scored by RMSE in log-acceptance, the space damages live in.

    `require_cv` is the important part.  Fitting one scalar to a dozen noisy
    plans improves in-sample error almost for free, and that improvement is
    largely imaginary.  Measured on distilgpt2 with a properly averaged
    profile, p = 0.90 cut in-sample RMSE from 0.098 to 0.096 while making
    leave-one-out error *worse*: 0.098 -> 0.114.  So a fit is adopted only when
    leave-one-out says it generalizes; otherwise `p` stays at 1.0 and the
    result records that it was rejected.  Pass require_cv=False to inspect the
    unguarded fit.

    Returns in-sample and leave-one-out error for both models, plus `adopted`.
    """
    keep = float(prof.p)
    if len(observations) < 2:
        return {"p": keep, "best_p": keep, "rmse_before": float("nan"),
                "rmse_after": float("nan"), "loo_before": float("nan"),
                "loo_after": float("nan"), "adopted": False, "n": len(observations)}

    def err(p: float, obs) -> float:
        old, prof.p = prof.p, p
        e = [
            (np.log(max(prof.predicted_alpha(pl), 1e-4)) - np.log(max(a, 1e-4))) ** 2
            for pl, a in obs
        ]
        prof.p = old
        return float(np.sqrt(np.mean(e)))

    best_p = min(grid, key=lambda q: err(q, observations))
    before, after = err(1.0, observations), err(best_p, observations)

    loo_b, loo_a = [], []
    for i in range(len(observations)):
        train = list(observations[:i]) + list(observations[i + 1:])
        held = [observations[i]]
        loo_b.append(err(1.0, held) ** 2)
        loo_a.append(err(min(grid, key=lambda q: err(q, train)), held) ** 2)
    loo_before, loo_after = float(np.sqrt(np.mean(loo_b))), float(np.sqrt(np.mean(loo_a)))

    adopted = (not require_cv) or (loo_after < loo_before)
    prof.p = float(best_p) if adopted else 1.0
    return {"p": prof.p, "best_p": float(best_p), "rmse_before": before,
            "rmse_after": after, "loo_before": loo_before, "loo_after": loo_after,
            "adopted": bool(adopted), "n": len(observations)}


def calibrate_composition(
    model,
    prompt: Sequence[int],
    prof: SensitivityProfile,
    n_plans: int = 12,
    gamma: int = 6,
    n_new: int = 128,
    seed: int = 0,
) -> dict:
    """Measure random mixed plans and fit `p` to them.

    Costs `n_plans` extra profiling runs on top of the per-layer sweep. That is
    the price of not guessing how damages compose.
    """
    rng = np.random.default_rng(seed)
    n_layers = model.n_layers
    obs: list[tuple[LayerPlan, float]] = []
    for _ in range(n_plans):
        cfgs = tuple(prof.candidates[int(rng.integers(len(prof.candidates)))] for _ in range(n_layers))
        plan = LayerPlan(cfgs)
        obs.append((plan, measure_alpha(model, prompt, plan, gamma, n_new, seed)))
    return fit_composition(prof, obs)


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
            gain = (prof.objective(L, cur) - prof.objective(L, nxt)) / d_bytes if d_bytes > 0 else np.inf
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
                states.append((c, d0 + prof.objective(L, cfg), j, pi))
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
