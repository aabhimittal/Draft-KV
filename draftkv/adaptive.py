"""Learning the per-layer budget online, instead of profiling it offline.

The static allocator in `layers.py` measures a sensitivity table once and
allocates against it forever.  The real-model work showed why that is fragile:
acceptance is dominated by content, the same plan measures 0.95 on one passage
and 0.10 on another, and a profile taken on one corpus is a prior rather than
an answer.

This module keeps the same objective and the same exact knapsack, but sources
its damages from the live verify pass rather than a calibration run.  Most
rounds use the current plan; occasionally one runs a **swap probe** -- a notch
taken from one layer and given to another.

Swapping rather than simply degrading a layer matters.  Degrading is only
possible while a layer is above the floor, so as the plan compresses, the
layers most in need of measurement become the ones that can no longer be
probed and learning stalls exactly where it is needed.  A swap is always
feasible while any layer is above the floor, is byte-neutral by construction
so it never breaches the memory budget, and measures the comparison the
allocator actually has to make: is this layer worth more than that one?

Damages are posterior means, the knapsack is re-solved periodically, and
everything is keyed by content so code and prose hold different plans.

The reason this is allowed at all is the premise of the whole project:
**output is identical whatever the plan**, so a probe cannot corrupt anything.
It costs a little throughput on the rounds it runs and nothing else.  An
allocator for a compressor that traded against quality could not explore in
production; this one can.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .config import CompressionConfig, LayerPlan, default_arms
from .controller import _Beta
from .layers import SensitivityProfile, allocate
from .throughput import CostModel, best_gamma


@dataclass
class _LayerStats:
    """Running estimate of how much each layer repays a notch of budget.

    `value[L]` is updated by *paired comparison*: a swap probe that moves a
    notch from M to L scores its acceptance against the current base rate, and
    credits +delta to L and -delta to M.

    The obvious alternative -- one posterior for "L was upgraded" and another
    for "L was downgraded" -- looks equivalent and is not. A layer pinned at
    the ceiling can never be upgraded and one at the floor can never be
    downgraded, so exactly the layers the allocator has committed to hardest
    keep one stale side forever, and the comparison between the two sides
    silently measures age rather than sensitivity. In testing that mis-assigned
    a shifted sensitivity to the wrong layer entirely. Paired updates touch
    both layers on every probe regardless of where they sit.
    """

    base: _Beta
    value: dict[int, float] = field(default_factory=dict)
    probe_n: dict[int, int] = field(default_factory=dict)


class AdaptiveLayerController:
    """Controller protocol, allocating a per-layer byte budget online.

    `select` returns a `LayerPlan`, which the engine already accepts wherever a
    `CompressionConfig` goes, so nothing downstream changes.
    """

    def __init__(
        self,
        cost: CostModel,
        n_layers: int,
        budget_bytes: float,
        candidates: Sequence[CompressionConfig] | None = None,
        ctx_hint: int = 4096,
        gamma_max: int = 8,
        probe_prob: float = 0.15,
        max_probe_overhead: float = 0.05,
        resolve_every: int = 64,
        decay: float = 0.995,
        lr: float = 0.08,
        min_speedup: float = 1.05,
        stall_limit: int = 12,
        seed: int = 0,
        prior: tuple[float, float] = (2.0, 1.0),
    ) -> None:
        self.cost = cost
        self.n_layers = n_layers
        self.budget_bytes = budget_bytes
        self.candidates = tuple(candidates or [
            CompressionConfig(4, 0.25), CompressionConfig(4, 0.5), CompressionConfig(4, 1.0)
        ])
        self.ranked = tuple(sorted(
            self.candidates,
            key=lambda c: LayerPlan((c,)).per_layer_bytes(ctx_hint, cost.n_kv_heads, cost.d_head),
        ))
        self.ctx_hint = ctx_hint
        self.gamma_max = gamma_max
        self.probe_prob = probe_prob
        self.max_probe_overhead = max_probe_overhead
        self.resolve_every = max(1, resolve_every)
        self.decay = decay
        self.lr = lr
        self.min_speedup = min_speedup
        self.stall_limit = max(1, stall_limit)
        self.prior = prior
        self._stall: dict[str | None, int] = {}
        self.n_stall_recoveries = 0
        self.rng = np.random.default_rng(seed)

        self.stats: dict[str | None, _LayerStats] = {}
        self.plans: dict[str | None, LayerPlan] = {}
        self._last: tuple[str | None, int | None] | None = None
        self.n_decisions = 0
        self.n_probes = 0
        self.n_resolves = 0

    # ------------------------------------------------------------- internals
    def _stats(self, key: str | None) -> _LayerStats:
        if key not in self.stats:
            self.stats[key] = _LayerStats(base=_Beta(*self.prior))
        return self.stats[key]

    def _plan(self, key: str | None) -> LayerPlan:
        """Start from the most generous plan the budget affords everywhere."""
        if key not in self.plans:
            affordable = [
                c for c in self.ranked
                if LayerPlan.uniform(c, self.n_layers).per_layer_bytes(
                    self.ctx_hint, self.cost.n_kv_heads, self.cost.d_head) <= self.budget_bytes
            ]
            start = affordable[-1] if affordable else self.ranked[0]
            self.plans[key] = LayerPlan.uniform(start, self.n_layers)
        return self.plans[key]

    def _notch_cost(self, i: int) -> float:
        return LayerPlan((self.ranked[i],)).per_layer_bytes(
            self.ctx_hint, self.cost.n_kv_heads, self.cost.d_head)

    def _swap_probe(self, base: LayerPlan, key: str | None
                    ) -> tuple[LayerPlan, int, int] | None:
        """One notch from layer `down` to layer `up`, staying inside the budget.

        Notch sizes are *not* uniform -- going keep-0.5 -> keep-1.0 costs twice
        what keep-0.25 -> keep-0.5 does -- so a swap chosen by position rather
        than by bytes is usually over budget and gets rejected. An earlier
        version did exactly that and silently lost ~80% of its probes, which
        starved the layers it most needed to measure. So enumerate the pairs
        that actually fit and choose among those, preferring the least-probed
        `up`.
        """
        idx = [self.ranked.index(c) for c in base.cfgs]
        spent = base.per_layer_bytes(self.ctx_hint, self.cost.n_kv_heads, self.cost.d_head)
        headroom = self.budget_bytes - spent
        st = self._stats(key)

        feasible: list[tuple[int, int, int]] = []      # (probe_count, up, down)
        for up in range(self.n_layers):
            if idx[up] + 1 >= len(self.ranked):
                continue
            gain = self._notch_cost(idx[up] + 1) - self._notch_cost(idx[up])
            for down in range(self.n_layers):
                if down == up or idx[down] == 0:
                    continue
                give = self._notch_cost(idx[down]) - self._notch_cost(idx[down] - 1)
                if gain - give <= headroom + 1e-6:
                    feasible.append((st.probe_n.get(up, 0), up, down))
        if not feasible:
            return None
        fewest = min(f[0] for f in feasible)
        pool = [f for f in feasible if f[0] == fewest]
        _, up, down = pool[int(self.rng.integers(len(pool)))]

        cfgs = list(base.cfgs)
        cfgs[up] = self.ranked[idx[up] + 1]
        cfgs[down] = self.ranked[idx[down] - 1]
        return LayerPlan(tuple(cfgs)), up, down

    def effective_probe_prob(self, ctx: int) -> float:
        """Probe rate, capped so exploration stays inside a throughput budget.

        Probing is free in *correctness* -- output is identical whatever the
        plan -- but not in cost: a probe changes the plan, and changing the
        plan requantizes the whole drafter cache, which is O(ctx). At long
        context that is far more expensive than the round it rides on, so a
        fixed probe rate would quietly spend a large fraction of the throughput
        it is trying to earn.

        So the rate is the smaller of `probe_prob` and the rate at which
        rebuilds cost at most `max_probe_overhead` of a round.
        """
        rebuild = self.cost.rebuild_cost(ctx, layers_changed=2)   # a swap touches two
        if rebuild <= 0:
            return self.probe_prob
        # a probe costs one rebuild now and one to switch back
        round_cost = self.cost.t_verify(ctx, self.gamma_max)
        affordable = self.max_probe_overhead * round_cost / (2.0 * rebuild)
        return float(min(self.probe_prob, max(0.0, affordable)))

    # ------------------------------------------------------------------- api
    def select(self, ctx: int, key: str | None = None) -> tuple[LayerPlan, int]:
        self.n_decisions += 1
        if self.n_decisions % self.resolve_every == 0:
            self.resolve(key)

        base = self._plan(key)
        st = self._stats(key)
        plan, probed = base, None
        if self.rng.random() < self.effective_probe_prob(ctx):
            got = self._swap_probe(base, key)
            if got is not None:
                plan, up, down = got
                probed = (up, down)
                self.n_probes += 1
        self._last = (key, probed)

        alpha = st.base.mean
        g, t = best_gamma(self.cost, plan, alpha, ctx, self.gamma_max)
        if g > 0 and t < self.min_speedup / self.cost.t_baseline(ctx):
            g = 0

        # gamma == 0 is a dead end for a *learning* allocator: the round runs on
        # the full cache, so it reports nothing about the plan, so the plan
        # never improves and the controller sits there forever. Treat a run of
        # them as evidence against the plan rather than as a steady state.
        if g == 0:
            self._stall[key] = self._stall.get(key, 0) + 1
            if self._stall[key] >= self.stall_limit:
                self._recover(key)
                plan = self.plans[key]
                st.base = _Beta(*self.prior)
                g, t = best_gamma(self.cost, plan, st.base.mean, ctx, self.gamma_max)
                self._last = (key, None)
        else:
            self._stall[key] = 0
        return plan, g

    def update(self, cfg, gamma: int, accepted: int, ctx: int, key: str | None = None) -> None:
        if gamma <= 0 or self._last is None:
            return   # a full-KV round says nothing about the compressed plan
        k, probed = self._last
        st = self._stats(k)
        if probed is None:
            st.base.observe(accepted, gamma, self.decay)
            return
        up, down = probed
        observed = np.log(max(accepted / gamma, 1e-3))
        expected = np.log(max(st.base.mean, 1e-3))
        delta = float(observed - expected)
        st.value[up] = (1 - self.lr) * st.value.get(up, 0.0) + self.lr * delta
        st.value[down] = (1 - self.lr) * st.value.get(down, 0.0) - self.lr * delta
        st.probe_n[up] = st.probe_n.get(up, 0) + 1
        st.probe_n[down] = st.probe_n.get(down, 0) + 1

    def _spend_surplus(self, plan: LayerPlan, key: str | None) -> LayerPlan:
        """Upgrade layers while budget remains.

        Unspent budget buys nothing, and with damages still unknown the
        knapsack has nothing to minimize -- so it returns the *cheapest*
        feasible plan, which is the worst possible opening move. Spending the
        surplus makes 'no information' mean 'compress as little as the budget
        forces', which is the right prior.
        """
        dmg = self.damages(key)
        cfgs = list(plan.cfgs)
        cost_of = lambda c: LayerPlan((c,)).per_layer_bytes(
            self.ctx_hint, self.cost.n_kv_heads, self.cost.d_head)
        spent = LayerPlan(tuple(cfgs)).per_layer_bytes(
            self.ctx_hint, self.cost.n_kv_heads, self.cost.d_head)
        while True:
            best = None
            for L in range(self.n_layers):
                i = self.ranked.index(cfgs[L])
                if i + 1 >= len(self.ranked):
                    continue
                extra = cost_of(self.ranked[i + 1]) - cost_of(cfgs[L])
                if spent + extra > self.budget_bytes:
                    continue
                # prefer the layer that hurts most, then the cheapest upgrade
                score = (dmg[L], -extra)
                if best is None or score > best[0]:
                    best = (score, L, extra)
            if best is None:
                return LayerPlan(tuple(cfgs))
            _, L, extra = best
            cfgs[L] = self.ranked[self.ranked.index(cfgs[L]) + 1]
            spent += extra

    def _recover(self, key: str | None) -> None:
        """Back out of a stall by spending everything the budget allows."""
        cheapest = LayerPlan.uniform(self.ranked[0], self.n_layers)
        self.plans[key] = self._spend_surplus(cheapest, key)
        self._stall[key] = 0
        self.n_stall_recoveries += 1

    # -------------------------------------------------------------- planning
    def damages(self, key: str | None = None) -> list[float]:
        """Per-notch value of each layer, shifted so the least valuable is 0.

        The knapsack needs a non-negative cost of *taking a notch away*, and
        only relative order and scale matter to it, so the raw paired-comparison
        values are shifted rather than calibrated into absolute acceptance.
        """
        st = self._stats(key)
        vals = [st.value.get(L, 0.0) for L in range(self.n_layers)]
        floor = min(vals)
        return [v - floor for v in vals]

    def resolve(self, key: str | None = None) -> LayerPlan:
        """Re-solve the knapsack from the current posteriors.

        Probes only measure one notch below the plan, so the damage of a deeper
        cut is extrapolated linearly in notches -- crude, and the reason
        `resolve` runs repeatedly rather than once: each new plan is itself
        probed, so the estimate is refreshed where the allocator actually is.
        """
        st = self._stats(key)
        dmg = self.damages(key)
        prof = SensitivityProfile(self.ranked[-1], max(st.base.mean, 1e-4), self.ranked)
        for L in range(self.n_layers):
            for i, c in enumerate(self.ranked):
                notches = len(self.ranked) - 1 - i
                prof.alpha[(L, c)] = float(np.exp(np.log(prof.alpha_ref) - dmg[L] * notches))
        plan = allocate(prof, self.budget_bytes, self.ctx_hint,
                        self.cost.n_kv_heads, self.cost.d_head, self.n_layers)
        plan = self._spend_surplus(plan, key)
        self.plans[key] = plan
        self.n_resolves += 1
        return plan
