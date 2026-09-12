"""Per-layer allocation.

Most of these use a synthetic `SensitivityProfile`: the allocator is an
optimizer, and its contract (never exceed the budget, never score worse than
uniform under its own objective) is exactly checkable without paying for model
runs.  One integration test measures a real plan, because an optimizer that
optimizes the wrong thing would pass every synthetic test.
"""

import numpy as np
import pytest

from draftkv import CompressionConfig, CostModel, LayerPlan, allocate, measure_alpha, profile_layers
from draftkv.compress import CompressedKVCache, FullKVCache
from draftkv.layers import SensitivityProfile, allocate_greedy
from draftkv.model import demo_model

CANDS = [CompressionConfig(2, 0.5), CompressionConfig(3, 0.5),
         CompressionConfig(4, 1.0), CompressionConfig(8, 1.0)]
CTX, NKV, DH, NL = 400, 4, 16, 3
UB = lambda c: LayerPlan.uniform(c, NL).per_layer_bytes(CTX, NKV, DH)


def synthetic(damages: dict[tuple[int, CompressionConfig], float]) -> SensitivityProfile:
    """Build a profile directly from stipulated per-layer damages."""
    prof = SensitivityProfile(CompressionConfig(16, 1.0), 1.0, tuple(CANDS))
    for (L, c), d in damages.items():
        prof.alpha[(L, c)] = float(np.exp(-d))
    return prof


@pytest.fixture(scope="module")
def measured():
    m = demo_model()
    rng = np.random.default_rng(0)
    prompt = [int(x) for x in rng.integers(0, m.vocab_size, 256)]
    return m, prompt, profile_layers(m, prompt, CANDS, n_new=160)


# ----------------------------------------------------------------- plumbing
def test_plan_bytes_sum_per_layer_not_averaged():
    mixed = LayerPlan((CompressionConfig(2, 0.25), CompressionConfig(16, 1.0)))
    exact = sum(LayerPlan((c,)).per_layer_bytes(1024, 8, 128) for c in mixed)
    assert mixed.per_layer_bytes(1024, 8, 128) == pytest.approx(exact)


def test_cost_model_prices_a_plan():
    cm = CostModel(n_layers=2, n_kv_heads=8, d_head=128)
    plan = LayerPlan.uniform(CompressionConfig(4, 1.0), 2)
    assert cm.draft_kv_bytes(plan, 1024) == pytest.approx(
        cm.draft_kv_bytes(CompressionConfig(4, 1.0), 1024)
    )


def test_per_layer_cache_quantizes_each_layer_independently():
    m = demo_model()
    prompt = np.random.default_rng(0).integers(0, m.vocab_size, 64)
    full = FullKVCache(m.n_layers)
    m.forward(prompt, full, 0)
    plan = LayerPlan(tuple(
        CompressionConfig(16, 1.0) if L == 0 else CompressionConfig(2, 1.0)
        for L in range(m.n_layers)
    ))
    comp = CompressedKVCache(m.n_layers, plan)
    comp.sync_from(full, 0)
    assert np.allclose(comp.read(0)[0], full.read(0)[0])          # layer 0 untouched
    assert not np.allclose(comp.read(1)[0], full.read(1)[0])      # layer 1 quantized


# ---------------------------------------------------------------- optimizer
def test_allocation_never_exceeds_budget():
    prof = synthetic({(L, c): 2.0 / (1 + i) for L in range(NL) for i, c in enumerate(CANDS)})
    for frac in np.linspace(0.0, 1.2, 13):
        budget = UB(CANDS[0]) + frac * (UB(CANDS[-1]) - UB(CANDS[0]))
        plan = allocate(prof, budget, CTX, NKV, DH, NL)
        assert plan.per_layer_bytes(CTX, NKV, DH) <= max(budget, UB(CANDS[0])) + 1e-6


def test_allocation_finds_an_exactly_affordable_uniform_plan():
    """Regression for a quantization bug.

    With a byte-bucketed DP, three layers each rounding up by a third of a
    bucket read as one bucket over budget, so a uniform plan costing precisely
    the budget was rejected and a strictly worse plan returned. Exact-cost
    Pareto DP must recover it.
    """
    prof = synthetic({(L, c): 3.0 - i for L in range(NL) for i, c in enumerate(CANDS)})
    plan = allocate(prof, UB(CANDS[2]), CTX, NKV, DH, NL)
    assert all(c == CANDS[2] for c in plan), plan.label()


def test_allocation_never_scores_worse_than_uniform_under_its_own_objective():
    rng = np.random.default_rng(0)
    for _ in range(20):
        prof = synthetic({
            (L, c): float(rng.uniform(0, 3) * (len(CANDS) - i))
            for L in range(NL) for i, c in enumerate(CANDS)
        })
        for frac in (0.2, 0.5, 0.8, 1.0):
            budget = UB(CANDS[0]) + frac * (UB(CANDS[-1]) - UB(CANDS[0]))
            plan = allocate(prof, budget, CTX, NKV, DH, NL)
            dmg = sum(prof.damage(L, c) for L, c in enumerate(plan))
            for c in CANDS:
                if UB(c) <= budget + 1e-9:
                    uni = sum(prof.damage(L, c) for L in range(NL))
                    assert dmg <= uni + 1e-9


def test_exact_dp_beats_greedy_on_its_failure_mode():
    """Greedy climbs one cheap step at a time, so a layer whose entire value sits
    in its top config is unreachable: every intermediate step scores zero gain.
    It spends the budget on the two gently-graded layers instead and strands the
    one that mattered at 2-bit. Exact DP sees the whole assignment at once."""
    prof = synthetic({
        (0, CANDS[0]): 3.0, (0, CANDS[1]): 3.0, (0, CANDS[2]): 3.0, (0, CANDS[3]): 0.0,
        (1, CANDS[0]): 1.0, (1, CANDS[1]): 0.7, (1, CANDS[2]): 0.4, (1, CANDS[3]): 0.3,
        (2, CANDS[0]): 1.0, (2, CANDS[1]): 0.7, (2, CANDS[2]): 0.4, (2, CANDS[3]): 0.3,
    })
    per = lambda c: LayerPlan((c,)).per_layer_bytes(CTX, NKV, DH)
    budget = per(CANDS[3]) + 2 * per(CANDS[0])
    dp = allocate(prof, budget, CTX, NKV, DH, NL)
    gr = allocate_greedy(prof, budget, CTX, NKV, DH, NL)
    d_dp = sum(prof.damage(L, c) for L, c in enumerate(dp))
    d_gr = sum(prof.damage(L, c) for L, c in enumerate(gr))
    assert d_dp < d_gr
    assert dp[0] == CANDS[3] and gr[0] == CANDS[0]


def test_starvation_budget_returns_cheapest_plan():
    prof = synthetic({(L, c): 1.0 for L in range(NL) for c in CANDS})
    plan = allocate(prof, 1.0, CTX, NKV, DH, NL)
    assert all(c == CANDS[0] for c in plan)


# -------------------------------------------------------------- integration
def test_layer_sensitivity_is_not_uniform(measured):
    """If every layer cost the same bits, allocation would be pointless."""
    m, _, prof = measured
    alphas = [prof.alpha[(L, CANDS[0])] for L in range(m.n_layers)]
    assert max(alphas) - min(alphas) > 0.05


def test_allocated_plan_beats_uniform_on_measured_acceptance(measured):
    m, prompt, prof = measured
    budget = UB(CANDS[0]) + 0.7 * (UB(CANDS[-1]) - UB(CANDS[0]))
    plan = allocate(prof, budget, CTX, NKV, DH, m.n_layers)
    best_uniform = max([c for c in CANDS if UB(c) <= budget], key=UB)
    a_plan = measure_alpha(m, prompt, plan, n_new=160)
    a_uni = measure_alpha(m, prompt, LayerPlan.uniform(best_uniform, m.n_layers), n_new=160)
    assert a_plan > a_uni
