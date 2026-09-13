"""The online per-layer allocator.

The static allocator learns a sensitivity table once; this one learns it from
the verify pass while serving. These tests use a simulated environment with a
known sensitive layer, so "did it find the right layer" is checkable exactly.
"""

import numpy as np
import pytest

from draftkv import CompressionConfig, CostModel, LayerPlan
from draftkv.adaptive import AdaptiveLayerController
from draftkv.sim import LayerEnv

NL = 6
CANDS = [CompressionConfig(4, 0.25), CompressionConfig(4, 0.5), CompressionConfig(4, 1.0)]
CTX = 32768


def _cost(resync=1.0e-7):
    """resync_per_token is what decides whether online probing is affordable:
    a probe changes two layers and those layers must be requantized. The
    default here reflects a GPU requantizing a layer of context in well under a
    millisecond; `test_probe_throttle_engages_when_rebuilds_are_expensive`
    covers the opposite regime."""
    return CostModel(n_layers=NL, n_kv_heads=12, d_head=64, weight_bytes=2.0e9,
                     full_kv_bandwidth=6.4e10, hbm_kv_budget=4.0e8,
                     resync_per_token=resync)


def _budget(frac: float) -> float:
    cm = _cost()
    lo = LayerPlan.uniform(CANDS[0], NL).per_layer_bytes(CTX, cm.n_kv_heads, cm.d_head)
    hi = LayerPlan.uniform(CANDS[-1], NL).per_layer_bytes(CTX, cm.n_kv_heads, cm.d_head)
    return lo + frac * (hi - lo)


def _run(ctrl, env, rounds, key=None):
    for _ in range(rounds):
        plan, gamma = ctrl.select(env.ctx, key)
        if gamma > 0:
            ctrl.update(plan, gamma, env.pull(plan, gamma), env.ctx, key)
        else:
            ctrl.update(plan, 0, 0, env.ctx, key)
    return ctrl.plans.get(key)


def test_finds_the_sensitive_layer_without_any_offline_profile():
    cm = _cost()
    env = LayerEnv(damage={2: 2.5}, ranked=CANDS, ctx=CTX, seed=1)
    ctrl = AdaptiveLayerController(cm, NL, _budget(0.5), CANDS, ctx_hint=CTX,
                                   probe_prob=0.4, resolve_every=40, seed=2)
    _run(ctrl, env, 900)
    d = ctrl.damages()
    assert int(np.argmax(d)) == 2, d
    # and the plan protects it: layer 2 is not the one starved
    plan = ctrl.plans[None]
    assert CANDS.index(plan[2]) >= max(CANDS.index(plan[L]) for L in range(NL) if L != 2)


def test_never_exceeds_the_byte_budget():
    cm = _cost()
    env = LayerEnv(damage={0: 1.5, 4: 0.8}, ranked=CANDS, ctx=CTX, seed=3)
    for frac in (0.0, 0.3, 0.7, 1.0):
        budget = _budget(frac)
        ctrl = AdaptiveLayerController(cm, NL, budget, CANDS, ctx_hint=CTX,
                                       probe_prob=0.3, resolve_every=30, seed=4)
        for _ in range(300):
            plan, gamma = ctrl.select(env.ctx)
            # probes may spend *less*, never more
            assert plan.per_layer_bytes(CTX, cm.n_kv_heads, cm.d_head) <= budget + 1e-6
            ctrl.update(plan, gamma, env.pull(plan, gamma) if gamma else 0, env.ctx)


def test_tracks_a_shift_that_a_static_profile_would_miss():
    """The case the real-model work exposed: the sensitive layer is a property
    of the content, so a profile measured once goes stale."""
    cm = _cost()
    env = LayerEnv(damage={1: 2.5}, ranked=CANDS, ctx=CTX, seed=5)
    ctrl = AdaptiveLayerController(cm, NL, _budget(0.5), CANDS, ctx_hint=CTX,
                                   probe_prob=0.4, resolve_every=40, decay=0.97, seed=6)
    _run(ctrl, env, 800)
    assert int(np.argmax(ctrl.damages())) == 1

    env.shift({4: 2.5})
    _run(ctrl, env, 1200)
    assert int(np.argmax(ctrl.damages())) == 4, ctrl.damages()


def test_content_keys_hold_independent_plans():
    cm = _cost()
    ctrl = AdaptiveLayerController(cm, NL, _budget(0.5), CANDS, ctx_hint=CTX,
                                   probe_prob=0.4, resolve_every=40, seed=7)
    _run(ctrl, LayerEnv(damage={0: 2.5}, ranked=CANDS, ctx=CTX, seed=8), 700, key="prose")
    _run(ctrl, LayerEnv(damage={5: 2.5}, ranked=CANDS, ctx=CTX, seed=9), 700, key="code")
    assert int(np.argmax(ctrl.damages("prose"))) == 0
    assert int(np.argmax(ctrl.damages("code"))) == 5


def test_probe_rate_stays_near_its_setting():
    """Exploration is free in correctness -- output is identical whatever the
    plan -- but not in cost, so it must stay bounded."""
    cm = _cost()
    env = LayerEnv(damage={3: 1.0}, ranked=CANDS, ctx=CTX, seed=10)
    ctrl = AdaptiveLayerController(cm, NL, _budget(0.6), CANDS, ctx_hint=CTX,
                                   probe_prob=0.1, resolve_every=50, seed=11)
    _run(ctrl, env, 1000)
    assert 0.02 <= ctrl.n_probes / ctrl.n_decisions <= 0.18


def test_probe_throttle_engages_when_rebuilds_are_expensive():
    """The operating envelope, stated as a test.

    A probe changes the plan and the changed layers must be requantized. When
    that costs more than the round it rides on -- very long context, or slow
    requantization -- the throttle shuts probing down rather than spending the
    throughput it is trying to earn, and the allocator degrades to a static
    plan. That is the correct behaviour, and the boundary should be visible.
    """
    cheap = AdaptiveLayerController(_cost(1.0e-8), NL, _budget(0.6), CANDS,
                                    ctx_hint=CTX, probe_prob=0.3, seed=1)
    dear = AdaptiveLayerController(_cost(2.0e-6), NL, _budget(0.6), CANDS,
                                   ctx_hint=CTX, probe_prob=0.3, seed=1)
    assert cheap.effective_probe_prob(CTX) == pytest.approx(0.3)
    assert dear.effective_probe_prob(CTX) < 0.05
    # and short context is affordable even with slow requantization
    assert dear.effective_probe_prob(256) > dear.effective_probe_prob(CTX)


def test_only_changed_layers_are_rebuilt():
    """Rebuilding all layers for a two-layer swap is what made probing look
    unaffordable in the first place."""
    from draftkv import DraftKVEngine
    from draftkv.compress import CompressedKVCache, FullKVCache
    from draftkv.model import demo_model

    m = demo_model()
    toks = np.arange(64) % m.vocab_size
    full = FullKVCache(m.n_layers)
    m.forward(toks, full, 0)
    plan_a = LayerPlan.uniform(CANDS[-1], m.n_layers)
    comp = CompressedKVCache(m.n_layers, plan_a)
    comp.sync_from(full, 0)
    before = [comp.read(L)[0].copy() for L in range(m.n_layers)]

    cfgs = list(plan_a.cfgs)
    cfgs[1] = CANDS[0]
    rows = comp.rebuild_layers(full, [1], cfgs)
    assert rows == 64
    assert comp.cfgs[1] == CANDS[0]
    for L in range(m.n_layers):
        same = np.array_equal(comp.read(L)[0], before[L])
        assert same == (L != 1), L


def test_probing_cannot_change_the_emitted_tokens():
    """The property that makes online exploration legitimate at all."""
    from draftkv import DraftKVEngine
    from draftkv.model import demo_model

    m = demo_model()
    rng = np.random.default_rng(0)
    prompt = [int(x) for x in rng.integers(0, m.vocab_size, 192)]
    base = DraftKVEngine(m, seed=0).generate_baseline(prompt, 24, 0.0)
    cm = CostModel(n_layers=m.n_layers, n_kv_heads=m.n_heads, d_head=m.d_head,
                   weight_bytes=2.0e6, bandwidth=2.0e10, full_kv_bandwidth=6.4e8,
                   hbm_kv_budget=2.0e5)
    ctrl = AdaptiveLayerController(cm, m.n_layers, _budget(0.5), CANDS,
                                   ctx_hint=256, probe_prob=0.9, resolve_every=5, seed=3)
    r = DraftKVEngine(m, ctrl, seed=0).generate(prompt, 24, 0.0)
    assert r.tokens == base
