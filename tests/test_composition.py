"""How per-layer damages compose into a mixed plan's damage."""

import numpy as np
import pytest

from draftkv import CompressionConfig, LayerPlan, allocate
from draftkv.layers import SensitivityProfile, fit_composition

CANDS = [CompressionConfig(4, 0.15), CompressionConfig(4, 0.25),
         CompressionConfig(4, 0.5), CompressionConfig(4, 1.0)]
NL = 4


def synthetic(damages, p=1.0):
    prof = SensitivityProfile(CompressionConfig(4, 1.0), 1.0, tuple(CANDS), p=p)
    for (L, c), d in damages.items():
        prof.alpha[(L, c)] = float(np.exp(-d))
    return prof


def _grid(rng, scale=2.0):
    return {(L, c): float(rng.uniform(0, scale) * (len(CANDS) - 1 - i))
            for L in range(NL) for i, c in enumerate(CANDS)}


def test_p_equal_one_is_exactly_log_additive():
    prof = synthetic(_grid(np.random.default_rng(0)), p=1.0)
    plan = LayerPlan(tuple(CANDS[i % len(CANDS)] for i in range(NL)))
    additive = sum(prof.damage(L, c) for L, c in enumerate(plan))
    assert prof.composed_damage(plan) == pytest.approx(additive)
    assert prof.predicted_alpha(plan) == pytest.approx(np.exp(-additive))


def test_larger_p_predicts_less_damage():
    """The direction that matters: overlapping errors, not stacking ones."""
    dmg = _grid(np.random.default_rng(1))
    plan = LayerPlan(tuple(CANDS[0] for _ in range(NL)))     # all layers hurt
    prev = None
    for p in (1.0, 1.5, 2.0, 4.0):
        prof = synthetic(dmg, p=p)
        d = prof.composed_damage(plan)
        if prev is not None:
            assert d < prev
        prev = d


def test_fit_recovers_a_known_exponent():
    rng = np.random.default_rng(2)
    dmg = _grid(rng)
    truth = synthetic(dmg, p=2.5)
    obs = []
    for _ in range(24):
        plan = LayerPlan(tuple(CANDS[int(rng.integers(len(CANDS)))] for _ in range(NL)))
        obs.append((plan, truth.predicted_alpha(plan)))
    prof = synthetic(dmg, p=1.0)
    res = fit_composition(prof, obs)
    assert res["adopted"] and res["p"] == pytest.approx(2.5, abs=0.25)
    assert res["rmse_after"] < res["rmse_before"]
    assert res["loo_after"] < res["loo_before"]


def test_fit_never_does_worse_than_log_additive():
    """p = 1 is in the grid, so the fit can only tie or improve."""
    rng = np.random.default_rng(3)
    dmg = _grid(rng)
    prof = synthetic(dmg, p=1.0)
    obs = [
        (LayerPlan(tuple(CANDS[int(rng.integers(len(CANDS)))] for _ in range(NL))),
         float(rng.uniform(0.05, 1.0)))
        for _ in range(20)
    ]
    res = fit_composition(prof, obs)
    assert res["rmse_after"] <= res["rmse_before"] + 1e-9


def test_fit_on_no_observations_is_a_noop():
    prof = synthetic(_grid(np.random.default_rng(4)), p=1.7)
    res = fit_composition(prof, [])
    assert prof.p == 1.7 and res["n"] == 0 and not res["adopted"]


def test_guard_only_adopts_a_fit_that_cross_validates():
    """The invariant the guard exists to enforce, over many random datasets."""
    rng = np.random.default_rng(11)
    for _ in range(12):
        prof = synthetic(_grid(rng), p=1.0)
        obs = [
            (LayerPlan(tuple(CANDS[int(rng.integers(len(CANDS)))] for _ in range(NL))),
             float(rng.uniform(0.05, 1.0)))
            for _ in range(12)
        ]
        res = fit_composition(prof, obs)
        if res["adopted"]:
            assert res["loo_after"] < res["loo_before"]
            assert prof.p == res["best_p"]
        else:
            assert prof.p == 1.0


def test_noise_around_an_additive_truth_leaves_p_alone():
    """When log-additivity is right and the data is merely noisy, the fit must
    not wander off it. This is the distilgpt2 case: a properly averaged profile
    put additive RMSE (0.098) below the measurement noise (0.13), and the
    exponent bought nothing that survived leave-one-out."""
    rng = np.random.default_rng(12)
    dmg = {(L, c): float(rng.uniform(0, 0.4) * (len(CANDS) - 1 - i))
           for L in range(NL) for i, c in enumerate(CANDS)}
    truth = synthetic(dmg, p=1.0)
    obs = []
    for _ in range(16):
        plan = LayerPlan(tuple(CANDS[int(rng.integers(len(CANDS)))] for _ in range(NL)))
        a = truth.predicted_alpha(plan) * float(np.exp(rng.normal(0, 0.13)))
        obs.append((plan, float(np.clip(a, 1e-3, 1.0))))
    prof = synthetic(dmg, p=1.0)
    fit_composition(prof, obs)
    assert 0.75 <= prof.p <= 1.35, prof.p


def test_objective_preserves_the_damage_ordering():
    """x ** p is monotone for p > 0 and D >= 0, so re-fitting p can never
    reshuffle which config the allocator considers worse within a layer."""
    dmg = _grid(np.random.default_rng(5))
    base = synthetic(dmg, p=1.0)
    for p in (1.0, 2.0, 3.5):
        prof = synthetic(dmg, p=p)
        for L in range(NL):
            order_d = sorted(range(len(CANDS)), key=lambda i: base.damage(L, CANDS[i]))
            order_o = sorted(range(len(CANDS)), key=lambda i: prof.objective(L, CANDS[i]))
            assert order_d == order_o


def test_allocator_still_respects_budget_under_any_p():
    ctx, nkv, dh = 400, 4, 16
    ub = lambda c: LayerPlan.uniform(c, NL).per_layer_bytes(ctx, nkv, dh)
    dmg = _grid(np.random.default_rng(6))
    for p in (1.0, 2.0, 4.0):
        prof = synthetic(dmg, p=p)
        for frac in (0.1, 0.4, 0.8, 1.0):
            budget = ub(CANDS[0]) + frac * (ub(CANDS[-1]) - ub(CANDS[0]))
            plan = allocate(prof, budget, ctx, nkv, dh, NL)
            assert plan.per_layer_bytes(ctx, nkv, dh) <= budget + 1e-6
