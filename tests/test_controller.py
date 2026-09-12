import numpy as np
import pytest

from draftkv import CompressionConfig, CostModel, FlatBandit, OraclePolicy, ThompsonController, default_arms
from draftkv.sim import BanditEnv, plausible_alpha, run_bandit


def _cost():
    return CostModel(full_kv_bandwidth=6.4e10, hbm_kv_budget=1.0e9)


def _env(seed=3, difficulty=1.0):
    arms = default_arms()
    return BanditEnv({c: plausible_alpha(c, difficulty) for c in arms}, ctx=32768, seed=seed)


def test_converges_near_oracle():
    cm, env = _cost(), _env()
    out = run_bandit(ThompsonController(cm, default_arms(), seed=1), _env(), cm, 2000)
    assert out["mean_throughput"] / out["optimal_throughput"] > 0.90


def test_regret_is_sublinear():
    cm = _cost()
    ctrl = ThompsonController(cm, default_arms(), seed=1)
    env = _env()
    first = run_bandit(ctrl, env, cm, 500)["normalized_regret"]
    later = run_bandit(ctrl, env, cm, 500)["normalized_regret"]
    assert later < first


def test_factored_beats_flat_product_bandit():
    """The design claim: gamma should be solved, not explored."""
    cm = _cost()
    tho = run_bandit(ThompsonController(cm, default_arms(), seed=1), _env(), cm, 600)
    flat = run_bandit(FlatBandit(cm, default_arms(), seed=1), _env(), cm, 600)
    assert tho["normalized_regret"] < flat["normalized_regret"]


def test_switch_penalty_reduces_thrashing():
    """Pricing the rebuild in the objective should cut config switches without
    costing throughput -- unlike a hard hold, which cuts switches by refusing
    to learn (see `test_hard_commitment_is_not_free`)."""
    cm = _cost()
    priced = run_bandit(ThompsonController(cm, default_arms(), switch_amortize=8, seed=1), _env(), cm, 800)
    free = run_bandit(ThompsonController(cm, default_arms(), switch_amortize=10**9, seed=1), _env(), cm, 800)
    assert priced["switches"] <= free["switches"]
    assert priced["mean_throughput"] >= free["mean_throughput"] * 0.98


def test_hard_commitment_is_not_free():
    """Negative result, kept because it shaped the default: forcing the
    controller to hold a config for 32 rounds cuts switching but loses more to
    slower convergence than it saves. commit_rounds therefore defaults to 1."""
    cm = _cost()
    held = run_bandit(ThompsonController(cm, default_arms(), commit_rounds=32, seed=1), _env(), cm, 800)
    free = run_bandit(ThompsonController(cm, default_arms(), commit_rounds=1, seed=1), _env(), cm, 800)
    assert held["switches"] < free["switches"]
    assert held["mean_throughput"] < free["mean_throughput"]


def test_matches_hindsight_best_fixed_policy_without_knowing_alpha():
    """The honest version of "beats a fixed config".

    In a *stationary* environment a fixed arm chosen with oracle knowledge is
    nearly optimal by construction, and the bandit cannot beat it -- it pays
    an exploration tax up front. Two separate claims, both checked:

      * from cold, it already beats a config picked blind by a wide margin;
      * after warm-up it converges onto the hindsight-best arm.
    """
    from draftkv import FixedPolicy

    cm = _cost()
    fixed = [
        run_bandit(FixedPolicy(c, g), _env(), cm, 1500)["mean_throughput"]
        for c in default_arms()
        for g in (2, 4, 8)
    ]
    ctrl = ThompsonController(cm, default_arms(), seed=1)
    cold = run_bandit(ctrl, _env(), cm, 1500)
    assert cold["mean_throughput"] > float(np.mean(fixed)) * 1.2

    run_bandit(ctrl, _env(), cm, 1500)                 # warm up
    warm = run_bandit(ctrl, _env(), cm, 1000)
    assert warm["mean_throughput"] > max(fixed) * 0.95


def test_gates_off_on_short_context():
    """Batch 1, HBM-resident cache, 256-token context: drafting cannot pay, and
    the controller must decline rather than burn cycles."""
    cm = CostModel()
    ctrl = ThompsonController(cm, default_arms(), seed=1)
    assert all(ctrl.select(256)[1] == 0 for _ in range(50))


def test_tracks_a_content_shift():
    """Forgetting only matters when the shift moves the argmax.

    The budget here is tight enough that tolerant content prefers an
    aggressive arm (4b/keep0.25) while brittle content prefers a mild one
    (8b/keep1), so a stale posterior is actively wrong rather than merely
    stale.
    """
    cm = CostModel(full_kv_bandwidth=6.4e10, hbm_kv_budget=3.0e8)
    arms = default_arms()
    easy = {c: plausible_alpha(c, 0.3) for c in arms}
    hard = {c: plausible_alpha(c, 2.0) for c in arms}
    assert BanditEnv(easy, 32768).true_best(cm)[0] != BanditEnv(hard, 32768).true_best(cm)[0]
    tracking = ThompsonController(cm, arms, decay=0.97, seed=2)
    stale = ThompsonController(cm, arms, decay=1.0, seed=2)
    for ctrl in (tracking, stale):
        run_bandit(ctrl, BanditEnv(easy, 32768, seed=4), cm, 1500)
    res = {}
    for name, ctrl in (("track", tracking), ("stale", stale)):
        res[name] = run_bandit(ctrl, BanditEnv(hard, 32768, seed=5), cm, 800)
    frac = {k: v["mean_throughput"] / v["optimal_throughput"] for k, v in res.items()}
    assert frac["track"] > frac["stale"] + 0.1


def test_contextual_posteriors_are_separate():
    cm = _cost()
    ctrl = ThompsonController(cm, default_arms(), seed=1)
    cfg = CompressionConfig(4, 0.5)
    for _ in range(50):
        ctrl.update(cfg, 8, 8, 32768, key="prose")
        ctrl.update(cfg, 8, 0, 32768, key="code")
    assert ctrl.alpha_estimates("prose")[cfg] > 0.8
    assert ctrl.alpha_estimates("code")[cfg] < 0.2
