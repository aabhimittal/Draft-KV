"""The depth-indexed acceptance model, and the bias it removes."""

import numpy as np
import pytest

from draftkv import (
    CompressionConfig,
    CostModel,
    DepthThompsonController,
    ThompsonController,
    best_gamma,
    best_gamma_profile,
    default_arms,
    expected_tokens,
    expected_tokens_profile,
)
from draftkv.sim import BanditEnv, plausible_alpha, rising_profile, run_bandit


def _cost():
    return CostModel(full_kv_bandwidth=6.4e10, hbm_kv_budget=1.0e9)


@pytest.mark.parametrize("alpha", [0.0, 0.3, 0.75, 1.0])
@pytest.mark.parametrize("gamma", [1, 4, 9])
def test_profile_reduces_to_geometric(alpha, gamma):
    """A flat profile must reproduce (1 - a^(g+1))/(1 - a) exactly."""
    assert expected_tokens_profile([alpha] * gamma) == pytest.approx(
        expected_tokens(alpha, gamma), abs=1e-12
    )


def test_rising_profile_yields_more_tokens_than_its_first_rate():
    """The whole point: pooling a rising profile into one alpha understates E."""
    prof = rising_profile(0.5, 8)
    assert prof[0] < prof[-1]
    assert expected_tokens_profile(prof) > expected_tokens(prof[0], 8)


def test_profile_matches_monte_carlo():
    rng = np.random.default_rng(0)
    prof = rising_profile(0.4, 6)
    draws = []
    for _ in range(20000):
        k = 0
        while k < len(prof) and rng.random() < prof[k]:
            k += 1
        draws.append(k + 1)
    assert np.mean(draws) == pytest.approx(expected_tokens_profile(prof), abs=0.05)


def test_iid_model_truncates_gamma_on_rising_data():
    """Regression guard on the bias, not just on the arithmetic."""
    cm, cfg = _cost(), CompressionConfig(4, 0.5)
    prof = rising_profile(0.51, 12)
    g_prof, _ = best_gamma_profile(cm, cfg, prof, 32768)
    g_iid, _ = best_gamma(cm, cfg, prof[0], 32768, 12)
    assert g_prof > g_iid
    # and the short draft really is worse under the true profile
    assert cm.throughput_profile(cfg, prof[:g_prof], 32768) > cm.throughput_profile(
        cfg, prof[:g_iid], 32768
    )


def test_depth_controller_beats_pooled_controller():
    cm, arms = _cost(), default_arms()
    alphas = {c: plausible_alpha(c) for c in arms}
    profiles = {c: rising_profile(alphas[c], 12) for c in arms}
    env = lambda: BanditEnv(alphas, 32768, seed=3, profiles=profiles)
    depth = run_bandit(DepthThompsonController(cm, arms, gamma_max=12, seed=1), env(), cm, 1500)
    pooled = run_bandit(ThompsonController(cm, arms, gamma_max=12, seed=1), env(), cm, 1500)
    assert depth["mean_throughput"] > pooled["mean_throughput"]


def test_depth_controller_is_no_worse_on_flat_data():
    """It must not pay for its extra parameters when the i.i.d. model is right."""
    cm, arms = _cost(), default_arms()
    alphas = {c: plausible_alpha(c) for c in arms}
    env = lambda: BanditEnv(alphas, 32768, seed=3)          # genuinely flat
    depth = run_bandit(DepthThompsonController(cm, arms, gamma_max=8, seed=1), env(), cm, 1500)
    pooled = run_bandit(ThompsonController(cm, arms, gamma_max=8, seed=1), env(), cm, 1500)
    assert depth["mean_throughput"] > pooled["mean_throughput"] * 0.95


def test_depth_controller_still_gates_off_short_context():
    ctrl = DepthThompsonController(CostModel(), default_arms(), seed=1)
    assert all(ctrl.select(256)[1] == 0 for _ in range(50))
