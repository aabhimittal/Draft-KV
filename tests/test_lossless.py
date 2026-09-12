"""The property the whole design rests on: compression cannot change output."""

import numpy as np
import pytest

from draftkv import CompressionConfig, DraftKVEngine, ThompsonController, CostModel, default_arms
from draftkv.model import demo_model

CONFIGS = [
    CompressionConfig(16, 1.0),
    CompressionConfig(8, 1.0),
    CompressionConfig(4, 1.0),
    CompressionConfig(4, 0.5),
    CompressionConfig(3, 0.5),
    CompressionConfig(2, 0.25),
]


@pytest.fixture(scope="module")
def setup():
    m = demo_model()
    rng = np.random.default_rng(0)
    prompt = [int(x) for x in rng.integers(0, m.vocab_size, 192)]
    base = DraftKVEngine(m, seed=0).generate_baseline(prompt, 32, 0.0)
    return m, prompt, base


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: c.label())
@pytest.mark.parametrize("gamma", [1, 3, 6])
def test_greedy_output_identical(setup, cfg, gamma):
    from draftkv import FixedPolicy

    m, prompt, base = setup
    r = DraftKVEngine(m, FixedPolicy(cfg, gamma), seed=0).generate(prompt, 32, 0.0)
    assert r.tokens == base, f"{cfg.label()} gamma={gamma} diverged from full-KV decoding"


def test_aggressive_compression_actually_costs_acceptance(setup):
    """Guards against a vacuous test suite: if compression changed nothing, the
    identity above would prove nothing."""
    from draftkv import FixedPolicy

    m, prompt, _ = setup
    mild = DraftKVEngine(m, FixedPolicy(CompressionConfig(16, 1.0), 6), seed=0).generate(prompt, 32, 0.0)
    harsh = DraftKVEngine(m, FixedPolicy(CompressionConfig(2, 0.25), 6), seed=0).generate(prompt, 32, 0.0)
    assert mild.acceptance_rate > harsh.acceptance_rate + 0.2


def test_controller_driven_run_is_lossless(setup):
    m, prompt, base = setup
    cm = CostModel(n_layers=m.n_layers, n_kv_heads=m.n_heads, d_head=m.d_head,
                   weight_bytes=2.0e6, bandwidth=2.0e10, full_kv_bandwidth=6.4e8,
                   hbm_kv_budget=2.0e5)
    ctrl = ThompsonController(cm, default_arms(), seed=3)
    r = DraftKVEngine(m, ctrl, seed=0).generate(prompt, 32, 0.0)
    assert r.tokens == base
    assert r.verify_passes <= r.new_tokens


def test_resync_prevents_drift(setup):
    """Drafted K/V are computed under a compressed context, so they are not the
    true K/V. If `sync_from` were skipped the error would compound; this test
    pins that the rollback point is the last committed position."""
    from draftkv.compress import CompressedKVCache, FullKVCache

    m, prompt, _ = setup
    full = FullKVCache(m.n_layers)
    m.forward(np.array(prompt), full, 0)
    comp = CompressedKVCache(m.n_layers, CompressionConfig(8, 1.0))
    comp.sync_from(full, 0)
    assert comp.length == full.length

    comp.truncate(10)
    assert comp.length == 10
    rows = comp.sync_from(full, 10)
    assert rows == (len(prompt) - 10) * m.n_layers
    assert comp.length == len(prompt)
