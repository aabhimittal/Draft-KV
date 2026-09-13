"""Block-paged full cache: the rollback semantics a vLLM integration needs.

DRAFTKV rewinds the authoritative cache once per verify pass. On a contiguous
array that is a slice; in a paged pool it means freeing whole blocks, keeping a
partially filled tail, and doing so once per round without leaking. These tests
pin that behaviour against the contiguous cache before any GPU code exists.
"""

import numpy as np
import pytest

from draftkv import CompressionConfig, DraftKVEngine, FixedPolicy
from draftkv.compress import FullKVCache
from draftkv.model import demo_model
from draftkv.paged import BlockPool, BlockPoolExhausted, PagedKVCache, plan_blocks_needed


def _paged(m, block_size=16, n_blocks=512):
    return PagedKVCache(m.n_layers, m.n_heads, m.d_head, block_size, n_blocks)


def test_pool_allocate_and_free_keep_invariants():
    pool = BlockPool(8, 16)
    got = [pool.allocate() for _ in range(5)]
    pool.check_invariants()
    assert pool.n_used == 5 and pool.n_free == 3
    for b in got:
        pool.free(b)
    pool.check_invariants()
    assert pool.n_used == 0 and pool.n_free == 8


def test_pool_rejects_double_free_and_exhaustion():
    pool = BlockPool(2, 16)
    a, b = pool.allocate(), pool.allocate()
    with pytest.raises(BlockPoolExhausted):
        pool.allocate()
    pool.free(a)
    with pytest.raises(RuntimeError, match="double free"):
        pool.free(a)
    pool.free(b)


@pytest.mark.parametrize("block_size", [1, 4, 16, 64])
def test_paged_matches_contiguous_cache(block_size):
    m = demo_model()
    toks = np.random.default_rng(0).integers(0, m.vocab_size, 100)
    full, paged = FullKVCache(m.n_layers), _paged(m, block_size)
    lf = m.forward(toks, full, 0)
    lp = m.forward(toks, paged, 0)
    assert np.allclose(lf, lp)
    for L in range(m.n_layers):
        kf, vf, pf = full.read(L)
        kp, vp, pp = paged.read(L)
        assert np.allclose(kf, kp) and np.allclose(vf, vp) and np.array_equal(pf, pp)


def test_truncate_frees_blocks_and_keeps_the_tail():
    m = demo_model()
    toks = np.random.default_rng(1).integers(0, m.vocab_size, 64)
    paged = _paged(m, block_size=16)
    m.forward(toks, paged, 0)
    assert paged.length == 64
    before = paged.blocks_in_use
    paged.truncate(33)                       # 33 tokens -> ceil(33/16) = 3 blocks/layer
    paged.pool.check_invariants()
    assert paged.length == 33
    assert paged.blocks_in_use == 3 * m.n_layers < before


def test_repeated_rollback_cycles_do_not_leak():
    """One rollback per round is the steady state; a leak of one block per
    round would exhaust any pool."""
    m = demo_model()
    rng = np.random.default_rng(2)
    paged = _paged(m, block_size=8, n_blocks=256)
    m.forward(rng.integers(0, m.vocab_size, 40), paged, 0)
    baseline = paged.blocks_in_use
    for _ in range(60):
        n = paged.length
        m.forward(rng.integers(0, m.vocab_size, 6), paged, n)
        paged.truncate(n)                    # reject every drafted token
        paged.pool.check_invariants()
        assert paged.length == n
    assert paged.blocks_in_use == baseline


def test_pool_exhaustion_raises_instead_of_corrupting():
    m = demo_model()
    paged = _paged(m, block_size=8, n_blocks=m.n_layers)   # one block per layer
    with pytest.raises(BlockPoolExhausted):
        m.forward(np.arange(64) % m.vocab_size, paged, 0)


def test_blocks_needed_matches_actual_allocation():
    m = demo_model()
    for ctx in (1, 15, 16, 17, 100):
        paged = _paged(m, block_size=16)
        m.forward(np.arange(ctx) % m.vocab_size, paged, 0)
        assert paged.blocks_in_use == plan_blocks_needed(ctx, m.n_layers, 16)


def test_losslessness_holds_with_a_paged_authoritative_cache():
    """The end-to-end claim: swapping the full cache for a paged pool, with a
    rollback every round, must not change a single emitted token."""
    m = demo_model()
    rng = np.random.default_rng(3)
    prompt = [int(x) for x in rng.integers(0, m.vocab_size, 192)]
    base = DraftKVEngine(m, seed=0).generate_baseline(prompt, 32, 0.0)
    for block_size in (4, 16):
        eng = DraftKVEngine(
            m, FixedPolicy(CompressionConfig(4, 0.5), 5), seed=0,
            full_cache_factory=lambda bs=block_size: _paged(m, bs, n_blocks=2048),
        )
        r = eng.generate(prompt, 32, 0.0)
        assert r.tokens == base, f"block_size={block_size}"


def test_fragmentation_is_reported_and_bounded_by_block_size():
    m = demo_model()
    paged = _paged(m, block_size=16)
    m.forward(np.arange(33) % m.vocab_size, paged, 0)
    # 33 tokens in 3 blocks of 16 => 48 slots, 15 wasted
    assert paged.fragmentation() == pytest.approx(1 - 33 / 48)
