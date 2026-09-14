"""Modern architectures: RoPE, grouped-query attention, RMSNorm, SwiGLU.

Everything before this was GPT-2 family. The structural tests here run on a
random-weight Llama-style model so they work in CI with no download; the tests
that need real weights skip cleanly without them.
"""

from __future__ import annotations

import numpy as np
import pytest

from draftkv import CompressionConfig, CostModel, DraftKVEngine, FixedPolicy
from draftkv.compress import CompressedKVCache, FullKVCache
from draftkv.llama import LlamaLike, _rotate_half, default_llama_dir, demo_llama, load_llama
from draftkv.paged import PagedKVCache

needs_ckpt = pytest.mark.skipif(
    default_llama_dir() is None,
    reason="no SmolLM2 checkpoint (scripts/fetch_model.py --model HuggingFaceTB/SmolLM2-135M)",
)


# ------------------------------------------------------------------ RoPE
def test_rope_preserves_norm():
    m = demo_llama()
    cos, sin = m._rope(np.arange(8))
    x = np.random.default_rng(0).standard_normal((8, 2, m.d_head)).astype(np.float32)
    y = x * cos[:, None, :] + _rotate_half(x) * sin[:, None, :]
    assert np.allclose(np.linalg.norm(x, axis=-1), np.linalg.norm(y, axis=-1), atol=1e-4)


def test_rope_dot_product_depends_only_on_relative_position():
    """The property RoPE exists for. If this holds, the rotation is applied
    correctly; if the halves are swapped it still preserves norms but fails
    here."""
    m = demo_llama()
    rng = np.random.default_rng(1)
    q = rng.standard_normal(m.d_head).astype(np.float32)
    k = rng.standard_normal(m.d_head).astype(np.float32)

    def score(i, j):
        cos, sin = m._rope(np.array([i, j]))
        qi = q * cos[0] + _rotate_half(q) * sin[0]
        kj = k * cos[1] + _rotate_half(k) * sin[1]
        return float(qi @ kj)

    for delta in (1, 3, 7):
        vals = [score(p, p + delta) for p in (0, 5, 11, 40)]
        assert np.allclose(vals, vals[0], atol=1e-3), (delta, vals)


# ------------------------------------------------------------------- GQA
@pytest.mark.parametrize("n_heads,n_kv", [(8, 8), (8, 4), (8, 2), (8, 1)])
def test_cache_stores_kv_heads_not_query_heads(n_heads, n_kv):
    """GQA is the architectural change that matters here: the cache -- the
    thing DRAFTKV compresses -- is n_kv_heads wide, not n_heads."""
    m = demo_llama(n_heads=n_heads, n_kv_heads=n_kv)
    cache = m.new_cache()
    m.forward(np.arange(32) % m.vocab_size, cache, 0)
    k, v, pos = cache.read(0)
    assert k.shape == (32, n_kv, m.d_head)
    assert v.shape == (32, n_kv, m.d_head)


@pytest.mark.parametrize("n_kv", [8, 4, 1])
def test_losslessness_on_a_modern_architecture(n_kv):
    m = demo_llama(n_heads=8, n_kv_heads=n_kv, seed=2)
    prompt = [int(x) for x in np.random.default_rng(0).integers(0, m.vocab_size, 96)]
    base = DraftKVEngine(m, seed=0).generate_baseline(prompt, 20, 0.0)
    for cfg in (CompressionConfig(8, 1.0), CompressionConfig(4, 0.5),
                CompressionConfig(2, 0.25)):
        r = DraftKVEngine(m, FixedPolicy(cfg, 4), seed=0).generate(prompt, 20, 0.0)
        assert r.tokens == base, (n_kv, cfg.label())


def test_paged_cache_works_under_gqa():
    """The paged pool sizes itself from the cache's head count, which GQA
    changes."""
    m = demo_llama(n_heads=8, n_kv_heads=2)
    prompt = [int(x) for x in np.random.default_rng(4).integers(0, m.vocab_size, 96)]
    base = DraftKVEngine(m, seed=0).generate_baseline(prompt, 16, 0.0)
    r = DraftKVEngine(
        m, FixedPolicy(CompressionConfig(4, 0.5), 4), seed=0,
        full_cache_factory=lambda: PagedKVCache(m.n_layers, m.n_kv_heads, m.d_head, 16, 1024),
    ).generate(prompt, 16, 0.0)
    assert r.tokens == base


def test_attention_bias_path_runs():
    """Qwen2 puts a bias on q/k/v; Llama does not. Both must load."""
    m = demo_llama(attn_bias=True)
    out = m.forward(np.arange(8) % m.vocab_size, m.new_cache(), 0)
    assert out.shape == (8, m.vocab_size) and np.isfinite(out).all()


# --------------------------------------------------- GQA moves the ceiling
def test_gqa_pushes_the_breakeven_context_out_by_its_ratio():
    """The practical consequence of GQA for this technique, as arithmetic.

    The speedup ceiling is a ratio of memory traffic, and GQA has already
    removed most of the KV traffic DRAFTKV wants to save -- so the context at
    which drafting pays moves out roughly in proportion to the GQA ratio.
    """
    cfg = CompressionConfig(4, 0.5)
    base = dict(n_layers=32, d_head=128, weight_bytes=16e9, bandwidth=2.0e12)

    def breakeven(n_kv, target=1.5):
        cm = CostModel(n_kv_heads=n_kv, **base)
        lo, hi = 128, 8_000_000
        while lo < hi:
            mid = (lo + hi) // 2
            if cm.ceiling(cfg, mid) >= target:
                hi = mid
            else:
                lo = mid + 1
        return lo

    mha, gqa4, gqa8 = breakeven(32), breakeven(8), breakeven(4)
    assert mha < gqa4 < gqa8
    assert gqa4 / mha == pytest.approx(4.0, rel=0.1)
    assert gqa8 / mha == pytest.approx(8.0, rel=0.1)


def test_ceiling_is_lower_under_gqa_at_fixed_context():
    cfg = CompressionConfig(4, 0.5)
    base = dict(n_layers=32, d_head=128, weight_bytes=16e9, bandwidth=2.0e12)
    for ctx in (32768, 131072):
        mha = CostModel(n_kv_heads=32, **base).ceiling(cfg, ctx)
        gqa = CostModel(n_kv_heads=8, **base).ceiling(cfg, ctx)
        assert gqa < mha


# ---------------------------------------------------------- real weights
@needs_ckpt
def test_real_llama_tokenizer_roundtrip():
    _, tok = load_llama(default_llama_dir())
    for text in ("The capital of France is Paris.", "def f(x):\n    return x + 1\n"):
        assert tok.decode(tok.encode(text)) == text


@needs_ckpt
def test_real_llama_is_lossless():
    m, tok = load_llama(default_llama_dir())
    ids = tok.encode("Memory bandwidth, not arithmetic, is the binding constraint "
                     "on modern inference hardware. ") * 3
    base = DraftKVEngine(m, seed=0).generate_baseline(list(ids), 12, 0.0)
    for cfg in (CompressionConfig(4, 1.0), CompressionConfig(4, 0.5)):
        r = DraftKVEngine(m, FixedPolicy(cfg, 4), seed=0).generate(list(ids), 12, 0.0)
        assert r.tokens == base, cfg.label()
