import numpy as np
import pytest

from draftkv import CompressionConfig, CostModel, best_gamma, expected_tokens


def test_expected_tokens_endpoints():
    for g in (1, 4, 9):
        assert expected_tokens(0.0, g) == pytest.approx(1.0)      # only the bonus token
        assert expected_tokens(1.0, g) == pytest.approx(g + 1)    # everything accepted
    assert expected_tokens(0.5, 0) == pytest.approx(1.0)


def test_expected_tokens_matches_monte_carlo():
    rng = np.random.default_rng(0)   # fixed seed: tolerance below is MC noise, not slack
    for alpha in (0.2, 0.55, 0.9):
        for gamma in (2, 5, 8):
            draws = []
            for _ in range(20000):
                k = 0
                while k < gamma and rng.random() < alpha:
                    k += 1
                draws.append(k + 1)      # accepted run plus the bonus token
            assert np.mean(draws) == pytest.approx(expected_tokens(alpha, gamma), abs=0.06)


def test_expected_tokens_saturates():
    """E is bounded by 1/(1-alpha) no matter how long the draft -- the reason
    gamma has a finite optimum even when drafting is nearly free."""
    for alpha in (0.5, 0.8, 0.95):
        assert expected_tokens(alpha, 1000) == pytest.approx(1 / (1 - alpha), rel=1e-6)
        assert expected_tokens(alpha, 5) < expected_tokens(alpha, 6) <= 1 / (1 - alpha)


def test_best_gamma_monotone_in_alpha():
    cm = CostModel(full_kv_bandwidth=6.4e10, hbm_kv_budget=1e9)
    cfg = CompressionConfig(4, 0.5)
    gs = [best_gamma(cm, cfg, a, 32768, 16)[0] for a in (0.1, 0.3, 0.5, 0.7, 0.9, 0.99)]
    assert gs == sorted(gs)


def test_speedup_never_exceeds_ceiling():
    cm = CostModel(full_kv_bandwidth=6.4e10, hbm_kv_budget=1e9)
    for ctx in (512, 8192, 131072):
        for cfg in (CompressionConfig(16, 1.0), CompressionConfig(4, 0.5), CompressionConfig(2, 0.25)):
            ceil = cm.ceiling(cfg, ctx)
            for g in range(1, 32):
                assert cm.speedup(cfg, g, 0.999, ctx) <= ceil + 1e-9


def test_short_context_has_no_headroom():
    """The honest negative result: at batch 1 with an HBM-resident cache and a
    short context, there is essentially nothing to win."""
    cm = CostModel()
    assert cm.ceiling(CompressionConfig(2, 0.25), 512) < 1.02


def test_quantization_error_decreases_with_bits():
    from draftkv.compress import quantize

    x = np.random.default_rng(0).standard_normal((16, 8, 64)).astype(np.float32)
    errs = [np.abs(quantize(x, b) - x).mean() for b in (2, 3, 4, 8)]
    assert errs == sorted(errs, reverse=True)
    assert np.array_equal(quantize(x, 16), x)


def test_bytes_per_entry_shrinks_with_bits():
    d = 128
    sizes = [CompressionConfig(b, 1.0).bytes_per_entry(d) for b in (2, 3, 4, 8, 16)]
    assert sizes == sorted(sizes)
    assert sizes[-1] == 2 * d
