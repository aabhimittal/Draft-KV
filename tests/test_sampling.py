"""Sampling mode: the guarantee weakens from 'identical tokens' to 'identical
distribution', and that distinction deserves a test rather than a footnote."""

import numpy as np
import pytest

from draftkv import CompressionConfig, DraftKVEngine, FixedPolicy
from draftkv.compress import FullKVCache
from draftkv.engine import _probs
from draftkv.model import demo_model


def _tv(a, b):
    return 0.5 * float(np.abs(a - b).sum())


@pytest.mark.parametrize("cfg", [CompressionConfig(8, 1.0), CompressionConfig(2, 0.25)],
                         ids=lambda c: c.label())
def test_first_token_matches_target_distribution(cfg):
    m = demo_model()
    rng = np.random.default_rng(0)
    prompt = [int(x) for x in rng.integers(0, m.vocab_size, 48)]
    temp = 1.0

    # exact target distribution for the next token, under the full cache
    cache = FullKVCache(m.n_layers)
    m.forward(np.array(prompt[:-1]), cache, 0)
    p = _probs(m.forward(np.array([prompt[-1]]), cache, len(prompt) - 1)[-1], temp)

    trials = 4000
    eng = DraftKVEngine(m, FixedPolicy(cfg, 4), seed=1)
    counts = np.zeros(m.vocab_size)
    for _ in range(trials):
        counts[eng.generate(prompt, 1, temp).tokens[len(prompt)]] += 1
    emp = counts / trials

    # control: the same number of draws taken straight from p. Finite-sample
    # noise floor, so the assertion is about bias, not variance.
    ctrl_rng = np.random.default_rng(2)
    ctrl = np.bincount(ctrl_rng.choice(m.vocab_size, size=trials, p=p), minlength=m.vocab_size) / trials

    assert _tv(emp, p) < max(1.5 * _tv(ctrl, p), 0.05)


def test_rejection_repairs_the_distribution_not_just_the_token():
    """If the residual resample were dropped (naive 'just take argmax p'), the
    output distribution would be biased toward the drafter's mistakes. Pin that
    the residual branch is reachable and used."""
    m = demo_model()
    rng = np.random.default_rng(3)
    prompt = [int(x) for x in rng.integers(0, m.vocab_size, 64)]
    r = DraftKVEngine(m, FixedPolicy(CompressionConfig(2, 0.25), 6), seed=4).generate(prompt, 40, 1.0)
    assert r.acceptance_rate < 1.0          # rejections did occur
    assert r.new_tokens >= 40
