"""The real-model path.

The safetensors reader is tested against a file this test writes itself, so it
runs in CI with no download.  Everything that needs the 350 MB checkpoint skips
cleanly when it is absent -- `python scripts/fetch_model.py` enables it.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from draftkv import CompressionConfig, DraftKVEngine, FixedPolicy
from draftkv.gpt2 import default_model_dir, load_real_model, load_safetensors

needs_model = pytest.mark.skipif(
    default_model_dir() is None,
    reason="no distilgpt2 checkpoint (run scripts/fetch_model.py)",
)


def _write_safetensors(path, arrays: dict[str, np.ndarray]) -> None:
    header, offset, blobs = {}, 0, []
    for name, a in arrays.items():
        a = np.ascontiguousarray(a, dtype=np.float32)
        b = a.tobytes()
        header[name] = {"dtype": "F32", "shape": list(a.shape),
                        "data_offsets": [offset, offset + len(b)]}
        offset += len(b)
        blobs.append(b)
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))


def test_safetensors_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    arrays = {"a": rng.standard_normal((3, 4)), "b": rng.standard_normal((7,)),
              "c.nested.weight": rng.standard_normal((2, 3, 4))}
    f = tmp_path / "m.safetensors"
    _write_safetensors(f, arrays)
    got = load_safetensors(f)
    assert set(got) == set(arrays)
    for k, v in arrays.items():
        assert np.allclose(got[k], v.astype(np.float32))
        assert got[k].shape == v.shape


def test_safetensors_ignores_metadata(tmp_path):
    f = tmp_path / "m.safetensors"
    header = {"__metadata__": {"format": "pt"},
              "w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    raw = json.dumps(header).encode()
    f.write_bytes(struct.pack("<Q", len(raw)) + raw + np.float32([1, 2]).tobytes())
    assert list(load_safetensors(f)) == ["w"]


@needs_model
def test_tokenizer_roundtrip():
    _, tok = load_real_model()
    for text in ("The capital of France is Paris.",
                 "  leading space and 12345 digits!",
                 "punctuation, semicolons; and-hyphens."):
        assert tok.decode(tok.encode(text)) == text


@needs_model
def test_real_model_generates_expected_continuation():
    """Pins that the weights are loaded and wired correctly -- a transposed
    Conv1D or a wrong layernorm still produces text, just not this text."""
    m, tok = load_real_model()
    ids = tok.encode("The capital of France is")
    cache = m.new_cache()
    m.forward(np.array(ids[:-1]), cache, 0)
    out = list(ids)
    for _ in range(4):
        nxt = int(np.argmax(m.forward(np.array([out[-1]]), cache, len(out) - 1)[-1]))
        out.append(nxt)
    assert tok.decode(out[len(ids):]).strip().startswith("the capital")


@needs_model
@pytest.mark.parametrize("cfg", [CompressionConfig(8, 1.0), CompressionConfig(4, 1.0),
                                 CompressionConfig(4, 0.5), CompressionConfig(2, 0.25)],
                         ids=lambda c: c.label())
def test_losslessness_on_a_real_model(cfg):
    """The guarantee that actually matters, on real weights rather than random
    ones. Compression only moves the acceptance rate; the tokens are identical."""
    m, tok = load_real_model()
    ids = tok.encode(
        "Memory bandwidth, not arithmetic, is the binding constraint on modern "
        "inference hardware. Every generation widens the gap, and every "
        "generation of software invents a new way to hide it. "
    ) * 2
    base = DraftKVEngine(m, seed=0).generate_baseline(list(ids), 16, 0.0)
    r = DraftKVEngine(m, FixedPolicy(cfg, 4), seed=0).generate(list(ids), 16, 0.0)
    assert r.tokens == base
