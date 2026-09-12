"""A small but genuine causal transformer, in numpy.

It exists so that DRAFTKV's central claim -- identical output regardless of
compression -- is an *executable test* rather than an assertion.  The forward
pass never learns which cache it is reading; it just calls `append` / `read`.
That is exactly the property a real vLLM integration must preserve.
"""

from __future__ import annotations

import numpy as np

from .compress import CompressedKVCache, FullKVCache

Cache = FullKVCache | CompressedKVCache


def _rms_norm(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class TinyTransformer:
    """Decoder-only transformer with randomly initialized weights.

    Random weights mean the text is gibberish, but every quantity DRAFTKV cares
    about is real: attention mass concentrates unevenly (so eviction has a
    signal to act on), and quantization noise genuinely flips argmaxes (so the
    acceptance rate genuinely depends on the compression config).
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int | None = None,
        max_pos: int = 4096,
        attn_sharpness: float = 1.0,
        attn_gain: float = 1.0,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_head = d_model // n_heads
        d_ff = d_ff or 2 * d_model
        # Random weights give near-uniform attention, which would make the
        # cache irrelevant and every draft trivially accepted.  These two
        # knobs restore the two properties real LMs have and DRAFTKV needs:
        # peaked attention (so eviction is a real decision) and an attention
        # path that actually moves the logits (so quantization noise can flip
        # an argmax).
        self.attn_sharpness = attn_sharpness
        self.attn_gain = attn_gain

        s = 1.0 / np.sqrt(d_model)
        f32 = np.float32
        self.emb = (rng.standard_normal((vocab_size, d_model)) * s).astype(f32)
        self.pos_emb = self._sinusoidal(max_pos, d_model).astype(f32)
        self.layers = []
        for _ in range(n_layers):
            self.layers.append(
                dict(
                    n1=np.ones(d_model, f32),
                    wq=(rng.standard_normal((d_model, d_model)) * s * attn_sharpness).astype(f32),
                    wk=(rng.standard_normal((d_model, d_model)) * s).astype(f32),
                    wv=(rng.standard_normal((d_model, d_model)) * s).astype(f32),
                    wo=(rng.standard_normal((d_model, d_model)) * s * attn_gain).astype(f32),
                    n2=np.ones(d_model, f32),
                    w1=(rng.standard_normal((d_model, d_ff)) * s).astype(f32),
                    w2=(rng.standard_normal((d_ff, d_model)) * s).astype(f32),
                )
            )
        self.nf = np.ones(d_model, f32)
        self.head = (rng.standard_normal((d_model, vocab_size)) * s).astype(f32)

    @staticmethod
    def _sinusoidal(n: int, d: int) -> np.ndarray:
        p = np.arange(n)[:, None]
        i = np.arange(0, d, 2)[None, :]
        ang = p / np.power(10000.0, i / d)
        out = np.zeros((n, d))
        out[:, 0::2] = np.sin(ang)
        out[:, 1::2] = np.cos(ang)
        return out

    def new_cache(self) -> FullKVCache:
        return FullKVCache(self.n_layers)

    # ---------------------------------------------------------------- forward
    def forward(self, tokens: np.ndarray, cache: Cache, start_pos: int) -> np.ndarray:
        """Run `tokens` (shape (T,)) at positions [start_pos, start_pos+T).

        Writes K/V into `cache` and returns logits of shape (T, vocab).
        """
        tokens = np.asarray(tokens, dtype=np.int64).reshape(-1)
        T = tokens.shape[0]
        pos = np.arange(start_pos, start_pos + T)
        h = self.emb[tokens] + self.pos_emb[pos]
        H, D = self.n_heads, self.d_head

        for L, p in enumerate(self.layers):
            x = _rms_norm(h, p["n1"])
            q = (x @ p["wq"]).reshape(T, H, D)
            k = (x @ p["wk"]).reshape(T, H, D)
            v = (x @ p["wv"]).reshape(T, H, D)

            cache.append(L, k, v, pos)
            ck, cv, cpos = cache.read(L)

            # (Tq, H, Tk)
            scores = np.einsum("qhd,khd->qhk", q, ck) / np.sqrt(D)
            mask = cpos[None, :] > pos[:, None]
            scores = np.where(mask[:, None, :], -np.inf, scores)
            probs = _softmax(scores, axis=-1).astype(np.float32)
            cache.note_attention(L, probs)

            attn = np.einsum("qhk,khd->qhd", probs, cv).reshape(T, H * D)
            h = h + attn @ p["wo"]

            x = _rms_norm(h, p["n2"])
            h = h + np.maximum(x @ p["w1"], 0.0) @ p["w2"]

        return (_rms_norm(h, self.nf) @ self.head).astype(np.float32)

    def logits_to_probs(self, logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        if temperature <= 0:
            out = np.zeros_like(logits)
            out[..., int(np.argmax(logits, -1))] = 1.0
            return out
        return _softmax(logits / temperature, axis=-1)


def demo_model(seed: int = 1, **kw) -> "TinyTransformer":
    """Reference model used by the tests and benchmarks.

    `attn_sharpness`/`attn_gain` are calibrated so the acceptance rate spans a
    useful range across compression configs (fp16 ~1.0 down to ~0.05 at
    2-bit/keep-0.25).  Without them a random-weight model accepts everything
    and the control problem disappears.
    """
    params = dict(
        vocab_size=128, d_model=64, n_heads=4, n_layers=3,
        attn_sharpness=4.0, attn_gain=4.0, seed=seed,
    )
    params.update(kw)
    return TinyTransformer(**params)
