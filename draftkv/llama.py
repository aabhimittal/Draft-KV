"""Modern decoder architectures in numpy: RoPE, GQA, RMSNorm, SwiGLU.

Everything measured so far used the GPT-2 family -- learned positional
embeddings, full multi-head attention, GELU -- which is not what current models
do.  This module runs Llama-style and Qwen2-style checkpoints (SmolLM2, Qwen2.5,
TinyLlama) behind the same `forward(tokens, cache, start_pos)` contract, so the
engine, controller, allocator and paged cache are unchanged.

The architectural difference that actually matters here is **grouped-query
attention**.  The cache stores `n_kv_heads`, not `n_heads`, and attention
repeats each KV head across its query group.  So a model with 8:1 GQA has a KV
cache eight times smaller than an MHA model of the same width -- which moves
DRAFTKV's whole operating point, because the speedup ceiling is a ratio of
memory traffic and GQA has already removed most of the KV traffic.  See
`CostModel.ceiling`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .gpt2 import load_safetensors


# --------------------------------------------------------------------------
class ByteLevelBPE:
    """Byte-level BPE read from a HuggingFace `tokenizer.json`.

    Enough to encode a prompt and decode a completion; `decode(encode(x)) == x`
    is asserted in the tests, which is the property that matters for feeding
    the model real text.
    """

    def __init__(self, path: str | Path) -> None:
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
        model = spec["model"]
        self.encoder: dict[str, int] = model["vocab"]
        self.decoder = {v: k for k, v in self.encoder.items()}
        merges = model.get("merges", [])
        pairs = [tuple(m) if isinstance(m, list) else tuple(m.split()) for m in merges]
        self.ranks = {p: i for i, p in enumerate(pairs) if len(p) == 2}
        from .gpt2 import _byte_encoder

        self.b2u = _byte_encoder()
        self.u2b = {v: k for k, v in self.b2u.items()}

    def _bpe(self, token: str) -> list[str]:
        word = list(token)
        while len(word) > 1:
            cand = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            best = min(cand, key=lambda p: self.ranks.get(p, 1 << 30))
            if best not in self.ranks:
                break
            a, b = best
            out, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            word = out
        return word

    def encode(self, text: str) -> list[int]:
        import re

        pat = re.compile(
            r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?\d+| ?[^\sA-Za-z\d]+|\s+(?!\S)|\s+"
        )
        ids: list[int] = []
        for piece in pat.findall(text):
            u = "".join(self.b2u[b] for b in piece.encode("utf-8"))
            for tok in self._bpe(u):
                if tok in self.encoder:
                    ids.append(self.encoder[tok])
                else:  # fall back to single bytes
                    ids.extend(self.encoder[c] for c in tok)
        return ids

    def decode(self, ids: list[int]) -> str:
        text = "".join(self.decoder.get(int(i), "") for i in ids)
        return bytearray(self.u2b[c] for c in text if c in self.u2b).decode(
            "utf-8", errors="replace"
        )


# --------------------------------------------------------------------------
def _rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _rotate_half(x: np.ndarray) -> np.ndarray:
    d = x.shape[-1] // 2
    return np.concatenate([-x[..., d:], x[..., :d]], axis=-1)


class LlamaLike:
    """Llama / Qwen2 style decoder in numpy, cache-agnostic by construction."""

    def __init__(self, weights: dict[str, np.ndarray], config: dict) -> None:
        c = config
        self.cfg = c
        self.n_layers = c["num_hidden_layers"]
        self.d_model = c["hidden_size"]
        self.n_heads = c["num_attention_heads"]
        self.n_kv_heads = c.get("num_key_value_heads", self.n_heads)
        self.d_head = c.get("head_dim", self.d_model // self.n_heads)
        self.n_rep = self.n_heads // self.n_kv_heads
        self.vocab_size = c["vocab_size"]
        self.eps = c.get("rms_norm_eps", 1e-5)
        self.theta = float(c.get("rope_theta") or 10000.0)

        w = weights
        self.emb = w["model.embed_tokens.weight"]
        self.head = w.get("lm_head.weight", self.emb)     # tied when absent
        self.final_norm = w["model.norm.weight"]
        self.layers = []
        for i in range(self.n_layers):
            p = f"model.layers.{i}."
            self.layers.append(
                dict(
                    n1=w[p + "input_layernorm.weight"],
                    q=w[p + "self_attn.q_proj.weight"], qb=w.get(p + "self_attn.q_proj.bias"),
                    k=w[p + "self_attn.k_proj.weight"], kb=w.get(p + "self_attn.k_proj.bias"),
                    v=w[p + "self_attn.v_proj.weight"], vb=w.get(p + "self_attn.v_proj.bias"),
                    o=w[p + "self_attn.o_proj.weight"], ob=w.get(p + "self_attn.o_proj.bias"),
                    n2=w[p + "post_attention_layernorm.weight"],
                    gate=w[p + "mlp.gate_proj.weight"],
                    up=w[p + "mlp.up_proj.weight"],
                    down=w[p + "mlp.down_proj.weight"],
                )
            )
        self._cos: np.ndarray | None = None
        self._sin: np.ndarray | None = None

    # ------------------------------------------------------------- loading
    @classmethod
    def from_dir(cls, d: str | Path) -> "LlamaLike":
        d = Path(d)
        cfg = json.loads((d / "config.json").read_text())
        return cls(load_safetensors(d / "model.safetensors"), cfg)

    def new_cache(self):
        from .compress import FullKVCache

        return FullKVCache(self.n_layers)

    # ---------------------------------------------------------------- rope
    def _rope(self, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        need = int(pos.max()) + 1
        if self._cos is None or self._cos.shape[0] < need:
            n = max(need, 1024)
            inv = 1.0 / (self.theta ** (np.arange(0, self.d_head, 2) / self.d_head))
            ang = np.outer(np.arange(n), inv)
            emb = np.concatenate([ang, ang], axis=-1)
            self._cos, self._sin = np.cos(emb), np.sin(emb)
        return self._cos[pos], self._sin[pos]

    # ------------------------------------------------------------- forward
    def forward(self, tokens: np.ndarray, cache, start_pos: int) -> np.ndarray:
        tokens = np.asarray(tokens, dtype=np.int64).reshape(-1)
        T = tokens.shape[0]
        pos = np.arange(start_pos, start_pos + T)
        h = self.emb[tokens].astype(np.float32)
        H, KV, D = self.n_heads, self.n_kv_heads, self.d_head
        cos, sin = self._rope(pos)
        cos = cos[:, None, :]
        sin = sin[:, None, :]

        for L, p in enumerate(self.layers):
            x = _rms_norm(h, p["n1"], self.eps)
            q = x @ p["q"].T
            k = x @ p["k"].T
            v = x @ p["v"].T
            if p["qb"] is not None:
                q = q + p["qb"]
                k = k + p["kb"]
                v = v + p["vb"]
            q = q.reshape(T, H, D)
            k = k.reshape(T, KV, D)
            v = v.reshape(T, KV, D)
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin

            # the cache holds KV heads, not query heads: this is where GQA
            # shrinks the thing DRAFTKV is trying to compress
            cache.append(L, k, v, pos)
            ck, cv, cpos = cache.read(L)

            if self.n_rep > 1:
                ck = np.repeat(ck, self.n_rep, axis=1)
                cv = np.repeat(cv, self.n_rep, axis=1)

            scores = np.einsum("qhd,khd->qhk", q, ck) / np.sqrt(D)
            mask = cpos[None, :] > pos[:, None]
            scores = np.where(mask[:, None, :], -np.inf, scores)
            probs = _softmax(scores, axis=-1).astype(np.float32)
            if self.n_rep > 1:   # report attention per KV head, which is what is cached
                cache.note_attention(L, probs.reshape(T, KV, self.n_rep, -1).sum(axis=2))
            else:
                cache.note_attention(L, probs)

            attn = np.einsum("qhk,khd->qhd", probs, cv).reshape(T, H * D)
            h = h + attn @ p["o"].T + (p["ob"] if p["ob"] is not None else 0.0)

            x = _rms_norm(h, p["n2"], self.eps)
            h = h + (_silu(x @ p["gate"].T) * (x @ p["up"].T)) @ p["down"].T

        h = _rms_norm(h, self.final_norm, self.eps)
        return (h @ self.head.T).astype(np.float32)


def load_llama(d: str | Path):
    """Returns (model, tokenizer) for a Llama/Qwen2 style checkpoint dir."""
    d = Path(d)
    if not (d / "model.safetensors").exists():
        raise FileNotFoundError(f"no checkpoint at {d}; see scripts/fetch_model.py")
    tok = ByteLevelBPE(d / "tokenizer.json") if (d / "tokenizer.json").exists() else None
    return LlamaLike.from_dir(d), tok


def default_llama_dir(name: str = "smollm2-135m") -> Path | None:
    import os

    env = os.environ.get("DRAFTKV_LLAMA_DIR")
    for c in ([Path(env)] if env else []) + [Path.home() / ".cache" / "draftkv" / name]:
        if (c / "model.safetensors").exists():
            return c
    return None


def demo_llama(
    n_layers: int = 4,
    d_model: int = 64,
    n_heads: int = 8,
    n_kv_heads: int = 2,
    vocab_size: int = 128,
    intermediate: int = 128,
    attn_bias: bool = False,
    seed: int = 0,
    attn_gain: float = 4.0,
    attn_sharpness: float = 4.0,
) -> LlamaLike:
    """Random-weight Llama-style model: RoPE + GQA + RMSNorm + SwiGLU.

    Exists so the modern-architecture path is covered in CI without a 300 MB
    download. Same caveat as `demo_model`: random weights exaggerate and
    distort quantization damage, so this is for structural properties
    (losslessness, GQA cache shapes, rollback) and not for numbers.
    """
    rng = np.random.default_rng(seed)
    d_head = d_model // n_heads
    s = 1.0 / np.sqrt(d_model)
    f32 = np.float32
    R = lambda *shape, k=1.0: (rng.standard_normal(shape) * s * k).astype(f32)

    w: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": R(vocab_size, d_model),
        "model.norm.weight": np.ones(d_model, f32),
    }
    for i in range(n_layers):
        p = f"model.layers.{i}."
        w[p + "input_layernorm.weight"] = np.ones(d_model, f32)
        w[p + "post_attention_layernorm.weight"] = np.ones(d_model, f32)
        w[p + "self_attn.q_proj.weight"] = R(n_heads * d_head, d_model, k=attn_sharpness)
        w[p + "self_attn.k_proj.weight"] = R(n_kv_heads * d_head, d_model)
        w[p + "self_attn.v_proj.weight"] = R(n_kv_heads * d_head, d_model)
        w[p + "self_attn.o_proj.weight"] = R(d_model, n_heads * d_head, k=attn_gain)
        if attn_bias:
            w[p + "self_attn.q_proj.bias"] = R(n_heads * d_head)[0] * 0 + 0.01
            w[p + "self_attn.k_proj.bias"] = np.zeros(n_kv_heads * d_head, f32) + 0.01
            w[p + "self_attn.v_proj.bias"] = np.zeros(n_kv_heads * d_head, f32) + 0.01
            w[p + "self_attn.q_proj.bias"] = np.zeros(n_heads * d_head, f32) + 0.01
        w[p + "mlp.gate_proj.weight"] = R(intermediate, d_model)
        w[p + "mlp.up_proj.weight"] = R(intermediate, d_model)
        w[p + "mlp.down_proj.weight"] = R(d_model, intermediate)

    cfg = dict(
        num_hidden_layers=n_layers, hidden_size=d_model, num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads, vocab_size=vocab_size, rms_norm_eps=1e-5,
        rope_theta=10000.0,
    )
    return LlamaLike(w, cfg)
