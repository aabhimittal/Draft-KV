"""A real open-source model behind the same cache interface as the toy one.

The reference transformer in `model.py` has random weights, which is fine for
proving losslessness but says nothing about what acceptance rates or per-layer
sensitivities actually look like.  This module runs distilgpt2 -- real weights,
real text -- in numpy, exposing exactly the `forward(tokens, cache, start_pos)`
contract the engine already uses.  Nothing in the engine, controller or
allocator changes; only the model behind them does.

Weights are loaded straight from safetensors, so there is no torch dependency.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

_DTYPE = {"F32": np.float32, "F16": np.float16, "BF16": None, "I64": np.int64}


def load_safetensors(path: str | Path) -> dict[str, np.ndarray]:
    """Minimal safetensors reader: u64 header length, JSON header, raw buffer."""
    path = Path(path)
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
        blob = f.read()
    out: dict[str, np.ndarray] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dt = _DTYPE.get(meta["dtype"])
        if dt is None:  # bf16: reinterpret as upper half of f32
            lo, hi = meta["data_offsets"]
            raw = np.frombuffer(blob[lo:hi], dtype=np.uint16).astype(np.uint32) << 16
            out[name] = raw.view(np.float32).reshape(meta["shape"])
            continue
        lo, hi = meta["data_offsets"]
        out[name] = np.frombuffer(blob[lo:hi], dtype=dt).reshape(meta["shape"]).astype(np.float32)
    return out


# --------------------------------------------------------------------------
# GPT-2 byte-level BPE
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _byte_encoder() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(
        range(ord("\xae"), ord("\xff") + 1)
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


class BPETokenizer:
    """Byte-level BPE, enough to encode a prompt and decode a completion."""

    def __init__(self, vocab_path: str | Path, merges_path: str | Path) -> None:
        self.encoder: dict[str, int] = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        self.decoder = {v: k for k, v in self.encoder.items()}
        lines = Path(merges_path).read_text(encoding="utf-8").split("\n")[1:]
        merges = [tuple(l.split()) for l in lines if l.strip()]
        self.ranks = {m: i for i, m in enumerate(merges)}
        self.b2u = _byte_encoder()
        self.u2b = {v: k for k, v in self.b2u.items()}

    def _bpe(self, token: str) -> list[str]:
        word = list(token)
        while len(word) > 1:
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            best = min(pairs, key=lambda p: self.ranks.get(p, 1 << 30))
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
            unicode_piece = "".join(self.b2u[b] for b in piece.encode("utf-8"))
            ids.extend(self.encoder[t] for t in self._bpe(unicode_piece))
        return ids

    def decode(self, ids: list[int]) -> str:
        text = "".join(self.decoder[i] for i in ids)
        return bytearray(self.u2b[c] for c in text).decode("utf-8", errors="replace")


# --------------------------------------------------------------------------
def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def _gelu_new(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))


class GPT2:
    """distilgpt2 / gpt2 in numpy, cache-agnostic by construction.

    Deliberately mirrors `TinyTransformer.forward`: it calls `cache.append`,
    `cache.read` and `cache.note_attention` and never learns whether it is
    reading the full cache or the compressed mirror.  That is the property the
    whole scheme depends on, and it is the same property a real vLLM
    integration would have to preserve.
    """

    def __init__(self, weights: dict[str, np.ndarray], config: dict) -> None:
        self.cfg = config
        self.n_layers = config["n_layer"]
        self.n_heads = config["n_head"]
        self.d_model = config["n_embd"]
        self.d_head = self.d_model // self.n_heads
        self.vocab_size = config["vocab_size"]
        self.eps = config.get("layer_norm_epsilon", 1e-5)
        w = weights
        pre = "transformer." if "transformer.wte.weight" in w else ""
        self.wte = w[f"{pre}wte.weight"]
        self.wpe = w[f"{pre}wpe.weight"]
        self.blocks = []
        for i in range(self.n_layers):
            p = f"{pre}h.{i}."
            self.blocks.append(
                dict(
                    ln1w=w[p + "ln_1.weight"], ln1b=w[p + "ln_1.bias"],
                    qkvw=w[p + "attn.c_attn.weight"], qkvb=w[p + "attn.c_attn.bias"],
                    ow=w[p + "attn.c_proj.weight"], ob=w[p + "attn.c_proj.bias"],
                    ln2w=w[p + "ln_2.weight"], ln2b=w[p + "ln_2.bias"],
                    fcw=w[p + "mlp.c_fc.weight"], fcb=w[p + "mlp.c_fc.bias"],
                    pw=w[p + "mlp.c_proj.weight"], pb=w[p + "mlp.c_proj.bias"],
                )
            )
        self.lnfw = w[f"{pre}ln_f.weight"]
        self.lnfb = w[f"{pre}ln_f.bias"]

    # ------------------------------------------------------------- loading
    @classmethod
    def from_dir(cls, d: str | Path) -> "GPT2":
        d = Path(d)
        cfg = json.loads((d / "config.json").read_text())
        cfg.setdefault("vocab_size", 50257)
        return cls(load_safetensors(d / "model.safetensors"), cfg)

    def new_cache(self):
        from .compress import FullKVCache

        return FullKVCache(self.n_layers)

    # ------------------------------------------------------------- forward
    def forward(self, tokens: np.ndarray, cache, start_pos: int) -> np.ndarray:
        tokens = np.asarray(tokens, dtype=np.int64).reshape(-1)
        T = tokens.shape[0]
        pos = np.arange(start_pos, start_pos + T)
        h = self.wte[tokens] + self.wpe[pos]
        H, D = self.n_heads, self.d_head

        for L, p in enumerate(self.blocks):
            x = _layer_norm(h, p["ln1w"], p["ln1b"], self.eps)
            qkv = x @ p["qkvw"] + p["qkvb"]                       # Conv1D: x @ W
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(T, H, D)
            k = k.reshape(T, H, D)
            v = v.reshape(T, H, D)

            cache.append(L, k, v, pos)
            ck, cv, cpos = cache.read(L)

            scores = np.einsum("qhd,khd->qhk", q, ck) / np.sqrt(D)
            mask = cpos[None, :] > pos[:, None]
            scores = np.where(mask[:, None, :], -np.inf, scores)
            probs = _softmax(scores, axis=-1).astype(np.float32)
            cache.note_attention(L, probs)

            attn = np.einsum("qhk,khd->qhd", probs, cv).reshape(T, H * D)
            h = h + attn @ p["ow"] + p["ob"]

            x = _layer_norm(h, p["ln2w"], p["ln2b"], self.eps)
            h = h + _gelu_new(x @ p["fcw"] + p["fcb"]) @ p["pw"] + p["pb"]

        h = _layer_norm(h, self.lnfw, self.lnfb, self.eps)
        return (h @ self.wte.T).astype(np.float32)               # tied embeddings


def default_model_dir() -> Path | None:
    """Where a downloaded checkpoint is expected; None if absent.

    Override with DRAFTKV_GPT2_DIR.  Tests and benchmarks that need a real
    model skip themselves when this returns None, so the suite stays runnable
    (and CI stays fast) without a 350 MB download.
    """
    env = os.environ.get("DRAFTKV_GPT2_DIR")
    cands = [Path(env)] if env else []
    cands += [Path.home() / ".cache" / "draftkv" / "distilgpt2"]
    for c in cands:
        if (c / "model.safetensors").exists() and (c / "config.json").exists():
            return c
    return None


def load_real_model(d: str | Path | None = None):
    """Returns (model, tokenizer) or raises if the checkpoint is missing."""
    d = Path(d) if d else default_model_dir()
    if d is None:
        raise FileNotFoundError(
            "no distilgpt2 checkpoint; set DRAFTKV_GPT2_DIR or run "
            "scripts/fetch_model.py"
        )
    tok = None
    if (Path(d) / "vocab.json").exists():
        tok = BPETokenizer(Path(d) / "vocab.json", Path(d) / "merges.txt")
    return GPT2.from_dir(d), tok
