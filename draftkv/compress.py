"""KV compression primitives and the two cache implementations.

The whole point of the design: `FullKVCache` and `CompressedKVCache` expose the
*same* interface, so the identical model weights can run against either one.
The drafter is not a smaller model -- it is the target model reading a lossy
copy of its own memory.
"""

from __future__ import annotations

import numpy as np

from .config import CompressionConfig


# --------------------------------------------------------------------------
# group-wise asymmetric integer quantization
# --------------------------------------------------------------------------
def quantize(x: np.ndarray, bits: int, group: int = 64) -> np.ndarray:
    """Round-trip `x` through a `bits`-wide affine grid, per group of channels.

    Returns the dequantized array (we simulate storage rather than bit-pack;
    accuracy is exact, only the byte accounting is analytic -- see
    `CompressionConfig.bytes_per_entry`).
    """
    if bits >= 16:
        return x.astype(np.float32, copy=True)
    orig_shape = x.shape
    d = orig_shape[-1]
    pad = (-d) % group
    flat = x.reshape(-1, d).astype(np.float32)
    if pad:
        flat = np.pad(flat, ((0, 0), (0, pad)))
    g = flat.reshape(flat.shape[0], -1, group)

    lo = g.min(axis=-1, keepdims=True)
    hi = g.max(axis=-1, keepdims=True)
    levels = (1 << bits) - 1
    scale = np.maximum((hi - lo) / levels, 1e-8)
    q = np.rint((g - lo) / scale).clip(0, levels)
    deq = (q * scale + lo).reshape(flat.shape)
    if pad:
        deq = deq[:, :d]
    return deq.reshape(orig_shape)


class FullKVCache:
    """Exact fp16-equivalent cache.  The source of truth for verification."""

    def __init__(self, n_layers: int) -> None:
        self.n_layers = n_layers
        self.k: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
        self.v: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
        self.pos: list[list[int]] = [[] for _ in range(n_layers)]

    # -- model-facing interface -------------------------------------------
    def append(self, layer: int, k: np.ndarray, v: np.ndarray, pos: np.ndarray) -> None:
        for i in range(k.shape[0]):
            self.k[layer].append(k[i])
            self.v[layer].append(v[i])
            self.pos[layer].append(int(pos[i]))

    def read(self, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.k[layer]:
            d = (0,)
            return np.zeros((0, 0, 0), np.float32), np.zeros((0, 0, 0), np.float32), np.zeros(d, np.int64)
        return (
            np.stack(self.k[layer]),
            np.stack(self.v[layer]),
            np.asarray(self.pos[layer], dtype=np.int64),
        )

    def note_attention(self, layer: int, probs: np.ndarray) -> None:  # pragma: no cover - no-op
        pass

    # -- control ----------------------------------------------------------
    def truncate(self, n_pos: int) -> None:
        """Drop every entry whose position index is >= n_pos."""
        for L in range(self.n_layers):
            keep = [i for i, p in enumerate(self.pos[L]) if p < n_pos]
            self.k[L] = [self.k[L][i] for i in keep]
            self.v[L] = [self.v[L][i] for i in keep]
            self.pos[L] = [self.pos[L][i] for i in keep]

    @property
    def length(self) -> int:
        return len(self.pos[0])

    def nbytes(self, d_head: int) -> float:
        return 2.0 * self.length * len(self.k[0][0]) * d_head * self.n_layers if self.length else 0.0


class CompressedKVCache:
    """The drafter's lossy view: quantized entries plus token eviction.

    Eviction policy is sink + recent window + heavy hitters (H2O-style), where
    "heavy" is measured by attention mass accumulated through `note_attention`.
    Evicting is what actually buys speed: fewer rows to stream per draft step.
    """

    def __init__(self, n_layers: int, cfg: CompressionConfig) -> None:
        self.n_layers = n_layers
        self.cfg = cfg
        self.k: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
        self.v: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
        self.pos: list[list[int]] = [[] for _ in range(n_layers)]
        self.score: list[list[float]] = [[] for _ in range(n_layers)]
        self.seen_positions = 0  # logical context length, including evicted tokens

    # -- model-facing interface -------------------------------------------
    def append(self, layer: int, k: np.ndarray, v: np.ndarray, pos: np.ndarray) -> None:
        kq = quantize(k, self.cfg.bits)
        vq = quantize(v, self.cfg.bits)
        for i in range(kq.shape[0]):
            self.k[layer].append(kq[i])
            self.v[layer].append(vq[i])
            self.pos[layer].append(int(pos[i]))
            self.score[layer].append(0.0)
        if layer == 0:
            self.seen_positions = max(self.seen_positions, int(pos[-1]) + 1)
        self._evict(layer)

    def read(self, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.k[layer]:
            return np.zeros((0, 0, 0), np.float32), np.zeros((0, 0, 0), np.float32), np.zeros((0,), np.int64)
        return (
            np.stack(self.k[layer]),
            np.stack(self.v[layer]),
            np.asarray(self.pos[layer], dtype=np.int64),
        )

    def note_attention(self, layer: int, probs: np.ndarray) -> None:
        """probs: (Tq, H, Tk) attention weights over the rows currently cached."""
        if probs.size == 0:
            return
        mass = probs.sum(axis=(0, 1))
        n = min(len(self.score[layer]), mass.shape[0])
        for i in range(n):
            self.score[layer][i] += float(mass[i])

    # -- control ----------------------------------------------------------
    def budget(self) -> int:
        keep = int(round(self.cfg.keep_frac * max(self.seen_positions, 1)))
        return max(keep, self.cfg.sink + self.cfg.recent)

    def _evict(self, layer: int) -> None:
        n = len(self.pos[layer])
        budget = self.budget()
        if n <= budget:
            return
        pos = np.asarray(self.pos[layer])
        order = np.argsort(pos)
        protected = set(order[: self.cfg.sink].tolist()) | set(order[-self.cfg.recent :].tolist())
        free = [i for i in range(n) if i not in protected]
        n_drop = n - budget
        if n_drop <= 0 or not free:
            return
        # drop the lowest accumulated attention mass first
        free.sort(key=lambda i: self.score[layer][i])
        drop = set(free[:n_drop])
        keep = [i for i in range(n) if i not in drop]
        self.k[layer] = [self.k[layer][i] for i in keep]
        self.v[layer] = [self.v[layer][i] for i in keep]
        self.pos[layer] = [self.pos[layer][i] for i in keep]
        self.score[layer] = [self.score[layer][i] for i in keep]

    def truncate(self, n_pos: int) -> None:
        for L in range(self.n_layers):
            keep = [i for i, p in enumerate(self.pos[L]) if p < n_pos]
            self.k[L] = [self.k[L][i] for i in keep]
            self.v[L] = [self.v[L][i] for i in keep]
            self.pos[L] = [self.pos[L][i] for i in keep]
            self.score[L] = [self.score[L][i] for i in keep]
        self.seen_positions = min(self.seen_positions, n_pos)

    def sync_from(self, full: FullKVCache, start_pos: int) -> int:
        """Overwrite positions >= start_pos with quantized *true* K/V.

        Necessary, not cosmetic: K/V written during drafting were computed from
        a compressed context, so they are not the true K/V for those positions.
        Without this resync the drafter's error compounds across rounds.
        Returns the number of (layer, token) rows rewritten -- the resync cost.
        """
        self.truncate(start_pos)
        rows = 0
        for L in range(self.n_layers):
            k, v, pos = full.read(L)
            sel = np.where(pos >= start_pos)[0]
            if sel.size == 0:
                continue
            self.append(L, k[sel], v[sel], pos[sel])
            rows += int(sel.size)
        return rows

    @property
    def length(self) -> int:
        return len(self.pos[0])

    def nbytes(self, n_heads: int, d_head: int) -> float:
        per = self.cfg.bytes_per_entry(d_head)
        return 2.0 * self.length * n_heads * per * self.n_layers
