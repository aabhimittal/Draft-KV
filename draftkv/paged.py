"""Block-paged KV cache with the rollback semantics DRAFTKV needs.

The hard part of a vLLM integration is not the math, it is that the full cache
lives in a paged pool addressed through block tables, and DRAFTKV *rewinds* it
on every verify pass.  Rewinding a contiguous array is a slice; rewinding a
paged pool means freeing whole blocks, keeping a partially-filled tail block,
and never leaking or double-freeing under the churn of one rollback per round.

This module implements that allocator in numpy and tests it against the
contiguous cache, so the semantics are pinned before any GPU code exists.  It
is not a vLLM patch and does not pretend to be: there is no CUDA, no real
paged attention kernel, no graph capture.  What it does establish is that the
engine's rollback contract survives block-table indirection, which is the part
that would otherwise be discovered late and expensively.

Design mirrors vLLM's: a fixed-size block pool, one block table per layer, and
a `slot` addressing scheme (block_id * block_size + offset).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import CompressionConfig, LayerPlan


class BlockPoolExhausted(RuntimeError):
    """Raised instead of silently corrupting state when the pool runs dry."""


@dataclass
class BlockPool:
    """Fixed pool of KV blocks, handed out and returned by whole blocks."""

    n_blocks: int
    block_size: int
    _free: list[int] = field(default_factory=list)
    _allocated: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self._free = list(range(self.n_blocks))
        self._allocated = set()

    def allocate(self) -> int:
        if not self._free:
            raise BlockPoolExhausted(
                f"all {self.n_blocks} blocks of size {self.block_size} are in use"
            )
        b = self._free.pop()
        self._allocated.add(b)
        return b

    def free(self, block: int) -> None:
        if block not in self._allocated:
            raise RuntimeError(f"double free of block {block}")
        self._allocated.discard(block)
        self._free.append(block)

    @property
    def n_free(self) -> int:
        return len(self._free)

    @property
    def n_used(self) -> int:
        return len(self._allocated)

    def check_invariants(self) -> None:
        """Every block is either free or allocated, exactly once."""
        assert len(set(self._free)) == len(self._free), "duplicate in free list"
        assert not (set(self._free) & self._allocated), "block both free and allocated"
        assert len(self._free) + len(self._allocated) == self.n_blocks, "blocks leaked"


class PagedKVCache:
    """The full cache, stored in blocks, behind the usual cache interface.

    Interchangeable with `FullKVCache` from the model's point of view: it
    implements append / read / note_attention / truncate and stores exact
    values.  `CompressedKVCache` remains the drafter's mirror; this is about
    where the *authoritative* cache lives.
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        d_head: int,
        block_size: int = 16,
        n_blocks: int = 512,
        dtype=np.float32,
    ) -> None:
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_head
        self.block_size = block_size
        self.pool = BlockPool(n_blocks, block_size)
        # one physical store per layer, indexed by slot
        shape = (n_blocks * block_size, n_heads, d_head)
        self.k_store = [np.zeros(shape, dtype) for _ in range(n_layers)]
        self.v_store = [np.zeros(shape, dtype) for _ in range(n_layers)]
        self.block_table: list[list[int]] = [[] for _ in range(n_layers)]
        self.n_tokens: list[int] = [0] * n_layers
        self.pos: list[list[int]] = [[] for _ in range(n_layers)]

    # ------------------------------------------------------------ addressing
    def _slot(self, layer: int, index: int) -> int:
        block = self.block_table[layer][index // self.block_size]
        return block * self.block_size + index % self.block_size

    def _ensure_capacity(self, layer: int, extra: int) -> None:
        need = self.n_tokens[layer] + extra
        have = len(self.block_table[layer]) * self.block_size
        while have < need:
            self.block_table[layer].append(self.pool.allocate())
            have += self.block_size

    # ------------------------------------------------- model-facing interface
    def append(self, layer: int, k: np.ndarray, v: np.ndarray, pos: np.ndarray) -> None:
        n = k.shape[0]
        self._ensure_capacity(layer, n)
        for i in range(n):
            s = self._slot(layer, self.n_tokens[layer])
            self.k_store[layer][s] = k[i]
            self.v_store[layer][s] = v[i]
            self.pos[layer].append(int(pos[i]))
            self.n_tokens[layer] += 1

    def read(self, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.n_tokens[layer]
        if n == 0:
            return (np.zeros((0, self.n_heads, self.d_head), np.float32),
                    np.zeros((0, self.n_heads, self.d_head), np.float32),
                    np.zeros((0,), np.int64))
        slots = np.fromiter((self._slot(layer, i) for i in range(n)), np.int64, n)
        return (self.k_store[layer][slots], self.v_store[layer][slots],
                np.asarray(self.pos[layer], dtype=np.int64))

    def note_attention(self, layer: int, probs: np.ndarray) -> None:  # pragma: no cover
        pass

    # ----------------------------------------------------------- control
    def truncate(self, n_pos: int) -> None:
        """Drop entries at position >= n_pos, freeing whole blocks that empty.

        The partially-filled tail block is *kept*, not freed: the next append
        writes into it. Freeing it and reallocating would be correct but would
        churn a block per round, which is exactly the cost paging exists to
        avoid.
        """
        for L in range(self.n_layers):
            keep = sum(1 for p in self.pos[L] if p < n_pos)
            if keep == self.n_tokens[L]:
                continue
            # positions are appended in order, so a prefix survives
            assert all(p < n_pos for p in self.pos[L][:keep]), "positions not monotone"
            self.pos[L] = self.pos[L][:keep]
            self.n_tokens[L] = keep
            needed_blocks = -(-keep // self.block_size)      # ceil
            while len(self.block_table[L]) > needed_blocks:
                self.pool.free(self.block_table[L].pop())

    @property
    def length(self) -> int:
        return self.n_tokens[0]

    @property
    def blocks_in_use(self) -> int:
        return self.pool.n_used

    def fragmentation(self) -> float:
        """Fraction of allocated slots that hold no live token."""
        cap = self.pool.n_used * self.block_size
        live = sum(self.n_tokens)
        return 0.0 if cap == 0 else 1.0 - live / cap


def plan_blocks_needed(ctx: int, n_layers: int, block_size: int) -> int:
    """Blocks a context of `ctx` tokens occupies across all layers."""
    return n_layers * (-(-ctx // block_size))
