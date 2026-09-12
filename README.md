# DRAFTKV

**KV-cache compression as a pure speed setting.**

Compressing a KV cache normally buys memory and bandwidth by spending output
quality, so the compression setting is a quality knob someone has to tune
offline and defend. DRAFTKV removes quality from the equation: the compressed
cache is used only to *draft* tokens, and every drafted token is verified
against the full-precision cache before it is emitted. Output is identical to
full-KV decoding. Compression can then only affect speed — which makes it a
control problem with a free, unlabeled feedback signal: the acceptance rate the
verify pass reports on every step.

This repository is a from-scratch, dependency-light implementation (numpy only)
with the losslessness claim as an executable test rather than an assertion.

```bash
pip install -e .
python -m draftkv.bench all      # seven experiments
pytest -q                        # 41 tests
```

## The idea in one paragraph

Ordinary speculative decoding pairs a big accurate model with a small fast one.
The small model is a different model, so it disagrees with the target often, and
accepted runs are short. DRAFTKV keeps one model and gives it two memories: the
drafter *is* the target model, reading a lossy copy of its own KV cache. The
analogy: rather than asking an intern to guess what the expert will say, you ask
the expert to answer from memory — quickly, without re-reading the file — then
check the answer against the file. Someone working from a slightly blurry
recollection of their own notes agrees with their own careful self far more
often than a junior colleague does, which is why accepted runs get long.

Verification is what makes the blur harmless. The full cache produces the
logits that actually decide the output; the compressed cache only proposes.

## The mechanism

```
request -> draft gamma tokens (compressed KV, config c)
        -> verify all gamma in one pass (full KV) -> accept k of gamma
        -> update posterior over alpha(c) from (k, gamma) -> pick next c, gamma
```

Let `alpha` be the per-token acceptance rate and `gamma` the draft length. Each
verify pass emits the accepted run plus one bonus token that the pass produces
for free:

```
E(alpha, gamma) = (1 - alpha^(gamma+1)) / (1 - alpha)
T(c, gamma)     = E(alpha(c), gamma) / (gamma * t_draft(c) + t_verify)
```

`E` saturates at `1/(1-alpha)` while the denominator grows linearly in `gamma`,
so `gamma*` is finite and rises steeply with `alpha` — 1 at alpha=0.3, 9 at
alpha=0.9, 16+ at alpha=0.99 (`bench gamma`).

`alpha(c)` cannot be known in advance and drifts with content (code and prose
quantize differently), so it is learned online.

## Where the honest ceiling is

The most important number in this repository is not a speedup, it is a bound.
Drafting and verifying both run the same model, so both stream the full weights.
Over one round, weight traffic is `(gamma+1) * W` — exactly what plain decoding
pays. **Every byte DRAFTKV saves is a KV byte**, so the speedup is bounded by

```
ceiling = (W + K_full) / (W + K_compressed)
```

`CostModel.ceiling()` computes it, and `test_speedup_never_exceeds_ceiling`
pins that nothing in the controller can exceed it. Consequences (`bench
regimes`, Llama-3-8B shape, H100-class bandwidth):

| setup | ctx 1k | ctx 32k | ctx 256k |
|---|---|---|---|
| HBM cache, batch 1 | 1.01x | 1.22x | 2.42x |
| HBM cache, batch 32 | 1.22x | 4.34x | 6.54x |
| offloaded cache (PCIe5) | 1.26x | 9.04x | 7.97x |

At short context and batch 1 there is nothing to win, and a controller that
does not know this will burn cycles drafting for a ~1% return. The win lives
where KV traffic rivals weight traffic: long context, large batch, or an
offloaded full cache. That last case is the one worth naming plainly — the
scheme's real trade is **GPU memory for transfer cost**. The full cache must
live somewhere; if you push it to host memory to free HBM, verification pays
PCIe latency, and drafting from a GPU-resident compressed mirror is what hides
it.

`CostModel.hbm_kv_budget` prices exactly this. Without it the model concludes
that fp16/keep-everything drafting is free and compression is pointless —
correctly, for a machine that does not exist.

## The controller

Thompson sampling over compression configs, with a Beta posterior on `alpha(c)`.
Each verify pass yields a truncated-geometric observation — `k` successes plus
one failure iff the run was cut short — which is conjugate to the Beta, so the
update is two additions.

Three design decisions that the experiments actually justify:

**`gamma` is solved, not explored.** Under the same i.i.d. acceptance model that
`E(alpha, gamma)` already assumes, `gamma` cannot change `alpha` — it only
changes how much of the geometric tail you observe. So `alpha` is a property of
`c` alone, and `gamma*` follows in closed form from the sampled `alpha`. This
shrinks the arm space from `|C| x |gamma|` to `|C|`. Against the literal
`(c, gamma)` product bandit, at 3000 rounds: **95.3% of optimal vs 80.3%**, with
293 config switches vs 1112 (`bench controller`).

**Forgetting is a tuned trade, not a free win.** Beta counts decay toward the
prior so the posterior tracks content shifts. When the shift moves the argmax
(tolerant content prefers 4b/keep0.25; brittle content prefers 8b/keep1), a
stale posterior reaches 42% of optimum and a decaying one 71% (`bench
nonstationary`). In a stationary world, decay costs a little.

**Switching config is priced, but not forbidden.** Changing `c` requantizes the
whole drafter cache — `O(ctx)` — so a challenger arm is charged that cost
amortized over a horizon. The tempting stronger move, holding a config for a
fixed number of rounds, was implemented, measured, and defaulted *off*: it cuts
switches from 293 to 22 but loses more to slower convergence than it saves
(93.7% vs 95.3%). Thompson sampling already stops switching on its own once the
posteriors concentrate. `test_hard_commitment_is_not_free` keeps that negative
result from being quietly re-introduced.

**`gamma = 0` is always an arm.** Gating falls out of the same argmax: if no
config beats the baseline by `min_speedup`, the controller declines to draft.
No separate heuristic, no context-length threshold to tune.

Measured against baselines (stationary, 15 configs, 3000 rounds):

| policy | speedup | % of optimal | config switches |
|---|---|---|---|
| oracle (knows every alpha) | 2.55x | 100% | 0 |
| best fixed config, in hindsight | 2.53x | 99.3% | 0 |
| **Thompson (factored), after warm-up** | **2.50x** | **98.1%** | 39 |
| Thompson (factored), from cold | 2.43x | 95.3% | 293 |
| Thompson, forced 32-round commitment | 2.39x | 93.7% | 22 |
| flat `(c, gamma)` bandit | 2.05x | 80.3% | 1112 |
| average fixed config, picked blind | 1.36x | 53.6% | 0 |

Read that honestly: a fixed config chosen with hindsight is already near-optimal
in a stationary world, and the bandit cannot beat it — it pays an exploration
tax first. The case for the controller is that it lands on that arm without
hindsight, stays far above the config you would pick blind throughout, and keeps
working when the content changes, which no fixed choice does.

## Correctness

- **Greedy decoding: byte-identical output.** Every emitted token is the argmax
  of logits computed from the exact full cache. Tested across
  `{fp16, 8b, 4b, 3b, 2b} x {keep 1.0, 0.5, 0.25} x gamma in {1,3,6}` — 2-bit
  quantization with 75% of the cache evicted still emits the same tokens.
- **Sampling: identical in distribution.** The standard speculative-sampling
  rejection rule (accept with `min(1, p/q)`, else resample from
  `norm(p - q)+`). Checked against a finite-sample control in
  `tests/test_sampling.py`. The guarantee genuinely weakens here from "same
  tokens" to "same distribution", and that distinction is tested rather than
  footnoted.

Two subtleties that are easy to get wrong and are handled explicitly:

1. **Drafted K/V are not true K/V.** They were computed *from a compressed
   context*, so after each verify the drafter's cache is truncated to the
   accepted prefix and resynced from the full cache's freshly computed rows
   (`CompressedKVCache.sync_from`). Skip this and error compounds across rounds
   — silently, since the output stays correct and only throughput rots.
2. **The last committed token stays pending**, uncached, so the verify pass
   recomputes the drafter's very first step. Cache it instead and the first
   drafted token is trivially accepted, biasing every acceptance statistic the
   controller learns from.

## Layout

| file | contents |
|---|---|
| `compress.py` | group-wise int quantization; `FullKVCache` / `CompressedKVCache` behind one interface; sink + recent + heavy-hitter eviction |
| `model.py` | small causal transformer in numpy — cache-agnostic by construction |
| `engine.py` | the draft/verify loop, rollback, and resync |
| `throughput.py` | `E(alpha, gamma)`, the memory-traffic cost model, `ceiling()`, `best_gamma` |
| `controller.py` | Thompson sampling, flat-product baseline, oracle, fixed policies |
| `sim.py` | bandit environment with known ground-truth `alpha(c)` |
| `bench.py` | the seven experiments |

The reference model has random weights, so the text is gibberish — irrelevant,
because every quantity under test is structural. Its `attn_sharpness` /
`attn_gain` knobs exist because a plain random-weight transformer has diffuse
attention, accepts every draft regardless of compression, and would make the
losslessness test vacuous.

## What is not here

- **No vLLM or CUDA integration.** The hard part of shipping this is paged
  attention and CUDA graphs, not the math: a second paged pool for the
  compressed mirror, a rollback that respects block tables, and captured graphs
  for each `gamma` in use (`gamma` varying per round otherwise forces re-capture
  or eager mode — one reason to quantize `gamma` to a handful of values). The
  cost model is written so those costs can be measured and substituted rather
  than assumed.
- **No real-model acceptance rates.** `sim.plausible_alpha` is a stylized shape,
  not a calibration. Real `alpha(c)` is exactly the thing the controller exists
  to measure, and quoting invented numbers for it would defeat the point.
- **No claim that this is VeriCache.** It implements the mechanism the idea
  describes, and specifically does not assume the paper's handling of where the
  full cache lives — that is modeled explicitly here as a bandwidth and budget
  parameter so the assumption is visible instead of inherited.
