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

![tests](https://github.com/aabhimittal/Draft-KV/actions/workflows/tests.yml/badge.svg)

```bash
pip install -e .
python -m draftkv.bench all      # sixteen experiments
pytest -q                        # 121 tests

python scripts/fetch_model.py                                     # distilgpt2
python scripts/fetch_model.py --model HuggingFaceTB/SmolLM2-135M  # RoPE + GQA + SwiGLU
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

## What survives contact with a real model

Everything above and below was developed against a random-weight transformer.
`draftkv/gpt2.py` runs **distilgpt2** -- real open-source weights, real text --
in numpy, behind the same `forward(tokens, cache, start_pos)` contract, so the
engine, controller and allocator are unchanged and only the model differs.
Weights load straight from safetensors; there is no torch dependency.

```
config        identical   alpha   tok/verify
fp16/keep1          yes   1.000         4.80
8b/keep1            yes   1.000         4.80
4b/keep1            yes   0.857         4.00
4b/keep0.5          yes   0.026         1.09
2b/keep0.25         yes   0.024         1.09
```

**Losslessness transfers exactly** -- the one thing that had to.

**Two findings invert what the toy model implied:**

- *Quantization is nearly free.* 8-bit is exact; 4-bit stays at 0.86-1.00. The
  random-weight model put 4-bit at 0.66. A random-weight transformer has
  diffuse attention and no learned structure to preserve, so it exaggerates
  quantization damage badly.
- *Eviction is the damaging axis.* Dropping tokens is what collapses
  acceptance, not narrowing them. On a real model the budget worth allocating
  per layer is **tokens, not bits** -- so `profile_layers` is now run over
  `keep_frac` candidates at fixed 4-bit.

**One finding replicates, more strongly.** The depth-rise is real:

```
P(accept at depth i | reached i), 4b/keep0.5, distilgpt2
depth   0     1     2     3     4     5     6     7
      0.28  1.00  1.00  1.00  1.00  1.00  1.00  1.00
```

## Acceptance belongs to the content, not the config

The single most important number from the real-model work is a noise figure.
Measuring the same plan on different passages:

| | |
|---|---|
| sd of log-acceptance for one profile cell, across prompts | **~1.2** |
| error of the best per-layer damage model | ~1.5 |

The same plan measures 0.95 on one passage and 0.10 on another. Content
dominates the compression config, and by a wide margin. Consequences, in order
of how much they hurt:

1. **A single-prompt profile is not a measurement of the layer, it is a
   measurement of the prompt.** `profile_layers` now takes a *list* of prompts,
   aggregates cells as a geometric mean, and records `spread` --
   `SensitivityProfile.noise` reports the sd so a damage estimate can be
   compared against its own error bar. The first version of this happily
   reported a profile with no idea that half its cells were noise.
2. **The absolute damage prediction does not survive a change of content**, so
   `predicted_alpha` is a ranking device and must not be read as a forecast.
3. **The ranking does survive**, which is what saves the allocator:

| prompt | ref | L0 | L1 | L2 | L3 | L4 | L5 | most sensitive |
|---|---|---|---|---|---|---|---|---|
| tech | 0.88 | **0.23** | 0.77 | 0.87 | 0.88 | 0.83 | 0.77 | L0 |
| prose | 1.00 | **0.48** | 1.00 | 1.00 | 0.74 | 1.00 | 1.00 | L0 |
| code | 1.00 | **0.78** | 1.00 | 1.00 | 0.92 | 1.00 | 1.00 | L0 |
| list | 1.00 | **0.32** | 0.96 | 1.00 | 0.67 | 1.00 | 1.00 | L0 |

Layer 0 is the most sensitive layer on **every** prompt; prose, code and list
agree on the entire ordering (Spearman +1.0). Layers 1, 2, 4 and 5 tolerate
having 85% of their KV tokens evicted at no measurable cost. That is a large
win available to a per-layer allocator and invisible to a uniform one -- and it
is exactly the ordering the knapsack consumes.

4. **It strengthens the online controller and weakens offline profiling.** If
   alpha moves this much with content, the component that measures it
   continuously from the verify pass is doing the real work; a static profile
   is a prior, not an answer.

## Fixing the wrong thing: log-additivity was not the problem

The allocator composes per-layer damages by adding them in log-acceptance
(equivalently, acceptance multiplies across layers). On the random-weight model
that rule mispredicted mixed plans badly, so the obvious next move was a better
composition rule. `SensitivityProfile.p` generalizes additivity to a power mean,

```
predicted damage = (sum_L D_L ** p) ** (1/p)
```

with p = 1 recovering log-additivity, p > 1 interpolating toward "only the
worst layer matters" (errors overlap) and p < 1 toward super-additive (errors
amplify). It costs the optimizer nothing: minimizing `(sum D**p)**(1/p)` is
minimizing `sum D**p`, still separable, so the same exact knapsack runs against
`objective()` instead of `damage()`.

Then measuring it properly on distilgpt2 showed the premise was wrong.

| | RMSE (log alpha) |
|---|---|
| log-additive, p = 1 | **0.098** |
| fitted p = 0.90, in-sample | 0.096 |
| fitted p = 0.90, leave-one-out | **0.114** (worse) |
| measurement noise floor | 0.13 |

Additive prediction error is *already below the noise floor of the measurement
it is fitted to*. The exponent buys 0.002 in-sample and loses 0.016 under
cross-validation -- it is fitting noise. The earlier "log-additivity is a poor
predictor" conclusion was an artifact of profiling on a single prompt, where
the noise was 1.17 rather than 0.13. **The fix was averaging the profile over
several prompts, not a better composition rule.**

What ships is therefore a guard rather than a better model:

- `p` defaults to 1.0 and stays there.
- `fit_composition` computes leave-one-out error and **refuses to adopt a fit
  that does not generalize**, returning `adopted: False` and leaving `p` at 1.0.
  `require_cv=False` shows the unguarded fit, which on this data moves p and
  makes things worse.
- `test_noise_around_an_additive_truth_leaves_p_alone` pins the behaviour, and
  the p-norm machinery is kept because the direction of the error is genuinely
  model-dependent -- if a model does show composition structure above its noise
  floor, the mechanism is there and cross-validated.

The general lesson is worth more than the feature: an in-sample improvement on
a dozen noisy points is not evidence, and the honest first question about a
model that predicts badly is whether the thing it is predicting was measured
well enough to predict.

## Does any of it survive a modern architecture?

Every measurement up to here used the GPT-2 family: learned positional
embeddings, full multi-head attention, GELU. Current models use none of those.
`draftkv/llama.py` runs Llama- and Qwen2-style checkpoints in numpy -- RoPE,
grouped-query attention, RMSNorm, SwiGLU -- behind the same cache contract:

| model | layers | attention | verified |
|---|---|---|---|
| SmolLM2-135M | 30 | GQA 3:1 | lossless at every config |
| Qwen2.5-0.5B | 24 | GQA 7:1 | lossless at every config |

Nothing in the engine, controller, allocator or paged pool changed. The cache
stores **KV heads, not query heads**, which is what GQA alters, and that flows
through the abstraction untouched -- a random-weight Llama factory
(`demo_llama`) gives the whole path CI coverage with no download.

### The bad news: GQA has already eaten what DRAFTKV wants to save

This is the largest consequence of leaving GPT-2 behind, and it is negative.
The speedup ceiling is a ratio of memory traffic; GQA shrinks the KV cache by
its ratio, so it removes most of the KV traffic *before* DRAFTKV gets to
compress any of it.

| attention | context for a 1.5x ceiling (HBM) | with the cache offloaded |
|---|---|---|
| MHA (32 kv heads) | 19,343 | 492 |
| GQA 4:1 (8 kv) | 77,372 | 1,967 |
| GQA 8:1 (4 kv) | 154,742 | 3,934 |
| MQA (1 kv) | 618,967 | 15,736 |

Break-even moves out by almost exactly the GQA ratio
(`test_gqa_pushes_the_breakeven_context_out_by_its_ratio` pins the proportion).
On an HBM-resident cache with 4:1 GQA you need ~77k tokens before a 1.5x
ceiling exists at all. Offloading is what rescues it: there the comparison is
against PCIe rather than HBM, and break-even is a couple of thousand tokens
even under GQA.

**So on a current model the honest pitch is narrower than it looked on GPT-2:
long context, or an offloaded cache, and not much else.** The earlier regime
tables were computed with `n_kv_heads=8` and so already assumed GQA, but the
per-architecture comparison makes the size of the effect explicit.

### The early-layer concentration is a GPT-2 property

The per-layer allocator was motivated by sensitivity concentrating in a few
layers. Same probe (one layer evicted to keep-0.25, others 4-bit full,
averaged over three prompts), now across both families:

| model | family | layers | attention | damage in first 25% | top-3 sensitive |
|---|---|---|---|---|---|
| distilgpt2 | GPT-2 | 6 | MHA | 62% | 0, 3, 5 |
| gpt2 | GPT-2 | 12 | MHA | 94% | 1, 0, 11 |
| gpt2-medium | GPT-2 | 24 | MHA | 100% | 1, 2, 3 |
| **SmolLM2-135M** | Llama | 30 | GQA 3:1 | **11%** | 18, 14, 12 |
| **Qwen2.5-0.5B** | Qwen2 | 24 | GQA 7:1 | **27%** | 1, 17, 10 |

It does not transfer. On the modern models the damage is spread thinly across
the stack, the most sensitive layers sit in the *middle*, and the absolute
numbers are tiny -- the largest single-layer damage is 0.07 on SmolLM2 and 0.09
on Qwen2.5, against 3.02 on gpt2-medium. Evicting any one layer's KV barely
moves acceptance.

Previous rounds reported the concentration "sharpening with depth" from 62% to
100%. That trend was real *within the GPT-2 family* and I over-generalized it:
what the depth sweep actually varied was depth and family together, and family
turns out to be the variable that mattered.

The consequence for the design is direct and unflattering to the static
allocator: **there is much less per-layer structure to exploit on a current
model.** The adaptive allocator degrades correctly here -- with near-uniform
damages it allocates near-uniformly -- but its headline win on the simulator
assumed one clearly sensitive layer, which these models do not have.

### 8-bit is not universally free either

`8b/keep1` is exactly lossless in acceptance on distilgpt2 and costs nothing.
On Qwen2.5-0.5B it drops acceptance to 0.647. Qwen's activation outliers are
harder to quantize, and "quantization is nearly free" -- stated earlier from
GPT-2 measurements -- is another family-specific result rather than a property
of KV caches.

### The depth effect does not clearly replicate

Measured acceptance by draft depth on SmolLM2-135M at 4b/keep-0.25:

```
depth   0     1     2     3     4     5     6     7
      0.68  0.62  0.56  0.56  0.60  1.00  0.33  0.00
```

Flat, not rising -- against the strong rise seen across the GPT-2 family
(0.28 -> 1.00 on distilgpt2). Qwen2.5-0.5B is mildly rising (0.89, 0.69, 0.91,
1.00, 1.00, 0.90, 1.00, 1.00), so the effect is somewhere between absent and
weak on modern architectures. The deep tail rests on very few surviving drafts
in all cases and should not be read as a trend in either direction.

This matters for how much weight the depth-indexed controller deserves. It does
not invalidate the design: `expected_tokens_profile` reduces exactly to the
geometric closed form on a flat profile, and
`test_depth_controller_is_no_worse_on_flat_data` pins that the depth-indexed
controller costs nothing when the effect is absent. So the generalization is
safe to keep, but its *benefit* is architecture-dependent and should not be
quoted as a general property.

## Is the early-layer result a small-model artifact?

The most actionable finding here -- sensitivity concentrating in the first
layers -- was measured on a 6-layer model, which is exactly the shape of thing
that turns out to be an artifact. Checking it against depth, same probe
(one layer evicted to keep-0.25, others at 4-bit full), averaged over three
prompts:

| model | layers | most sensitive | share of damage in first 25% of stack |
|---|---|---|---|
| distilgpt2 | 6 | L0, L3, L5 | 62% |
| gpt2 | 12 | L1, L0, L11 | 94% |
| gpt2-medium | 24 | L1, L2, L3 | **100%** |

The concentration does not wash out with depth, it *sharpens*. At 24 layers,
every layer past the third is indistinguishable from lossless under this probe
while layers 1 and 2 carry damages of 3.02 and 2.11.

Two corrections to how I stated this before:

- It is **early layers**, not literally layer 0. On gpt2-medium layer 0 measures
  0.00 and the damage sits in layers 1-2.
- The deep tail is not merely tolerant, it is *flat*. Several layers measure
  slightly negative damage, i.e. indistinguishable from zero at this noise
  level, which is what makes the allocation win large.

**What this still does not establish.** All three are GPT-2 family, all under
1B, all learned-positional / MHA / GELU. A 7B model is untested, and so is any
architecture with rotary embeddings, grouped-query attention or SwiGLU -- the
things every current model actually uses. The trend across 6 -> 12 -> 24 layers
is evidence against the artifact hypothesis, not a demonstration that it holds
at scale.

## Allocating the layer budget online

The measured prompt-dependence above undermines the static allocator: a profile
taken once is a prior, not an answer. `AdaptiveLayerController` keeps the same
objective and the same exact knapsack but sources its damages from the live
verify pass.

The reason it may explore *in production* is the premise of the whole project:
**output is identical whatever the plan**, so a probe costs a little throughput
and cannot corrupt anything. An allocator for a compressor that traded against
quality could never do this.

Against a simulated content shift, at a fixed byte budget:

| | true sensitive layer | found | acceptance |
|---|---|---|---|
| content A | 1 | 1 | **0.95** |
| content B (shifted) | 4 | 4 | **0.95** |
| fixed uniform plan, same budget | — | — | 0.27 on content B |

Two design errors surfaced while building it, both now regression tests:

- **Probing only downward starves the layers that matter.** A layer at the
  floor cannot be degraded further, so as the plan compresses, the layers most
  in need of measurement become unmeasurable. Replaced with **swap probes** —
  a notch taken from one layer and given to another, always feasible and
  byte-neutral by construction. A first cut at this chose swap partners by
  position rather than bytes; notch sizes are not uniform (keep 0.5 → 1.0 costs
  twice keep 0.25 → 0.5), so ~80% of probes were silently over budget and
  rejected.
- **Separate up/down posteriors measure staleness, not sensitivity.** A layer
  pinned at the ceiling never gets an "up" observation and one at the floor
  never gets a "down" one, so the comparison between the two sides drifts into
  comparing their ages. In testing this assigned a shifted sensitivity to
  entirely the wrong layer. Replaced with **paired comparison**: each swap
  scores against the concurrent base rate and credits `+delta` to the layer
  that gained and `-delta` to the one that paid, touching both regardless of
  where they sit.

A third failure was structural: with no information the knapsack has nothing to
minimize, so it returns the *cheapest* feasible plan, acceptance collapses, the
gate switches drafting off — and with `gamma = 0` no round reports anything, so
the controller can never recover. Fixed by spending surplus budget (unspent
bytes buy nothing, so "no information" should mean "compress as little as the
budget forces") and by treating a run of gated rounds as evidence against the
plan rather than a steady state.

## Rollback under a paged cache

The hard part of a vLLM integration was never the math: it is that the
authoritative cache lives in a paged pool addressed through block tables, and
DRAFTKV rewinds it once per verify pass. On a contiguous array a rewind is a
slice; in a pool it means freeing whole blocks, keeping a partially filled tail,
and doing that every round without leaking.

`draftkv/paged.py` implements that allocator — fixed block pool, per-layer block
tables, slot addressing — and the engine's full cache is now swappable, so it
runs end to end:

| block size | blocks used | fragmentation | output identical |
|---|---|---|---|
| 4 | 168 | 0.4% | yes |
| 16 | 42 | 0.4% | yes |
| 64 | 12 | 12.9% | yes |

Identical tokens at every block size, with a rollback every round, no leaked
blocks, and pool invariants asserted after each one. `test_repeated_rollback_cycles_do_not_leak`
runs 60 reject-everything rounds, because a one-block-per-round leak would
exhaust any pool in production and pass a single-shot test.

**Be clear about what this is not.** There is no CUDA, no real paged-attention
kernel, no CUDA-graph capture, and no vLLM patch. What it establishes is that
the rollback contract survives block-table indirection — the part that would
otherwise be discovered late and expensively. The remaining integration work is
real and untouched.

## Two corrections the measurements forced

The first version of this took the problem statement's model at face value: one
acceptance rate per config, applied uniformly to every layer. Instrumenting the
reference model showed both halves of that are wrong, in ways that cost real
throughput.

### Acceptance is not flat in draft depth -- it *rises*

Measured P(accept at depth i | reached depth i), config 4b/keep0.5:

```
depth   0     1     2     3     4     5     6     7
      0.51  0.80  0.83  0.85  0.88  1.00  0.93  0.93
```

This is survivorship, not drift. Reaching depth *i* is itself evidence that the
current stretch of text drafts easily, so the conditional rate climbs. I
expected the opposite -- that the drafter's own approximate K/V would compound
error within a round and push acceptance *down*. It does not.

The consequence is not a rounding error. Fitting one alpha to this data anchors
it near the depth-0 rate, and `E(alpha, gamma)` then truncates gamma hard:

| model | gamma* | throughput under the true profile |
|---|---|---|
| depth-indexed profile | 12 | 1.99x |
| one alpha, fitted at depth 0 | 2 | 1.54x |

`expected_tokens_profile(accept)` replaces the closed form with
`1 + sum_k prod_{i<k} a_i`, and reduces to it exactly when the profile is flat
(tested). `DepthThompsonController` keeps one Beta per (config, depth) -- the
same observations, indexed rather than pooled. Online, against an environment
with this shape: **97.6% of optimal vs 88.2%** for the pooled controller, and no
worse than it when the data really is flat.

### Layers do not deserve equal bits

Acceptance with exactly one layer degraded, everything else fp16:

| layer | 2b/keep0.5 | 3b/keep0.5 | 4b/keep1 | 8b/keep1 |
|---|---|---|---|---|
| 0 | 0.31 | 0.35 | 0.69 | 1.00 |
| 1 | 0.40 | 0.59 | **0.96** | 1.00 |
| 2 | 0.29 | 0.47 | 0.91 | 1.00 |

Layer 1 shrugs off 4-bit; layer 2 does not. Uniform compression overpays in
tolerant layers and starves sensitive ones. `LayerPlan` allows a per-layer
assignment, `profile_layers` measures the table above, and `allocate` spends a
byte budget to minimize total damage. Measured acceptance at equal bytes:

| budget (B) | allocated plan | alpha | best uniform in budget | alpha |
|---|---|---|---|---|
| 84,480 | 3b x2, 4b x1 | **0.747** | 3b/keep0.5 | 0.446 |
| 115,200 | 4b x3 (uniform *is* optimal here) | 0.636 | 4b/keep1 | 0.636 |
| 145,920 | 4b x2, 8b x1 | **0.870** | 4b/keep1 | 0.636 |
| 176,640 | 4b x1, 8b x2 | **0.958** | 4b/keep1 | 0.636 |

Two things had to be right, and neither was on the first attempt:

- **The objective.** Per-layer damages are assumed to add in log-acceptance.
  On this random-weight model that looked like a poor absolute predictor; the
  real-model work below shows the diagnosis was wrong, and what was actually
  broken was the measurement. See *Fixing the wrong thing* -- the tests pin the
  ordering, not the prediction.
- **The optimizer.** Greedy over adjacent upgrades is not optimal: it cannot
  climb a layer whose value sits entirely in its top config, since every
  intermediate step scores zero gain, and it strands that layer at 2-bit while
  overspending elsewhere. `allocate` solves the multiple-choice knapsack exactly
  over a Pareto frontier of (cost, damage) states; `allocate_greedy` is kept as
  the baseline that shows why. An intermediate version bucketed the byte axis
  and was also wrong -- three layers each rounding up by a third of a bucket
  read as one bucket over budget, so an exactly-affordable uniform plan was
  rejected in favour of something worse. Exact costs remove the failure mode,
  and a regression test holds the line.

A caveat that matters for anyone profiling a real model: the acceptance
differences between neighbouring configs are often ~0.05, and a short profiling
run cannot resolve them. `SensitivityProfile.trials` reports the sample count
behind each cell for exactly this reason -- under a couple of hundred drafted
tokens the profile is noise, and the allocation degrades with it.

### One feature that did not survive measurement

Gating the draft loop on the drafter's own confidence -- stop early when its
next-token distribution goes flat -- is an obvious idea and is *not* implemented.
On the reference model every sample lands in a single confidence bucket
(point-biserial correlation with acceptance: 0.185), so there is no signal here
to build on. It may well work on a real LM with sharper distributions. Without
evidence, it stays out.

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
| `throughput.py` | `E(alpha, gamma)`, the depth-indexed `E`, the memory-traffic cost model, `ceiling()`, `best_gamma` |
| `controller.py` | Thompson sampling (pooled and depth-indexed), flat-product baseline, oracle, fixed policies |
| `layers.py` | multi-prompt sensitivity profiling, composition model, exact-knapsack allocation |
| `gpt2.py` | distilgpt2 / gpt2 / gpt2-medium in numpy (safetensors, BPE), same cache contract |
| `llama.py` | Llama / Qwen2 in numpy: RoPE, GQA, RMSNorm, SwiGLU, same cache contract |
| `adaptive.py` | online per-layer allocation via budget-neutral swap probes |
| `paged.py` | block-paged authoritative cache with rollback, the vLLM-shaped part |
| `scripts/fetch_model.py` | optional checkpoint download; real-model tests skip without it |
| `sim.py` | bandit environment with known ground-truth `alpha(c)` |
| `bench.py` | the sixteen experiments |
| `.github/workflows/tests.yml` | CI: full suite on Python 3.10-3.13, losslessness as a separate fast lane |

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
- **No 7B.** Architecture coverage now spans GPT-2 family and Llama/Qwen2
  (RoPE, GQA, RMSNorm, SwiGLU) from 82M to 0.5B, but a 7B model in fp32 needs
  ~28 GB of RAM and this environment has 15 GB, so it is genuinely out of reach
  here rather than merely skipped. `sim.plausible_alpha` remains a stylized
  shape for the simulator, not a calibration.
- **No claim that this is VeriCache.** It implements the mechanism the idea
  describes, and specifically does not assume the paper's handling of where the
  full cache lives — that is modeled explicitly here as a bandwidth and budget
  parameter so the assumption is visible instead of inherited.
