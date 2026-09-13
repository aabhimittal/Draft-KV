"""Experiments.  `python -m draftkv.bench all`"""

from __future__ import annotations

import argparse
import time

import numpy as np

from .config import CompressionConfig, LayerPlan, default_arms
from .controller import (
    DepthThompsonController,
    FixedPolicy,
    FlatBandit,
    OraclePolicy,
    ThompsonController,
)
from .engine import DraftKVEngine
from .layers import allocate, allocate_greedy, measure_alpha, profile_layers
from .model import demo_model
from .sim import BanditEnv, plausible_alpha, rising_profile, run_bandit
from .throughput import CostModel, best_gamma, best_gamma_profile, expected_tokens

LINE = "-" * 78


def _h(t: str) -> None:
    print(f"\n{LINE}\n{t}\n{LINE}")


# ---------------------------------------------------------------- regimes
def regimes() -> None:
    _h("1. When is DRAFTKV worth turning on?  (ceiling = speedup at alpha->1)")
    print("Both draft and verify stream the full weights, so weight traffic is a")
    print("wash. Every saved byte is a KV byte -- the win only exists where KV")
    print("traffic rivals weight traffic: long context, big batch, or offload.\n")
    cfg = CompressionConfig(4, 0.5)
    setups = [
        ("HBM cache, batch 1", CostModel()),
        ("HBM cache, batch 32", CostModel(batch=32)),
        ("offloaded cache (PCIe5)", CostModel(full_kv_bandwidth=6.4e10, hbm_kv_budget=1.0e9)),
    ]
    print(f"{'setup':<26}" + "".join(f"{c:>13}" for c in ("ctx 1k", "ctx 32k", "ctx 256k")))
    for name, cm in setups:
        cells = []
        for ctx in (1024, 32768, 262144):
            g, t = best_gamma(cm, cfg, 0.9, ctx, 16)
            cells.append(f"{cm.ceiling(cfg, ctx):>6.2f}x g*={g:<3}")
        print(f"{name:<26}" + "".join(f"{c:>13}" for c in cells))
    print("\nAt 1k context the ceiling is ~1.0: the controller must switch drafting")
    print("off, and it does -- gamma*=0 falls out of the same argmax.")


# ---------------------------------------------------------------- gamma math
def gamma_curve() -> None:
    _h("2. E(alpha, gamma) and the optimal draft length")
    print("E saturates at 1/(1-alpha) while cost grows linearly in gamma, so")
    print("gamma* is finite and rises steeply with alpha.\n")
    cm = CostModel(full_kv_bandwidth=6.4e10, hbm_kv_budget=1.0e9)
    print(f"{'alpha':>6}{'E(a,4)':>9}{'1/(1-a)':>10}{'gamma*':>8}{'speedup':>9}")
    for a in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
        g, t = best_gamma(cm, CompressionConfig(4, 0.5), a, 32768, 16)
        print(f"{a:>6.2f}{expected_tokens(a, 4):>9.2f}{1/(1-a):>10.1f}{g:>8}{t*cm.t_baseline(32768):>9.2f}x")


# ---------------------------------------------------------------- losslessness
def losslessness(n_new: int = 40) -> None:
    _h("3. Output identity: compression cannot change what is emitted")
    m = demo_model()
    rng = np.random.default_rng(0)
    prompt = [int(x) for x in rng.integers(0, m.vocab_size, 256)]
    base = DraftKVEngine(m, seed=0).generate_baseline(prompt, n_new, 0.0)
    print(f"{'config':<16}{'gamma':>6}{'identical':>11}{'alpha_obs':>11}{'tok/verify':>12}")
    for cfg in [CompressionConfig(16, 1.0), CompressionConfig(8, 1.0),
                CompressionConfig(4, 1.0), CompressionConfig(4, 0.5),
                CompressionConfig(2, 0.25)]:
        for g in (4,):
            r = DraftKVEngine(m, FixedPolicy(cfg, g), seed=0).generate(prompt, n_new, 0.0)
            ok = "yes" if r.tokens == base else "NO"
            print(f"{cfg.label():<16}{g:>6}{ok:>11}{r.acceptance_rate:>11.3f}{r.tokens_per_verify:>12.2f}")
    print("\nSame tokens at 2 bits with 75% of the cache thrown away. That is the")
    print("premise: quality is no longer a function of the compression setting.")


# ---------------------------------------------------------------- controller
def controller_study(rounds: int = 3000) -> None:
    _h("4. Learning alpha(c) online -- factored vs flat bandit")
    cm = CostModel(full_kv_bandwidth=6.4e10, hbm_kv_budget=1.0e9)
    arms = default_arms()
    alphas = {c: plausible_alpha(c) for c in arms}
    env = BanditEnv(alphas, ctx=32768, seed=3)
    opt_cfg, opt_g, opt_t = env.true_best(cm)
    print(f"ground truth optimum: {opt_cfg.label()} gamma={opt_g} "
          f"({opt_t*cm.t_baseline(env.ctx):.2f}x baseline), {len(arms)} configs\n")

    fixed_scores = {
        (c, g): run_bandit(FixedPolicy(c, g), BanditEnv(alphas, 32768, seed=3), cm, rounds)
        for c in arms for g in (2, 4, 8)
    }
    (bc, bg), best_fixed = max(fixed_scores.items(), key=lambda kv: kv[1]["mean_throughput"])
    mean_fixed = float(np.mean([v["mean_throughput"] for v in fixed_scores.values()]))

    runs = {
        "Thompson (factored)": ThompsonController(cm, arms, seed=1),
        "Thompson, unpriced switch": ThompsonController(cm, arms, switch_amortize=10**9, seed=1),
        "Thompson, hard 32-commit": ThompsonController(cm, arms, commit_rounds=32, seed=1),
        "flat bandit (c,gamma)": FlatBandit(cm, arms, seed=1),
        "oracle": OraclePolicy(cm, alphas),
    }
    print(f"{'policy':<24}{'speedup':>9}{'% of opt':>10}{'switches':>10}{'final pick':>20}")
    base = cm.t_baseline(env.ctx)
    print(f"{'best fixed (hindsight)':<24}{best_fixed['mean_throughput']*base:>9.2f}x"
          f"{100*best_fixed['mean_throughput']/opt_t:>9.1f}%{0:>10}"
          f"{bc.label() + ' g=' + str(bg):>20}")
    print(f"{'avg fixed (no hindsight)':<24}{mean_fixed*base:>9.2f}x"
          f"{100*mean_fixed/opt_t:>9.1f}%{0:>10}{'-':>20}")
    for name, ctrl in runs.items():
        out = run_bandit(ctrl, BanditEnv(alphas, ctx=32768, seed=3), cm, rounds)
        cfg, g = out["final_pick"]
        print(f"{name:<24}{out['mean_throughput']*cm.t_baseline(env.ctx):>9.2f}x"
              f"{100*out['mean_throughput']/opt_t:>9.1f}%"
              f"{out['switches']:>10}"
              f"{cfg.label() + ' g=' + str(g):>20}")
    warm = ThompsonController(cm, arms, seed=1)
    run_bandit(warm, BanditEnv(alphas, 32768, seed=3), cm, 3000)
    tail = run_bandit(warm, BanditEnv(alphas, 32768, seed=3), cm, 1000)
    print(f"{'Thompson, after warm-up':<24}{tail['mean_throughput']*base:>9.2f}x"
          f"{100*tail['mean_throughput']/opt_t:>9.1f}%{tail['switches']:>10}"
          f"{tail['final_pick'][0].label() + ' g=' + str(tail['final_pick'][1]):>20}")
    print("\nRead this honestly: in a *stationary* world a fixed config chosen with")
    print("hindsight is already near-optimal, and the bandit cannot beat it while it")
    print("is still exploring -- the cold-start row pays a real tax. Its case is that")
    print("it lands on that arm without hindsight, stays far above the config you")
    print("would pick blind throughout, and keeps working when the content changes")
    print("(experiment 5), which no fixed choice does.")
    print("The flat bandit explores |C|x|gamma| arms to learn a quantity that only")
    print("depends on c; solving gamma from the cost model instead is the single")
    print("largest structural win in the controller.")


def nonstationary(rounds: int = 4000) -> None:
    _h("5. Content shift: prose -> code halfway through")
    # Budget tight enough that the *argmax* moves with content, not just the
    # margin -- otherwise "tracking" is untestable and the experiment is theater.
    cm = CostModel(full_kv_bandwidth=6.4e10, hbm_kv_budget=3.0e8)
    arms = default_arms()
    easy = {c: plausible_alpha(c, 0.3) for c in arms}     # tolerant content
    hard = {c: plausible_alpha(c, 2.0) for c in arms}     # brittle content
    print(f"optimum on tolerant content: {BanditEnv(easy, 32768).true_best(cm)[0].label()}  ->  "
          f"on brittle content: {BanditEnv(hard, 32768).true_best(cm)[0].label()}\n")
    for label, decay in (("decay=1.0 (no forgetting)", 1.0), ("decay=0.97", 0.97)):
        ctrl = ThompsonController(cm, arms, decay=decay, seed=2)
        picks = []
        for phase, alphas in (("easy", easy), ("hard", hard)):
            out = run_bandit(ctrl, BanditEnv(alphas, ctx=32768, seed=4), cm, rounds // 2)
            picks.append((phase, out["final_pick"], out["mean_throughput"] / out["optimal_throughput"]))
        s = "  ".join(f"{p}: {c.label()} g={g} ({f:.0%} of opt)" for p, (c, g), f in picks)
        print(f"{label:<28}{s}")
    print("\nA stale posterior keeps drafting aggressively into content that no")
    print("longer tolerates it. Discounting the Beta counts is what makes the")
    print("controller a tracker rather than an estimator -- but note it only")
    print("recovers part of the gap inside this window, and it costs a little")
    print("in a stationary world. Forgetting is a tuned trade, not a free win.")


def context_gating() -> None:
    _h("6. Self-gating by context length")
    cm = CostModel(full_kv_bandwidth=6.4e10, hbm_kv_budget=1.0e9)
    arms = default_arms()
    alphas = {c: plausible_alpha(c) for c in arms}
    print(f"{'ctx':>8}{'pick':>22}{'gamma':>7}{'predicted speedup':>20}")
    for ctx in (128, 512, 2048, 8192, 32768, 131072):
        ctrl = ThompsonController(cm, arms, seed=5)
        for _ in range(400):
            cfg, g = ctrl.select(ctx)
            if g:
                ctrl.update(cfg, g, BanditEnv(alphas, ctx, seed=6).pull(cfg, g), ctx)
        cfg, g, t = ctrl.best_known(ctx)
        sp = t * cm.t_baseline(ctx)
        gate = g if sp >= ctrl.min_speedup else 0
        print(f"{ctx:>8}{cfg.label():>22}{gate:>7}{sp:>19.2f}x")
    print("\nTwo separate effects, worth not conflating:")
    print(" * short context -> the predicted win falls under the gate, and the")
    print("   controller declines to draft at all (gamma=0).")
    print(" * mid context   -> the drafter's mirror fits in HBM uncompressed, so")
    print("   the right answer is self-speculation with *no* compression.")
    print("   Compression only starts earning its keep once the mirror stops")
    print("   fitting, which is the trade the memory budget actually encodes.")


def end_to_end(n_new: int = 240) -> None:
    _h("7. End to end on the reference model")
    m = demo_model()
    cm = CostModel(n_layers=m.n_layers, n_kv_heads=m.n_heads, d_head=m.d_head,
                   weight_bytes=2.0e6, full_kv_bandwidth=6.4e8, bandwidth=2.0e10,
                   hbm_kv_budget=2.0e5)
    rng = np.random.default_rng(7)
    prompt = [int(x) for x in rng.integers(0, m.vocab_size, 512)]
    base = DraftKVEngine(m, seed=0).generate_baseline(prompt, n_new, 0.0)

    ctrl = ThompsonController(cm, default_arms(), gamma_max=8, seed=11)
    t0 = time.perf_counter()
    r = DraftKVEngine(m, ctrl, seed=0).generate(prompt, n_new, 0.0)
    wall = time.perf_counter() - t0
    print(f"identical to full-KV decoding : {r.tokens == base}")
    print(f"verify passes / new tokens    : {r.verify_passes} / {r.new_tokens} "
          f"({r.tokens_per_verify:.2f} tokens per pass)")
    print(f"observed acceptance rate      : {r.acceptance_rate:.3f}")
    print(f"drafter cache rebuilds        : {r.cache_rebuilds} (config switches over {len(r.rounds)} rounds)")
    print(f"numpy wall clock              : {wall:.2f}s (reference impl, not a speed claim)")
    print("\nNote the round count: 15 arms over a few dozen rounds is squarely in")
    print("the exploration phase, so the alphas below are not converged. This")
    print("experiment demonstrates correctness under a live controller; experiment")
    print("4 is where convergence is measured.")
    print("\nlearned alpha per config:")
    for cfg, a in sorted(ctrl.alpha_estimates().items()):
        print(f"  {cfg.label():<16}{a:.3f}")
    used = {}
    for rd in r.rounds:
        used[rd.cfg.label()] = used.get(rd.cfg.label(), 0) + 1
    top = sorted(used.items(), key=lambda x: -x[1])[:5]
    print("\nmost played configs:", ", ".join(f"{k} x{v}" for k, v in top))


def depth_profile(rounds: int = 2000) -> None:
    _h("8. Acceptance is not flat in draft depth -- and it rises")
    print("Measured on the reference model, P(accept at depth i | reached i):")
    print("  4b/keep0.5 :  0.51  0.80  0.83  0.85  0.88  1.00  0.93  0.93")
    print("Not drift -- survivorship. Reaching depth i is itself evidence that")
    print("this stretch drafts easily, so the conditional rate climbs. A single")
    print("alpha fitted to that data lands near the depth-0 rate and truncates")
    print("gamma far too early.\n")

    cm = CostModel(full_kv_bandwidth=6.4e10, hbm_kv_budget=1.0e9)
    cfg = CompressionConfig(4, 0.5)
    rising = rising_profile(0.51, 12)
    g_prof, t_prof = best_gamma_profile(cm, cfg, rising, 32768)
    g_iid, _ = best_gamma(cm, cfg, 0.51, 32768, 12)
    print(f"{'model':<34}{'gamma*':>8}{'true throughput':>18}")
    for name, g in (("depth profile (correct)", g_prof), ("one alpha fitted at depth 0", g_iid)):
        t = cm.throughput_profile(cfg, rising[:g], 32768)
        print(f"{name:<34}{g:>8}{t * cm.t_baseline(32768):>17.2f}x")

    arms = default_arms()
    alphas = {c: plausible_alpha(c) for c in arms}
    profiles = {c: rising_profile(alphas[c], 12) for c in arms}
    env = lambda: BanditEnv(alphas, 32768, seed=3, profiles=profiles)
    opt = env().true_best(cm, 12)[2]
    print(f"\nOnline, against an environment with this shape ({rounds} rounds):")
    print(f"{'controller':<34}{'speedup':>9}{'% of opt':>10}")
    for name, ctrl in (
        ("depth-indexed Thompson", DepthThompsonController(cm, arms, gamma_max=12, seed=1)),
        ("one-alpha Thompson", ThompsonController(cm, arms, gamma_max=12, seed=1)),
    ):
        o = run_bandit(ctrl, env(), cm, rounds)
        print(f"{name:<34}{o['mean_throughput'] * cm.t_baseline(32768):>9.2f}x"
              f"{100 * o['mean_throughput'] / opt:>9.1f}%")
    print("\nSame data, indexed by depth instead of pooled. The i.i.d. model is not")
    print("merely imprecise here -- it is biased toward short drafts.")


def layer_allocation(n_new: int = 160) -> None:
    _h("9. Per-layer bit allocation beats uniform compression at equal bytes")
    m = demo_model()
    rng = np.random.default_rng(0)
    # several prompts, not one: a single-prompt profile measures the prompt
    prompts = [[int(x) for x in rng.integers(0, m.vocab_size, 320)] for _ in range(3)]
    prompt = prompts[0]
    cands = [CompressionConfig(2, 0.5), CompressionConfig(3, 0.5),
             CompressionConfig(4, 1.0), CompressionConfig(8, 1.0)]
    prof = profile_layers(m, prompts, cands, n_new=n_new)

    print("acceptance with ONE layer degraded (others fp16):")
    print(f"{'layer':>6}" + "".join(f"{c.label():>14}" for c in cands))
    for L in range(m.n_layers):
        print(f"{L:>6}" + "".join(f"{prof.alpha[(L, c)]:>14.2f}" for c in cands))
    print(f"\nSame bits, very different cost. Profiled over {prof.n_prompts} prompts;"
          f" min drafted tokens behind a cell: {prof.min_trials},"
          f" sd(log alpha) across prompts: {prof.noise:.2f}.")
    print("A damage smaller than that sd is not a measurement. On distilgpt2 the")
    print("same figure is ~1.2, which is why the real-model profile is only")
    print("trustworthy as a ranking (experiment 11).")

    nkv, dh, ctx = m.n_heads, m.d_head, 400
    ub = lambda c: LayerPlan.uniform(c, m.n_layers).per_layer_bytes(ctx, nkv, dh)
    print(f"\nmeasured acceptance at equal byte budgets:")
    print(f"{'budget (B)':>11}  {'exact-DP plan':<30}{'alpha':>7}  "
          f"{'greedy plan':<30}{'alpha':>7}  {'best uniform':<12}{'alpha':>7}")
    for frac in (0.3, 0.5, 0.7, 0.9):
        b = ub(cands[0]) + frac * (ub(cands[-1]) - ub(cands[0]))
        dp = allocate(prof, b, ctx, nkv, dh, m.n_layers)
        gr = allocate_greedy(prof, b, ctx, nkv, dh, m.n_layers)
        uc = max([c for c in cands if ub(c) <= b], key=ub)
        print(f"{int(b):>11}  {dp.label():<30}{measure_alpha(m, prompt, dp, n_new=n_new):>7.3f}  "
              f"{gr.label():<30}{measure_alpha(m, prompt, gr, n_new=n_new):>7.3f}  "
              f"{uc.label():<12}"
              f"{measure_alpha(m, prompt, LayerPlan.uniform(uc, m.n_layers), n_new=n_new):>7.3f}")
    print("\nTwo things had to be right for this to work.")
    print(" * The objective: how per-layer damages compose. Log-additivity is the")
    print("   obvious guess and it is wrong in *both* directions depending on the")
    print("   model -- it over-predicts damage here (errors overlap) and")
    print("   under-predicts it on distilgpt2 (errors amplify). `SensitivityProfile.p`")
    print("   generalizes it to a power mean and `fit_composition` fits that one")
    print("   scalar from measured mixed plans, searching across p = 1 rather than")
    print("   assuming a side. The optimizer is untouched: minimizing")
    print("   (sum D**p)**(1/p) is minimizing sum D**p, still separable.")
    print("   The tests pin the ordering, not the absolute prediction.")
    print(" * The optimizer: greedy over adjacent upgrades is not optimal, though on")
    print("   this particular profile it happens to find the same plans -- the")
    print("   damages here are smoothly graded, which is the easy case. It breaks")
    print("   when a layer's value sits entirely in its top config: every")
    print("   intermediate step scores zero gain, so greedy never climbs and")
    print("   strands that layer at 2-bit while overspending elsewhere")
    print("   (test_exact_dp_beats_greedy_on_its_failure_mode pins that). Exact")
    print("   knapsack DP over the Pareto frontier is optimal either way, and")
    print("   recovers the uniform plan whenever uniform genuinely is best.")


def real_model(n_new: int = 24) -> None:
    _h("10. On a real open-source model (distilgpt2)")
    try:
        from .gpt2 import load_real_model
        m, tok = load_real_model()
    except Exception as e:                                   # noqa: BLE001
        print(f"skipped: {e}")
        print("run `python scripts/fetch_model.py` to enable this experiment.")
        return

    text = ("Memory bandwidth, not arithmetic, is the binding constraint on modern "
            "inference hardware. Every generation of accelerator widens the gap "
            "between compute throughput and the rate at which parameters can be "
            "delivered, and every generation of software invents a new way to hide "
            "it: caching, tiling, quantisation, speculation. ") * 4
    ids = tok.encode(text)[:256]
    base = DraftKVEngine(m, seed=0).generate_baseline(list(ids), n_new, 0.0)
    print(f"prompt {len(ids)} tokens; baseline continues:\n  {tok.decode(base[len(ids):])!r}\n")

    print(f"{'config':<14}{'identical':>11}{'alpha':>8}{'tok/verify':>12}")
    for cfg in (CompressionConfig(16, 1.0), CompressionConfig(8, 1.0),
                CompressionConfig(4, 1.0), CompressionConfig(4, 0.5),
                CompressionConfig(2, 0.25)):
        r = DraftKVEngine(m, FixedPolicy(cfg, 4), seed=0).generate(list(ids), n_new, 0.0)
        ok = "yes" if r.tokens == base else "NO"
        print(f"{cfg.label():<14}{ok:>11}{r.acceptance_rate:>8.3f}{r.tokens_per_verify:>12.2f}")

    print("\nTwo things the random-weight model got wrong, and one it got right:")
    print(" * Quantization is close to free: 8-bit is exact and 4-bit stays high")
    print("   (0.86-1.00 depending on the prompt), where the toy model put 4-bit at")
    print("   0.66. Real KV caches are far more quantization-tolerant than a")
    print("   random-weight proxy suggests.")
    print(" * Eviction is the damaging axis: dropping tokens is what collapses")
    print("   acceptance, not narrowing them. So on a real model the budget worth")
    print("   allocating per layer is tokens, not bits.")
    print(" * Losslessness holds exactly, which is the part that had to transfer.")
    print("\nNote how much the numbers move with the prompt -- that is not incidental,")
    print("it is the dominant effect, and experiment 11 measures it.")


def prompt_dependence() -> None:
    _h("11. Acceptance is a property of the content, not just the config")
    try:
        from .gpt2 import load_real_model
        m, tok = load_real_model()
    except Exception as e:                                   # noqa: BLE001
        print(f"skipped: {e}")
        return

    prompts = {
        "tech":  "Memory bandwidth, not arithmetic, is the binding constraint on modern "
                 "inference hardware. Every generation of accelerator widens the gap. ",
        "prose": "She had not expected the letter to arrive so late in the season, nor to "
                 "find it waiting on the hall table beneath a pile of unopened bills. ",
        "code":  "def merge(left, right):\n    out = []\n    i = j = 0\n    while i < "
                 "len(left) and j < len(right):\n        out.append(left[i]); i += 1\n",
        "list":  "1. Preheat the oven. 2. Combine the flour and butter. 3. Add cold water. "
                 "4. Rest the dough. 5. Roll it out thinly. 6. Line the tin. ",
    }
    ref, probe = CompressionConfig(4, 1.0), CompressionConfig(4, 0.25)
    print(f"acceptance with ONE layer evicted to {probe.label()}, others {ref.label()}:\n")
    print(f"{'prompt':<7}{'ref':>7}" + "".join(f"{'L' + str(i):>7}" for i in range(m.n_layers))
          + "   most sensitive")
    rows = {}
    for name, text in prompts.items():
        ids = tok.encode(text * 6)[:288]
        a_ref = measure_alpha(m, ids, LayerPlan.uniform(ref, m.n_layers), 6, 96, 0)
        row = []
        for L in range(m.n_layers):
            cfgs = [ref] * m.n_layers
            cfgs[L] = probe
            row.append(measure_alpha(m, ids, LayerPlan(tuple(cfgs)), 6, 96, 0))
        rows[name] = row
        # a tie is not a winner: only name a layer when the spread is real
        worst = f"L{int(np.argmin(row))}" if (max(row) - min(row)) > 0.02 else "-- (flat)"
        print(f"{name:<7}{a_ref:>7.2f}" + "".join(f"{v:>7.2f}" for v in row)
              + f"   {worst}")

    def spearman(a, b) -> float:
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        if np.std(ra) == 0 or np.std(rb) == 0:
            return float("nan")
        return float(np.corrcoef(ra, rb)[0, 1])

    live = [k for k, v in rows.items() if max(v) - min(v) > 0.02]
    print("\nrank agreement between prompts that show any spread:")
    for i, a in enumerate(live):
        for b in live[i + 1:]:
            print(f"  {a:>5} vs {b:<5} rho={spearman(rows[a], rows[b]):+.2f}")

    print("\nRead the columns, not the rows. The absolute numbers swing enormously")
    print("with content -- the same plan can measure ~0.95 on one passage and ~0.10")
    print("on another, and a passage may show no sensitivity at all -- but layer 0")
    print("is the most sensitive layer on every prompt that has a most-sensitive")
    print("layer, and the orderings agree.")
    print("\nSo the offline profile is a *ranking* device and nothing more. That is")
    print("all the allocator consumes, which is lucky, because the absolute damage")
    print("prediction does not survive a change of content.")


ALL = {
    "regimes": regimes,
    "gamma": gamma_curve,
    "lossless": losslessness,
    "controller": controller_study,
    "nonstationary": nonstationary,
    "gating": context_gating,
    "e2e": end_to_end,
    "depth": depth_profile,
    "layers": layer_allocation,
    "real": real_model,
    "prompts": prompt_dependence,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="DRAFTKV experiments")
    ap.add_argument("which", nargs="?", default="all", choices=["all", *ALL])
    a = ap.parse_args()
    for name, fn in ALL.items():
        if a.which in ("all", name):
            fn()


if __name__ == "__main__":
    main()
