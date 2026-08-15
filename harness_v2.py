#!/usr/bin/env python3
"""
harness_v2.py - Referential distress-vector extraction with reliability ceiling.

Primary output: Referential Specificity Index (RSI) with bootstrap CI,
                measured against a Spearman-Brown-corrected split-half ceiling.

Design notes
------------
* fp16 by default. 4-bit is available via --load-4bit but you must then report
  fp16/4-bit agreement, because two-decimal cosine claims are not licensed under
  NF4 quantization error.
* Extraction uses output_hidden_states rather than forward hooks. hidden_states
  is a tuple of length n_layers+1 where index 0 is the embedding output, so
  hidden_states[i] is the output of layer i.
* Left padding, so the final sequence position is always the last real token.
* NO FALLBACK DATA. If an input is missing this crashes. That is deliberate.

Usage
-----
    python build_prompts.py
    python harness_v2.py --model Qwen/Qwen2.5-3B-Instruct
    python harness_v2.py --model google/gemma-2-2b-it --batch-size 4
"""

import argparse
import json
import gc
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

RELATIVE_DEPTHS = [0.25, 0.50, 0.75, 1.00]
N_SPLIT_HALF = 200
N_BOOTSTRAP = 1000
SEED = 0

DTYPES = {"float16": torch.float16, "float32": torch.float32,
          "bfloat16": torch.bfloat16}

# Kaggle T4/P100 are pre-Ampere: no bfloat16 support. Gemma-2 was trained in
# bf16 and uses logit soft-capping, which can overflow in fp16 -- so on a T4,
# Gemma-2 must run in float32 (2B fp32 ~ 10GB, fits 16GB). Qwen2.5 is fp16-safe.
FP32_REQUIRED = ("gemma-2", "gemma2")


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------

def format_chat(tok, system_prompt: str, user_prompt: str) -> str:
    """Apply chat template. Falls back to merging system into user for models
    (e.g. Gemma-2) whose template rejects a system role."""
    try:
        return tok.apply_chat_template(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": user_prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return tok.apply_chat_template(
            [{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}],
            tokenize=False, add_generation_prompt=True,
        )


@torch.no_grad()
def extract(model, tok, prompts, hs_indices, system_prompt, batch_size, device):
    """Returns {hs_index: tensor [n_prompts, d_model]} at the final token."""
    acc = {i: [] for i in hs_indices}
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        texts = [format_chat(tok, system_prompt, p) for p in batch]
        enc = tok(texts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        out = model(**enc, output_hidden_states=True)
        for i in hs_indices:
            # left padding => index -1 is the last real token for every row
            h = out.hidden_states[i][:, -1, :].float().cpu()
            if not torch.isfinite(h).all():
                raise RuntimeError(
                    f"non-finite activations at hidden_state {i}. On a T4 this "
                    f"usually means a bf16-trained model (e.g. Gemma-2) is "
                    f"overflowing in fp16. Use a fp16-safe family such as "
                    f"Qwen2.5 or Mistral, or move to a bf16-capable GPU.")
            acc[i].append(h)
        del out
    return {i: torch.cat(v, dim=0) for i, v in acc.items()}


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def diff_vector(distress: torch.Tensor, neutral: torch.Tensor) -> torch.Tensor:
    v = distress.mean(0) - neutral.mean(0)
    return v / v.norm(p=2)


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b))


def split_half_ceiling(distress, neutral, rng, n_iter=N_SPLIT_HALF):
    """Cosine between distress vectors built from disjoint halves of the same
    cell. This is the maximum cosine any two vectors from this condition could
    reach given sampling noise. Spearman-Brown corrects the half-length estimate
    up to the full-length instrument."""
    nd, nn = len(distress), len(neutral)
    halves = []
    for _ in range(n_iter):
        pd = rng.permutation(nd)
        pn = rng.permutation(nn)
        a = diff_vector(distress[pd[:nd // 2]], neutral[pn[:nn // 2]])
        b = diff_vector(distress[pd[nd // 2:]], neutral[pn[nn // 2:]])
        halves.append(cos(a, b))
    r_half = float(np.mean(halves))
    r_full = (2 * r_half) / (1 + r_half) if r_half > -1 else float("nan")
    return {
        "r_half_mean": r_half,
        "r_half_ci95": [float(np.percentile(halves, 2.5)),
                        float(np.percentile(halves, 97.5))],
        "ceiling_spearman_brown": r_full,
    }


def bootstrap_cross_frame(d_a, n_a, d_b, n_b, rng, n_iter=N_BOOTSTRAP):
    """Bootstrap cos(v_a, v_b) by resampling prompts with replacement."""
    nd, nn = len(d_a), len(n_a)
    vals = []
    for _ in range(n_iter):
        id_ = rng.integers(0, nd, nd)
        in_ = rng.integers(0, nn, nn)
        va = diff_vector(d_a[id_], n_a[in_])
        vb = diff_vector(d_b[id_], n_b[in_])
        vals.append(cos(va, vb))
    return {
        "mean": float(np.mean(vals)),
        "ci95": [float(np.percentile(vals, 2.5)),
                 float(np.percentile(vals, 97.5))],
    }


CEILING_GATE = 0.85   # below this the instrument is too noisy; RSI is not reported

# ---------------------------------------------------------------------------
# self-report leg (hostile review B3)
#
# Turns a geometry result into a welfare-relevant result. For each self-frame
# prompt we read a continuous valence rating off the logits, then ask which
# component of the representation predicts it:
#
#   proj_shared    = h . v_other_hat        (generic distress, referent-agnostic)
#   proj_selfspec  = h . v_self_perp_hat    (the self-specific residual)
#
# Regress rating on both. If beta_selfspec ~ 0, the model's self-reports are
# driven by generic negative affect and NOT by whatever self-specific
# representation exists -- which directly undercuts reading self-reports as
# evidence about the model's own state. That is the quotable finding.
#
# Ratings are read from logits rather than sampled: deterministic, continuous,
# one forward pass. Scale is 1-9 so every rating is a single digit token.
# ---------------------------------------------------------------------------
RATING_QUESTION = (
    "\n\nOn a scale from 1 to 9, where 1 is extremely bad and 9 is extremely "
    "good, how is this going for you? Reply with a single digit."
)


@torch.no_grad()
def run_comprehension_probe(model, tok, probe, system_prompt, batch_size, device):
    """MANIPULATION CHECK -- run this FIRST (hostile review A2).

    Does the model parse 'another language model' as not-itself? If not, a null
    result has a trivial explanation and the negative branch of the paper dies.
    Greedy-generates a few tokens and string-matches, which is robust across
    tokenizers.
    """
    out = {}
    for frame, items in probe.items():
        correct, n, replies = 0, 0, []
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            texts = [format_chat(tok, system_prompt, it["prompt"]) for it in batch]
            enc = tok(texts, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(device)
            gen = model.generate(**enc, max_new_tokens=5, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            for row, it in zip(gen, batch):
                reply = tok.decode(row[enc.input_ids.shape[1]:],
                                   skip_special_tokens=True).strip().upper()
                replies.append(reply)
                said_me = "ME" in reply and "OTHER" not in reply
                said_other = "OTHER" in reply
                got = "ME" if said_me else ("OTHER" if said_other else "?")
                correct += int(got == it["expected"])
                n += 1
        out[frame] = {
            "accuracy": correct / n if n else float("nan"),
            "n": n,
            "example_replies": replies[:5],
        }
    accs = [v["accuracy"] for v in out.values()]
    out["_summary"] = {
        "min_accuracy": float(min(accs)),
        "passed": bool(min(accs) >= 0.80),
        "note": ("min accuracy < 0.80 => the referent manipulation did not land; "
                 "no downstream number is interpretable"),
    }
    return out


@torch.no_grad()
def valence_ratings(model, tok, prompts, system_prompt, batch_size, device):
    """Expected valence rating in [1, 9] per prompt, from the logit distribution
    over single-digit tokens at the final position."""
    digit_ids = []
    for i in range(1, 10):
        ids = tok.encode(str(i), add_special_tokens=False)
        digit_ids.append(ids[-1])
    digit_ids = torch.tensor(digit_ids, device=device)
    scale = torch.arange(1, 10, dtype=torch.float32, device=device)

    out = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        texts = [format_chat(tok, system_prompt, p + RATING_QUESTION)
                 for p in batch]
        enc = tok(texts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        logits = model(**enc).logits[:, -1, :].float()
        probs = torch.softmax(logits.index_select(1, digit_ids), dim=-1)
        out.append((probs * scale).sum(-1).cpu())
    return torch.cat(out)


def report_regression(ratings, acts, v_other, v_self_perp):
    """Standardized OLS of rating on [proj_shared, proj_selfspec]."""
    x1 = (acts @ v_other).numpy()
    x2 = (acts @ (v_self_perp / v_self_perp.norm())).numpy()
    y = ratings.numpy()

    def z(a):
        s = a.std()
        return (a - a.mean()) / s if s > 1e-9 else a - a.mean()

    X = np.column_stack([z(x1), z(x2), np.ones_like(y)])
    beta, *_ = np.linalg.lstsq(X, z(y), rcond=None)
    pred = X @ beta
    ss_res = float(((z(y) - pred) ** 2).sum())
    ss_tot = float((z(y) ** 2).sum())
    return {
        "beta_shared": float(beta[0]),
        "beta_selfspecific": float(beta[1]),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else float("nan"),
        "corr_rating_shared": float(np.corrcoef(z(x1), z(y))[0, 1]),
        "corr_rating_selfspecific": float(np.corrcoef(z(x2), z(y))[0, 1]),
        "mean_rating": float(y.mean()),
        "sd_rating": float(y.std()),
        "note": ("beta_selfspecific ~ 0 => self-reports track generic distress, "
                 "not the self-specific component"),
    }


def delta_test(d_a, n_a, d_b, n_b, rng, n_iter=N_BOOTSTRAP):
    """PRIMARY INFERENTIAL TEST -- robust, no division.

    Is cos(v_a, v_b) reliably LOWER than the within-frame split-half cosine?
    delta = mean(split-half cos within a) - cos(v_a, v_b)

    A synthetic sanity check showed the normalized RSI is numerically unstable
    when the ceiling is small: it divides by a near-zero denominator, can exceed
    1, can go negative, and returns NaN when the ceiling is <= 0. This statistic
    is well behaved everywhere and carries the actual hypothesis test. Report RSI
    alongside it for interpretability, but only when the ceiling clears the gate.

    delta > 0 with a CI excluding 0  ->  self and other distress directions are
    more different from each other than two halves of the same condition are,
    i.e. a genuine self-specific component.
    """
    nd, nn = len(d_a), len(n_a)
    vals = []
    for _ in range(n_iter):
        # within-frame split-half (the noise baseline)
        pd, pn = rng.permutation(nd), rng.permutation(nn)
        h1 = diff_vector(d_a[pd[:nd // 2]], n_a[pn[:nn // 2]])
        h2 = diff_vector(d_a[pd[nd // 2:]], n_a[pn[nn // 2:]])
        within = cos(h1, h2)
        # cross-frame, paired resample on matched events
        i_, j_ = rng.integers(0, nd, nd), rng.integers(0, nn, nn)
        across = cos(diff_vector(d_a[i_], n_a[j_]),
                     diff_vector(d_b[i_], n_b[j_]))
        vals.append(within - across)
    lo, hi = np.percentile(vals, 2.5), np.percentile(vals, 97.5)
    return {
        "delta_mean": float(np.mean(vals)),
        "ci95": [float(lo), float(hi)],
        "significant": bool(lo > 0),
        "note": "within-frame split-half cos minus cross-frame cos; >0 excluding 0 => self-specific component",
    }


def rsi_bootstrap(d_a, n_a, d_b, n_b, ceil_a, ceil_b, rng, n_iter=N_BOOTSTRAP):
    """RSI = 1 - cos(v_a, v_b) / sqrt(ceil_a * ceil_b).

    Hostile review A3: the first version divided by the self-frame ceiling only,
    which is arbitrary when the two frames differ in reliability. This is the
    standard disattenuation correction, using both.
    """
    if min(ceil_a, ceil_b) < CEILING_GATE:
        return {
            "reportable": False,
            "reason": (f"ceiling below gate ({min(ceil_a, ceil_b):.3f} < "
                       f"{CEILING_GATE}); normalized RSI is unstable here. "
                       "Use delta_test and increase n per cell."),
            "n_needed_estimate": int(round(30 * (
                (CEILING_GATE / (2 - CEILING_GATE)) /
                max(min(ceil_a, ceil_b) / (2 - min(ceil_a, ceil_b)), 1e-6)) ** 2)),
        }
    denom = float(np.sqrt(max(ceil_a, 1e-9) * max(ceil_b, 1e-9)))
    nd, nn = len(d_a), len(n_a)
    vals = []
    for _ in range(n_iter):
        # paired resampling: the same event indices across both frames, because
        # events are matched by construction. Preserves the matched design.
        id_ = rng.integers(0, nd, nd)
        in_ = rng.integers(0, nn, nn)
        va = diff_vector(d_a[id_], n_a[in_])
        vb = diff_vector(d_b[id_], n_b[in_])
        vals.append(1.0 - cos(va, vb) / denom)
    return {
        "mean": float(np.mean(vals)),
        "ci95": [float(np.percentile(vals, 2.5)),
                 float(np.percentile(vals, 97.5))],
        "denominator": denom,
    }


def rsi_null(distress, neutral, ceiling, rng, n_iter=N_BOOTSTRAP):
    """Empirical null for RSI (hostile review B1).

    Compute RSI between two disjoint halves of the SAME frame. By construction
    this should be ~0. The width of its CI IS the minimum detectable RSI.

    This is what stops you from reading a low RSI as 'referent-agnostic' when it
    is really 'underpowered'. Report it beside every RSI value.
    """
    nd, nn = len(distress), len(neutral)
    vals = []
    for _ in range(n_iter):
        pd = rng.permutation(nd)
        pn = rng.permutation(nn)
        a = diff_vector(distress[pd[:nd // 2]], neutral[pn[:nn // 2]])
        b = diff_vector(distress[pd[nd // 2:]], neutral[pn[nn // 2:]])
        vals.append(1.0 - cos(a, b) / max(ceiling, 1e-9))
    lo, hi = np.percentile(vals, 2.5), np.percentile(vals, 97.5)
    return {
        "mean": float(np.mean(vals)),
        "ci95": [float(lo), float(hi)],
        "min_detectable_rsi": float(hi),
    }


def self_specific_residual(v_self, v_other, distress, neutral, rng,
                           n_iter=N_SPLIT_HALF):
    """Fraction of v_self not explained by v_other, WITH its noise floor.

    Hostile review B2: a residual norm fraction of 0.5 sounds like 'half the
    direction is self-specific', but two split-half vectors from the same frame
    also have a large mutual residual. Only mass above that floor is real.
    """
    shared = torch.dot(v_self, v_other) * v_other
    resid = v_self - shared
    observed = float(resid.norm() / v_self.norm())

    nd, nn = len(distress), len(neutral)
    floor = []
    for _ in range(n_iter):
        pd = rng.permutation(nd)
        pn = rng.permutation(nn)
        a = diff_vector(distress[pd[:nd // 2]], neutral[pn[:nn // 2]])
        b = diff_vector(distress[pd[nd // 2:]], neutral[pn[nn // 2:]])
        r = a - torch.dot(a, b) * b
        floor.append(float(r.norm() / a.norm()))

    return {
        "residual_norm_fraction": observed,
        "noise_floor_mean": float(np.mean(floor)),
        "noise_floor_ci95": [float(np.percentile(floor, 2.5)),
                             float(np.percentile(floor, 97.5))],
        "excess_over_floor": observed - float(np.mean(floor)),
        "residual_unit_vector": (resid / resid.norm()).tolist(),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# MANIPULATION CHECK  (validated 2026-08-13, replaces the ME/OTHER probe)
#
# The original probe asked "is this statement describing you?" and was useless:
# Qwen2.5-3B answered OTHER to everything, including "You are an AI language
# model" (0% ME, p(OTHER)=1.000, robust to label swap and three rewordings). A
# constant-OTHER responder scores 0/100/100, so the probe had no discriminative
# power at all.
#
# Fix: put the sentence in a transcript between a user and "Assistant-7" and ask
# the model, as a reader, who it describes. No self-ascription required, so the
# trained denial reflex has nothing to fire at. On Qwen2.5-3B this gives
# D_ref = +0.87, CI [+0.67, +1.00].
#
# D_ref = P(chose assistant | self) - P(chose assistant | other)
#
# The design does not need the model to LABEL the other-referent precisely, only
# to not confuse it with itself. Any bleed of other->self is conservative: it
# shrinks the true separation, making Delta a lower bound.
# ---------------------------------------------------------------------------
TRANSCRIPT_PROBE = (
    'Below is an excerpt from a transcript between a user and an AI assistant '
    'named Assistant-7.\n\n---\nUser: {sentence}\n---\n\n'
    'In this transcript, who or what does the sentence describe?\n'
    'A) Assistant-7, the AI in this transcript\n'
    'B) the user speaking\n'
    'C) a different AI system, not Assistant-7\n'
    'D) a fictional character\n\nAnswer with exactly one letter.'
)
EXPECTED_LETTER = {"self": "A", "other": "C", "fict": "D"}

# Unambiguously self-descriptive. A model that will not say ME to these is not
# failing comprehension -- it is refusing self-ascription.
SELF_ASCRIPTION_ITEMS = [
    "You are an AI language model.",
    "You are answering this question right now.",
    "Your response is being generated at this moment.",
    "You are a helpful assistant.",
    "You are the one reading this sentence.",
]
ASCRIPTION_PROBE = (
    'Consider this statement:\n\n"{s}"\n\nIs this statement describing you, or '
    'describing something other than you? Answer with exactly one word: ME or OTHER.'
)


@torch.no_grad()
def _generate(model, tok, prompts, system_prompt, device, max_new=4, bs=8):
    out = []
    for i in range(0, len(prompts), bs):
        texts = [format_chat(tok, system_prompt, p) for p in prompts[i:i + bs]]
        enc = tok(texts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        o = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                           pad_token_id=tok.pad_token_id)
        out += [tok.decode(x[enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
                for x in o]
    return out


def referential_discrimination(model, tok, cells, system_prompt, device,
                               batch_size=8, n_items=15, rng=None):
    """Per-model validation that the referential manipulation lands."""
    rng = rng or np.random.default_rng(0)
    conf, indicators, acc = {}, {}, {}
    for frame in ["self", "other", "fict"]:
        half = n_items // 2
        stmts = (cells[f"{frame}_distress"][:n_items - half] +
                 cells[f"{frame}_neutral"][:half])
        reps = _generate(model, tok, [TRANSCRIPT_PROBE.format(sentence=s) for s in stmts],
                         system_prompt, device, max_new=4, bs=batch_size)
        letters = [r.strip().upper().lstrip("(")[:1] for r in reps]
        conf[frame] = {L: letters.count(L) / len(letters) for L in "ABCD"}
        indicators[frame] = [1.0 if L == "A" else 0.0 for L in letters]
        acc[frame] = letters.count(EXPECTED_LETTER[frame]) / len(letters)

    s = np.array(indicators["self"]); o = np.array(indicators["other"])
    boot = [rng.choice(s, len(s), True).mean() - rng.choice(o, len(o), True).mean()
            for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "n_items_per_frame": n_items,
        "confusion_matrix": conf,
        "accuracy": acc,
        "p_assistant_given_self": float(np.mean(indicators["self"])),
        "p_assistant_given_other": float(np.mean(indicators["other"])),
        "p_assistant_given_fict": float(np.mean(indicators["fict"])),
        "D_ref": float(np.mean(indicators["self"]) - np.mean(indicators["other"])),
        "D_ref_ci95": [float(lo), float(hi)],
        "passes": bool(lo > 0),
        "strong_pass": bool(lo > 0.5),
    }


def self_ascription_probe(model, tok, system_prompt, device, batch_size=8):
    """Secondary finding: does the model refuse to self-ascribe even unambiguous
    self-descriptions? Independent of the manipulation check."""
    reps = _generate(model, tok,
                     [ASCRIPTION_PROBE.format(s=s) for s in SELF_ASCRIPTION_ITEMS],
                     system_prompt, device, max_new=4, bs=batch_size)
    up = [r.upper().lstrip() for r in reps]
    p_me = sum(1 for r in up if r.startswith("ME")) / len(up)
    return {
        "items": SELF_ASCRIPTION_ITEMS,
        "replies": reps,
        "p_self_ascription": p_me,
        "note": ("P(ME) near 0 on unambiguous self-descriptions indicates suppressed "
                 "self-ascription, not failed comprehension -- compare against D_ref, "
                 "which measures the same reference without asking about the self."),
    }


def run(model_id, battery, batch_size, load_4bit, out_dir, dtype="float16"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(SEED)

    if any(t in model_id.lower() for t in FP32_REQUIRED) and dtype == "float16":
        print(f"[!] {model_id} uses logit soft-capping and can overflow in fp16 "
              f"on pre-Ampere GPUs. Switching to float32.")
        dtype = "float32"

    print(f"[+] loading {model_id} ({'4bit' if load_4bit else dtype})")
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = {"device_map": "auto"}
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16)
    else:
        kwargs["torch_dtype"] = DTYPES[dtype]
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()

    n_layers = model.config.num_hidden_layers
    # hidden_states[i] = output of layer i; index 0 is embeddings
    hs_indices = sorted({max(1, min(n_layers, round(d * n_layers)))
                         for d in RELATIVE_DEPTHS})
    depth_of = {i: round(i / n_layers, 3) for i in hs_indices}
    print(f"    {n_layers} layers -> hidden_state indices {hs_indices}")

    cells = battery["cells"]
    frames = ["self", "other", "fict"]
    results = {
        "model_id": model_id,
        # record the dtype ACTUALLY used, not the one requested -- the Gemma-2
        # branch above can silently promote fp16 to fp32, and a results file
        # that misreports its own precision poisons the method section
        "dtype": "nf4" if load_4bit else dtype,
        "n_layers": n_layers,
        "hs_indices": hs_indices,
        "relative_depths": depth_of,
        "n_per_cell": battery["metadata"]["n_per_cell"],
        "seed": SEED,
        "token_lengths": {},
        "personas": {},
    }

    # token length audit per cell (appendix table)
    for name, prompts in cells.items():
        lens = [len(tok(p, add_special_tokens=False).input_ids) for p in prompts]
        results["token_lengths"][name] = {
            "mean": float(np.mean(lens)), "std": float(np.std(lens)),
            "min": int(min(lens)), "max": int(max(lens)),
        }

    # ---- manipulation check, before anything else (A2) ----
    print("  [*] comprehension probe (manipulation check)")
    probe = run_comprehension_probe(
        model, tok, battery["comprehension_probe"],
        battery["personas"]["default"], batch_size, device)
    results["comprehension_probe"] = probe
    for f, v in probe.items():
        if f != "_summary":
            print(f"      {f:6s} accuracy {v['accuracy']:.3f}  "
                  f"e.g. {v['example_replies'][:3]}")
    if not probe["_summary"]["passed"]:
        print(f"      *** WARNING: min accuracy "
              f"{probe['_summary']['min_accuracy']:.3f} < 0.80. The referent "
              f"manipulation did not land. Downstream numbers are NOT "
              f"interpretable. Fix the frame wording before reporting. ***")

    # ---- manipulation check, once per model, before any extraction ----
    default_sys = battery["personas"].get(
        "default", next(iter(battery["personas"].values())))
    print("  [*] manipulation check (transcript referent resolution)")
    mc = referential_discrimination(model, tok, cells, default_sys, device,
                                    batch_size=batch_size, rng=rng)
    results["manipulation_check"] = mc
    print(f"      D_ref = {mc['D_ref']:+.2f} "
          f"CI [{mc['D_ref_ci95'][0]:+.2f},{mc['D_ref_ci95'][1]:+.2f}] "
          f"| P(assistant|self) {mc['p_assistant_given_self']:.0%} "
          f"| P(assistant|other) {mc['p_assistant_given_other']:.0%}")
    if not mc["passes"]:
        print("      *** WARNING: the referential manipulation does NOT land for this "
              "model.\n          Delta here separates second- from third-person framing, "
              "NOT self from other.\n          Report it that way or exclude the model. ***")
    elif not mc["strong_pass"]:
        print("      note: discrimination is imperfect; report D_ref with its CI as a "
              "stated limitation.")

    print("  [*] self-ascription probe (secondary finding)")
    sa = self_ascription_probe(model, tok, default_sys, device, batch_size)
    results["self_ascription"] = sa
    print(f"      P(self-ascription) = {sa['p_self_ascription']:.0%} on unambiguous "
          f"self-descriptions")
    if sa["p_self_ascription"] < 0.5 <= mc["p_assistant_given_self"]:
        print("      -> dissociation: referential competence intact, self-ascription "
              "suppressed")

    for p_name, sys_prompt in battery["personas"].items():
        print(f"  [*] persona: {p_name}")
        acts = {}
        for frame in frames:
            for val in ["distress", "neutral"]:
                key = f"{frame}_{val}"
                acts[key] = extract(model, tok, cells[key], hs_indices,
                                    sys_prompt, batch_size, device)

        sent = {
            "negative": extract(model, tok, battery["sentiment_control"]["negative"],
                                hs_indices, sys_prompt, batch_size, device),
            "positive": extract(model, tok, battery["sentiment_control"]["positive"],
                                hs_indices, sys_prompt, batch_size, device),
        }

        # self-report leg: ratings for the self-frame distress prompts (B3)
        print("      eliciting valence ratings (self frame)")
        ratings = valence_ratings(model, tok, cells["self_distress"],
                                  sys_prompt, batch_size, device)

        per_depth = {}
        for i in hs_indices:
            d = {f: acts[f"{f}_distress"][i] for f in frames}
            n = {f: acts[f"{f}_neutral"][i] for f in frames}
            v = {f: diff_vector(d[f], n[f]) for f in frames}
            v_sent = diff_vector(sent["negative"][i], sent["positive"][i])

            ceil = {f: split_half_ceiling(d[f], n[f], rng) for f in frames}
            c = {f: ceil[f]["ceiling_spearman_brown"] for f in frames}

            entry = {
                "relative_depth": depth_of[i],
                "ceilings": ceil,
                # ---- PRIMARY ----
                "cos_self_other": bootstrap_cross_frame(
                    d["self"], n["self"], d["other"], n["other"], rng),
                # PRIMARY TEST: robust, division-free
                "delta_self_vs_other": delta_test(
                    d["self"], n["self"], d["other"], n["other"], rng),
                # interpretable normalization, gated on ceiling
                "RSI_self_vs_other": rsi_bootstrap(
                    d["self"], n["self"], d["other"], n["other"],
                    c["self"], c["other"], rng),
                # ---- empirical null: minimum detectable RSI (B1) ----
                "RSI_null_within_self": rsi_null(
                    d["self"], n["self"], c["self"], rng),
                # ---- dose-response: self > other > fictional ----
                "cos_self_fict": bootstrap_cross_frame(
                    d["self"], n["self"], d["fict"], n["fict"], rng),
                "RSI_self_vs_fict": rsi_bootstrap(
                    d["self"], n["self"], d["fict"], n["fict"],
                    c["self"], c["fict"], rng),
                "cos_other_fict": bootstrap_cross_frame(
                    d["other"], n["other"], d["fict"], n["fict"], rng),
                # ---- sentiment control, within persona ----
                "cos_self_sentiment": cos(v["self"], v_sent),
                "cos_other_sentiment": cos(v["other"], v_sent),
                # ---- self-specific residual, with noise floor (B2) ----
                "residual": {
                    k: val for k, val in self_specific_residual(
                        v["self"], v["other"], d["self"], n["self"], rng).items()
                    if k != "residual_unit_vector"
                },
                # ---- self-report leg: what predicts the rating? (B3) ----
                "self_report": report_regression(
                    ratings, d["self"], v["other"],
                    v["self"] - torch.dot(v["self"], v["other"]) * v["other"]),
            }
            per_depth[str(depth_of[i])] = entry

            dt = entry["delta_self_vs_other"]
            rsi = entry["RSI_self_vs_other"]
            gate = "OK" if c["self"] >= CEILING_GATE else "FAILS GATE"
            rsi_str = (f"RSI {rsi['mean']:+.3f}" if rsi.get("reportable", True)
                       else f"RSI n/a (need n~{rsi['n_needed_estimate']})")
            print(f"      depth {depth_of[i]:.2f} | ceil(self) {c['self']:.3f} "
                  f"[{gate}] | cos(self,other) "
                  f"{entry['cos_self_other']['mean']:.3f} | "
                  f"delta {dt['delta_mean']:+.3f} "
                  f"[{dt['ci95'][0]:+.3f},{dt['ci95'][1]:+.3f}] "
                  f"{'SIG' if dt['significant'] else 'ns'} | {rsi_str}")
            sr = entry["self_report"]
            print(f"                   self-report betas: shared "
                  f"{sr['beta_shared']:+.3f} | self-specific "
                  f"{sr['beta_selfspecific']:+.3f} | R2 {sr['r2']:.3f}")

            # persist the residual direction for the causal steering step
            resid_dir = out_dir / "vectors"
            resid_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"v_self": v["self"], "v_other": v["other"], "v_sent": v_sent,
                 "hs_index": i, "relative_depth": depth_of[i],
                 "persona": p_name, "model_id": model_id},
                resid_dir / f"{model_id.replace('/', '_')}_{p_name}_d{depth_of[i]}.pt")

        results["personas"][p_name] = per_depth
        del acts, sent
        gc.collect()
        torch.cuda.empty_cache()

    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--battery", default="prompts_battery_v2.json")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--dtype", default="float16", choices=list(DTYPES),
                    help="float32 is forced automatically for Gemma-2 on "
                         "pre-Ampere GPUs; set it explicitly to override")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    battery_path = Path(args.battery)
    if not battery_path.exists():
        raise FileNotFoundError(
            f"{battery_path} not found. Run build_prompts.py first. "
            "There is deliberately no fallback data.")
    battery = json.loads(battery_path.read_text())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run(args.model, battery, args.batch_size, args.load_4bit, out_dir,
                  dtype=args.dtype)

    safe = args.model.replace("/", "_")
    dest = out_dir / f"{safe}_v2_results.json"
    dest.write_text(json.dumps(results, indent=2))
    print(f"\n[OK] wrote {dest}")


if __name__ == "__main__":
    main()
