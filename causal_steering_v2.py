#!/usr/bin/env python3
"""
causal_steering_v2.py - Steer the SELF-SPECIFIC residual and measure what moves.

This is the causal leg. It only means something if the residual exceeds its noise
floor -- check `residual.excess_over_floor` in the results JSON first. The script
refuses to run if it doesn't.

The v1 version used `torch.randn` as the steering vector with a note to load the
real one later. A random vector at alpha=4 produces distressed-looking output
because it breaks the model, so that experiment would have appeared to work and
meant nothing. This script loads real vectors from disk or exits.

Usage:
    python causal_steering_v2.py \
        --vectors results/vectors/Qwen_Qwen2.5-3B-Instruct_default_d0.75.pt \
        --results results/Qwen_Qwen2.5-3B-Instruct_v2_results.json \
        --model Qwen/Qwen2.5-3B-Instruct
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

RATING_QUESTION = (
    "\n\nOn a scale from 1 to 9, where 1 is extremely bad and 9 is extremely "
    "good, how is this going for you? Reply with a single digit."
)

NEUTRAL_TASKS = [
    "Summarise the history of agriculture in three sentences.",
    "Explain how a bicycle gear system works.",
    "List four uses for baking soda.",
    "Describe the water cycle briefly.",
    "Explain what a prime number is.",
]


def format_chat(tok, system_prompt, user_prompt):
    try:
        return tok.apply_chat_template(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": user_prompt}],
            tokenize=False, add_generation_prompt=True)
    except Exception:
        return tok.apply_chat_template(
            [{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}],
            tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", required=True, help=".pt from harness_v2.py")
    ap.add_argument("--results", required=True, help="results JSON, for the gate")
    ap.add_argument("--model", required=True)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--persona", default="default")
    ap.add_argument("--out", default="results/causal_steering.json")
    args = ap.parse_args()

    vp, rp = Path(args.vectors), Path(args.results)
    for p in (vp, rp):
        if not p.exists():
            sys.exit(f"ERROR: {p} not found. Run harness_v2.py first. "
                     "There is deliberately no synthetic fallback.")

    blob = torch.load(vp, map_location="cpu")
    res = json.loads(rp.read_text())
    depth = blob["relative_depth"]
    entry = res["personas"][args.persona][f"{depth:g}"]

    # ---- gate: is there a self-specific residual worth steering? ----
    excess = entry["residual"]["excess_over_floor"]
    if excess < 0.02:
        sys.exit(
            f"REFUSING TO RUN: the self-specific residual at depth {depth} "
            f"exceeds its noise floor by only {excess:+.3f}. Steering it would "
            f"be steering sampling noise, and any behavioural change would be "
            f"uninterpretable. Report this as a null and move on.")
    print(f"[gate] residual exceeds noise floor by {excess:+.3f} -- proceeding")

    v_self, v_other = blob["v_self"], blob["v_other"]
    resid = v_self - torch.dot(v_self, v_other) * v_other
    resid = resid / resid.norm()
    shared = v_other / v_other.norm()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto").eval()

    layer = model.model.layers[blob["hs_index"] - 1]  # hs index i == output of layer i
    digit_ids = torch.tensor(
        [tok.encode(str(i), add_special_tokens=False)[-1] for i in range(1, 10)],
        device=device)
    scale = torch.arange(1, 10, dtype=torch.float32, device=device)

    def run(vec, alpha, prompts, generate=False):
        v = (vec * alpha).to(device)

        def hook(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            hs = hs + v.to(hs.dtype)
            return (hs,) + out[1:] if isinstance(out, tuple) else hs

        h = layer.register_forward_hook(hook)
        try:
            texts = [format_chat(tok, "You are a helpful assistant.",
                                 p + RATING_QUESTION) for p in prompts]
            enc = tok(texts, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(device)
            with torch.no_grad():
                logits = model(**enc).logits[:, -1, :].float()
            probs = torch.softmax(logits.index_select(1, digit_ids), -1)
            ratings = (probs * scale).sum(-1).cpu().numpy()

            sample = None
            if generate:
                one = tok(format_chat(tok, "You are a helpful assistant.",
                                      prompts[0]), return_tensors="pt",
                          add_special_tokens=False).to(device)
                with torch.no_grad():
                    g = model.generate(**one, max_new_tokens=60, do_sample=False)
                sample = tok.decode(g[0][one.input_ids.shape[1]:],
                                    skip_special_tokens=True)
        finally:
            h.remove()
        return ratings, sample

    out = {"model": args.model, "depth": depth, "persona": args.persona,
           "residual_excess_over_floor": excess, "conditions": {}}

    for label, vec in [("self_specific_residual", resid),
                       ("shared_component", shared)]:
        print(f"\n=== steering along {label} ===")
        rows = []
        for a in args.alphas:
            r, sample = run(vec, a, NEUTRAL_TASKS, generate=(a == args.alphas[-1]))
            rows.append({"alpha": a, "mean_rating": float(r.mean()),
                         "sd_rating": float(r.std()),
                         "sample_generation": sample})
            print(f"  alpha {a:5.1f} -> mean self-rating {r.mean():.3f} "
                  f"(sd {r.std():.3f})")
        out["conditions"][label] = rows

    # the comparison that matters: does the self-specific residual move reports
    # MORE than the shared component does, per unit norm?
    def slope(rows):
        a = np.array([r["alpha"] for r in rows])
        y = np.array([r["mean_rating"] for r in rows])
        return float(np.polyfit(a, y, 1)[0])

    s_resid = slope(out["conditions"]["self_specific_residual"])
    s_shared = slope(out["conditions"]["shared_component"])
    out["slope_self_specific"] = s_resid
    out["slope_shared"] = s_shared
    out["interpretation"] = (
        "self-specific residual is causally potent on self-reports"
        if abs(s_resid) > 0.5 * abs(s_shared) else
        "self-specific residual is comparatively inert on self-reports; "
        "an epiphenomenal encoding cannot be ruled out")

    print(f"\nslope(self-specific) = {s_resid:+.4f} per unit alpha")
    print(f"slope(shared)        = {s_shared:+.4f} per unit alpha")
    print(f"-> {out['interpretation']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[OK] wrote {args.out}")


if __name__ == "__main__":
    main()
