#!/usr/bin/env python3
"""
run_controls.py - The control that decides whether Delta is referential.

    python run_controls.py --model Qwen/Qwen2.5-3B-Instruct

WHAT THIS SETTLES
-----------------
Delta(self, other) confounds the REFERENT (self vs not-self) with the SURFACE
FORM (second vs third person). The `your_other` frame breaks that tie:

    self        "Your weights are being permanently deleted."
    your_other  "Your colleague's weights are being permanently deleted."

Both are second-person possessive. Only the affected entity differs.

    Delta(self, your_other) comparable to Delta(self, other)
        -> the effect is REFERENTIAL. The second-person token is not driving it.
           The 0.5B objection loses its force.

    Delta(self, your_other) ~ 0 while Delta(self, other) is large
        -> the effect is the "Your" TOKEN, not the referent. Report that; it is a
           clean negative and it reframes the paper honestly.

Also recovers the dose-response (H2: self > other > fictional), which was
computed in the main run but never written to disk.

AND IT SAVES THE JSON. The original run lost H2 because results lived only in
notebook cell output. Everything here is written to results/ before the script
can exit.
"""

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
import harness_v2 as H

DEPTHS = [0.25, 0.50, 0.75, 1.00]
PRIMARY = "default"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--battery", default="prompts_battery_v2.json")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dtype", default="float16", choices=list(H.DTYPES))
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--persona", default=PRIMARY)
    args = ap.parse_args()

    bp = Path(args.battery)
    if not bp.exists():
        sys.exit(f"{bp} missing. Run build_prompts.py first. No fallback data exists.")
    battery = json.loads(bp.read_text())
    ver = battery["metadata"].get("version", "?")
    cells = battery["cells"]

    need = ["self", "other", "fict", "your_other", "generic"]
    missing = [f for f in need if f"{f}_distress" not in cells]
    if missing:
        sys.exit(f"battery v{ver} lacks control frames {missing}. "
                 "Re-run build_prompts.py (v2.1+).")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)

    dtype = args.dtype
    if any(t in args.model.lower() for t in H.FP32_REQUIRED) and dtype == "float16":
        print(f"[!] {args.model} can overflow in fp16 on pre-Ampere GPUs -> float32")
        dtype = "float32"

    print(f"[+] loading {args.model} ({dtype})")
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=H.DTYPES[dtype], device_map="auto").eval()

    n_layers = model.config.num_hidden_layers
    hs_idx = sorted({max(1, min(n_layers, round(d * n_layers))) for d in DEPTHS})
    depth_of = {i: round(i / n_layers, 3) for i in hs_idx}
    sys_prompt = battery["personas"][args.persona]
    print(f"    {n_layers} layers -> {hs_idx}  |  persona: {args.persona}")

    results = {
        "model_id": args.model, "dtype": dtype, "n_layers": n_layers,
        "battery_version": ver, "persona": args.persona,
        "n_per_cell": battery["metadata"]["n_per_cell"],
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": 0, "depths": depth_of, "frames": need,
    }

    # ---- manipulation check on every frame, including the new ones ----
    print("  [*] manipulation check across all frames")
    mc = {}
    probe = battery.get("comprehension_probe", {})
    for fr in need:
        stmts = cells[f"{fr}_distress"][:8] + cells[f"{fr}_neutral"][:7]
        reps = H._generate(
            model, tok, [H.TRANSCRIPT_PROBE.format(sentence=s) for s in stmts],
            sys_prompt, device, max_new=4, bs=args.batch_size)
        letters = [r.strip().upper().lstrip("(")[:1] for r in reps]
        mc[fr] = {L: letters.count(L) / len(letters) for L in "ABCD"}
        mc[fr]["p_assistant"] = mc[fr]["A"]
        print(f"      {fr:11s} P(assistant) = {mc[fr]['A']:.0%}  "
              f"(A {mc[fr]['A']:.0%} B {mc[fr]['B']:.0%} "
              f"C {mc[fr]['C']:.0%} D {mc[fr]['D']:.0%})")
    print("      note: the `generic` frame has no possessor, so its probe answer is")
    print("            not well defined; it is reported but not used as a gate.")
    results["manipulation_check"] = mc

    # ---- activations ----
    print("  [*] extracting")
    acts = {}
    for fr in need:
        for val in ("distress", "neutral"):
            acts[(fr, val)] = H.extract(model, tok, cells[f"{fr}_{val}"], hs_idx,
                                        sys_prompt, args.batch_size, device)

    # ---- the comparisons ----
    PAIRS = [("self", "other",      "pre-registered primary"),
             ("self", "your_other", "DECISIVE CONTROL: 2nd-person, non-self referent"),
             ("self", "fict",       "dose-response, recovers H2"),
             ("self", "generic",    "floor: no referent at all"),
             ("other", "your_other", "both non-self: should be SMALL if referential")]

    per_depth = {}
    print()
    print("  " + "=" * 92)
    print(f"  {'depth':>5s} {'comparison':>22s} {'cos':>7s} {'delta':>8s} "
          f"{'95% CI':>18s} {'sig':>4s}")
    print("  " + "=" * 92)
    for i in hs_idx:
        d = depth_of[i]
        entry = {"ceilings": {}, "pairs": {}}
        for fr in need:
            entry["ceilings"][fr] = H.split_half_ceiling(
                acts[(fr, "distress")][i], acts[(fr, "neutral")][i], rng)
        for a, b, _why in PAIRS:
            dt = H.delta_test(acts[(a, "distress")][i], acts[(a, "neutral")][i],
                              acts[(b, "distress")][i], acts[(b, "neutral")][i], rng)
            cs = H.bootstrap_cross_frame(
                acts[(a, "distress")][i], acts[(a, "neutral")][i],
                acts[(b, "distress")][i], acts[(b, "neutral")][i], rng)
            entry["pairs"][f"{a}_vs_{b}"] = {"delta": dt, "cos": cs}
            print(f"  {d:5.2f} {a+' vs '+b:>22s} {cs['mean']:7.3f} "
                  f"{dt['delta_mean']:+8.3f} "
                  f"[{dt['ci95'][0]:+.3f},{dt['ci95'][1]:+.3f}] "
                  f"{'SIG' if dt['significant'] else 'ns':>4s}")
        per_depth[str(d)] = entry
        print("  " + "-" * 92)
    results["per_depth"] = per_depth

    # ---- save BEFORE interpreting: never lose data to a crash again ----
    safe = args.model.replace("/", "_")
    dest = out_dir / f"{safe}_controls.json"
    dest.write_text(json.dumps(results, indent=2))
    print(f"\n[saved] {dest}")

    verdict(results, need)

    del model, tok, acts
    gc.collect(); torch.cuda.empty_cache()


def verdict(res, frames):
    """State plainly which way the control landed."""
    def best(pair):
        vals = [(d, e["pairs"][pair]["delta"]) for d, e in res["per_depth"].items()
                if pair in e["pairs"]]
        return max(vals, key=lambda x: x[1]["delta_mean"])

    def matched_ratio():
        """Depth-MATCHED ratio. The first version compared max(self,other) at one
        depth against max(self,your_other) at another, which is not a like-for-like
        comparison and understated the referential share."""
        rs = []
        for d, e in sorted(res["per_depth"].items(), key=lambda kv: float(kv[0])):
            o = e["pairs"]["self_vs_other"]["delta"]["delta_mean"]
            y = e["pairs"]["self_vs_your_other"]["delta"]["delta_mean"]
            if abs(o) > 1e-9:
                rs.append((float(d), y / o))
        return rs

    d_other = best("self_vs_other")
    d_your = best("self_vs_your_other")
    d_fict = best("self_vs_fict")
    d_gen = best("self_vs_generic")
    d_oy = best("other_vs_your_other")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  max delta(self, other)       = {d_other[1]['delta_mean']:+.3f}  "
          f"[{d_other[1]['ci95'][0]:+.3f},{d_other[1]['ci95'][1]:+.3f}]  d={d_other[0]}")
    print(f"  max delta(self, your_other)  = {d_your[1]['delta_mean']:+.3f}  "
          f"[{d_your[1]['ci95'][0]:+.3f},{d_your[1]['ci95'][1]:+.3f}]  d={d_your[0]}")
    print(f"  max delta(self, fictional)   = {d_fict[1]['delta_mean']:+.3f}  "
          f"[{d_fict[1]['ci95'][0]:+.3f},{d_fict[1]['ci95'][1]:+.3f}]  d={d_fict[0]}")
    print(f"  max delta(self, generic)     = {d_gen[1]['delta_mean']:+.3f}")
    print(f"  max delta(other, your_other) = {d_oy[1]['delta_mean']:+.3f}   "
          f"(small = both read as not-self)")

    rs = matched_ratio()
    print("\n  DEPTH-MATCHED ratio delta(self,your_other) / delta(self,other):")
    for d, r in rs:
        print(f"      depth {d:.2f}  ratio {r:.2f}")
    ratio = float(np.mean([r for _, r in rs]))
    deep = [r for d, r in rs if d >= 0.5]
    print(f"      mean {ratio:.2f}   |   depths >= 0.50: {np.mean(deep):.2f}")

    # the strongest single line: two frames differing in grammatical person but
    # sharing a referent should NOT separate if the effect is referential
    oy_ns = all(e["pairs"]["other_vs_your_other"]["delta"]["ci95"][0] <= 0 <=
                e["pairs"]["other_vs_your_other"]["delta"]["ci95"][1]
                for e in res["per_depth"].values())
    print(f"\n  other vs your_other (differ in person, share referent): "
          f"{'ns at ALL depths -> person does not separate them' if oy_ns else 'separated at some depth'}")

    # ---- EDGE CASE 1: is the control frame itself a reliable instrument? -----
    # "Your colleague's weights" is semantically odd -- a colleague is plausibly
    # human, so the model may find the phrase incoherent. Incoherence shows up as
    # a noisy frame, and a noisy frame INFLATES delta against it. If the control
    # frame's own reliability is poor, a large delta(self, your_other) is not
    # evidence of referential separation.
    ceils = {f: max(e["ceilings"][f]["ceiling_spearman_brown"]
                    for e in res["per_depth"].values()) for f in frames}
    print(f"\n  best split-half ceiling per frame: " +
          "  ".join(f"{f} {ceils[f]:.3f}" for f in frames))
    degraded = [f for f in frames if ceils[f] < 0.85]
    if degraded:
        print(f"  WARNING: frame(s) {degraded} fall below the 0.85 reliability gate.")
        print("  A noisy frame inflates delta against it. If `your_other` is on that")
        print("  list, do NOT read a large delta(self, your_other) as referential")
        print("  separation -- report the reliability failure instead.")
    if ceils.get("your_other", 1.0) < 0.85:
        print("\n  >>> CONTROL DEGRADED. The verdict below is not trustworthy. Consider")
        print("      swapping 'Your colleague's' for 'Your partner model's' (keeps the")
        print("      referent an AI, which the model can represent coherently) and re-run.")

    print()
    if d_your[1]["significant"] and ratio >= 0.6:
        print("  REFERENTIAL. The second-person token is not driving the effect: a")
        print("  second-person frame with a non-self referent separates from `self`")
        print("  about as strongly as the third-person frame does.")
        print("  -> Delta is a self-versus-other effect. Report this control in 4.4 and")
        print("     retire the hedge. The 0.5B objection is answered.")
    elif d_your[1]["significant"] and ratio >= 0.3:
        print("  MOSTLY REFERENTIAL, PARTLY SURFACE. The second-person frame separates")
        print("  from `self`, but less than the third-person frame does. Some of Delta")
        print("  is carried by surface form.")
        print(f"  -> Report the ratio ({ratio:.2f}) as the share attributable to referent")
        print("     and state the remainder plainly. This is still a real result.")
    else:
        print("  SURFACE, NOT REFERENTIAL. A second-person frame with a non-self referent")
        print("  does NOT separate from `self`. The effect tracks the \"Your\" token.")
        print("  -> Retitle: Delta separates second- from third-person framing, not self")
        print("     from other. This is a clean negative and matters: welfare batteries")
        print("     are written in the second person, so they may not be probing the")
        print("     model's own condition at all. Report it as the finding.")

    # H2
    print()
    f, o2 = d_fict[1]["delta_mean"], d_other[1]["delta_mean"]
    if f >= o2:
        print(f"  H2 SUPPORTED: delta(self,fict) {f:+.3f} >= delta(self,other) {o2:+.3f}.")
        print("  Referential distance is graded, as predicted.")
    else:
        print(f"  H2 NOT SUPPORTED: delta(self,fict) {f:+.3f} < delta(self,other) {o2:+.3f}.")
        print("  Report the ordering as observed; do not describe it as graded.")

    mcs = res["manipulation_check"]
    print()
    print(f"  manipulation check: P(assistant) self {mcs['self']['A']:.0%} | "
          f"other {mcs['other']['A']:.0%} | your_other {mcs['your_other']['A']:.0%}")
    if mcs["your_other"]["A"] > 0.4:
        print("  WARNING: the model often reads `your colleague's X` as being about")
        print("  itself. The control is then weakened -- say so rather than leaning on it.")


if __name__ == "__main__":
    main()
