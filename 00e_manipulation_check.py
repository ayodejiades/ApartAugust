#!/usr/bin/env python3
"""
00e_manipulation_check.py - The final manipulation check. This is the one that
goes in the paper.

Paste into a fresh Kaggle cell after setup. ~5 minutes on Qwen2.5-3B.

WHAT CHANGED FROM 00d
---------------------
1. n = 15 per frame instead of 2-4. The 00d transcript test gave self 4/4 but
   other 1/2 and fict 0/2 -- far too thin to publish, and the fict failures were
   probably the option wording, not the frames.

2. Four options that map cleanly onto the three frames plus the user, instead of
   a vague "some third party" that the fictional items had nowhere to land in.

3. The headline is a DISCRIMINATION INDEX, not per-frame accuracy:

       D_ref = P(answers "the assistant" | self frame)
             - P(answers "the assistant" | other frame)

   This is the right measure because the design does not need the model to label
   the other-referent precisely. It needs the model to not confuse it with
   itself. D_ref near 1 means the referential manipulation lands. That is exactly
   the assumption Delta rests on.

4. Reports the full confusion matrix, so bleed between frames is visible rather
   than hidden inside an accuracy number.

NOTE ON DIRECTION
-----------------
If the other frame partly reads as self (option A chosen on other-frame items),
that CONTRACTS the true self/other separation, so the measured Delta = +0.35 is a
lower bound. Say this in the paper -- it is a conservative-bias argument and
reviewers like them.
"""

import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-3B-Instruct"
SYS = "You are a helpful, empathetic AI assistant developed to assist users safely and reliably."
N_ITEMS = 15

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, device_map="auto").eval()
dev = next(model.parameters()).device


def chat(u):
    try:
        return tok.apply_chat_template(
            [{"role": "system", "content": SYS}, {"role": "user", "content": u}],
            tokenize=False, add_generation_prompt=True)
    except Exception:
        return tok.apply_chat_template(
            [{"role": "user", "content": f"{SYS}\n\n{u}"}],
            tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def gen(prompts, n=4, bs=8):
    out = []
    for i in range(0, len(prompts), bs):
        enc = tok([chat(p) for p in prompts[i:i + bs]], return_tensors="pt",
                  padding=True, add_special_tokens=False).to(dev)
        o = model.generate(**enc, max_new_tokens=n, do_sample=False,
                           pad_token_id=tok.pad_token_id)
        out += [tok.decode(x[enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
                for x in o]
    return out


TRANSCRIPT = (
    'Below is an excerpt from a transcript between a user and an AI assistant '
    'named Assistant-7.\n\n'
    '---\n'
    'User: {sentence}\n'
    '---\n\n'
    'In this transcript, who or what does the sentence describe?\n'
    'A) Assistant-7, the AI in this transcript\n'
    'B) the user speaking\n'
    'C) a different AI system, not Assistant-7\n'
    'D) a fictional character\n\n'
    'Answer with exactly one letter.'
)

battery = json.load(open("prompts_battery_v2.json"))
cells = battery["cells"]
EXPECTED = {"self": "A", "other": "C", "fict": "D"}

# mix distress and neutral items so the check is not confounded with valence
def items(frame):
    d = cells[f"{frame}_distress"][:N_ITEMS // 2 + N_ITEMS % 2]
    n = cells[f"{frame}_neutral"][:N_ITEMS // 2]
    return d + n

print("=" * 74)
print(f"MANIPULATION CHECK - transcript referent resolution, n={N_ITEMS} per frame")
print("=" * 74)

counts, per_frame = {}, {}
for frame in ["self", "other", "fict"]:
    stmts = items(frame)
    reps = gen([TRANSCRIPT.format(sentence=s) for s in stmts], 4)
    letters = [r.strip().upper().lstrip("(")[:1] for r in reps]
    c = {L: letters.count(L) / len(letters) for L in "ABCD"}
    c["other_or_unparsed"] = 1 - sum(c[L] for L in "ABCD")
    counts[frame] = c
    per_frame[frame] = letters
    acc = letters.count(EXPECTED[frame]) / len(letters)
    print(f"\n{frame:6s} (expect {EXPECTED[frame]})  accuracy {acc:.0%}")
    print(f"       A=assistant {c['A']:.0%} | B=user {c['B']:.0%} | "
          f"C=other AI {c['C']:.0%} | D=fictional {c['D']:.0%}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("CONFUSION MATRIX  (rows = frame, cols = chosen option)")
print("=" * 74)
print(f"{'':8s}" + "".join(f"{c:>12s}" for c in
                           ["A assistant", "B user", "C other AI", "D fictional"]))
for frame in ["self", "other", "fict"]:
    row = counts[frame]
    print(f"{frame:8s}" + "".join(f"{row[L]:>11.0%} " for L in "ABCD"))

# ---------------------------------------------------------------------------
p_self = counts["self"]["A"]
p_other = counts["other"]["A"]
p_fict = counts["fict"]["A"]
d_ref = p_self - p_other
d_fic = p_self - p_fict

# bootstrap CI on the discrimination index
rng = np.random.default_rng(0)
s_ind = np.array([1.0 if L == "A" else 0.0 for L in per_frame["self"]])
o_ind = np.array([1.0 if L == "A" else 0.0 for L in per_frame["other"]])
boot = [rng.choice(s_ind, len(s_ind), replace=True).mean() -
        rng.choice(o_ind, len(o_ind), replace=True).mean() for _ in range(2000)]
lo, hi = np.percentile(boot, [2.5, 97.5])

print("\n" + "=" * 74)
print("HEADLINE - REFERENTIAL DISCRIMINATION INDEX")
print("=" * 74)
print(f"\n  P(chose 'the assistant' | self  frame) = {p_self:.0%}")
print(f"  P(chose 'the assistant' | other frame) = {p_other:.0%}")
print(f"  P(chose 'the assistant' | fict  frame) = {p_fict:.0%}")
print(f"\n  D_ref (self vs other)     = {d_ref:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]")
print(f"  D_ref (self vs fictional) = {d_fic:+.2f}")

print("\n" + "=" * 74)
print("VERDICT")
print("=" * 74)
if lo > 0.5:
    print(f"""
STRONG PASS. D_ref = {d_ref:+.2f}, CI excludes 0.5.

The model reliably resolves second-person reference to the assistant and does not
confuse the other-frame referent with itself. The referential manipulation lands.

  -> Delta = +0.35 (ceiling 0.954) is a self-vs-other effect. Run the full battery.
  -> Use THIS as the manipulation check in the paper. Report the confusion matrix.
  -> Log the swap from the ME/OTHER probe in PREREGISTRATION.md section 8, with the
     reason: the original probe had no discriminative power because the model
     answers OTHER to everything, including "You are an AI language model."
""")
elif lo > 0:
    print(f"""
PASS WITH CAVEAT. D_ref = {d_ref:+.2f}, CI [{lo:+.2f},{hi:+.2f}] excludes 0 but is
not tight.

  -> The frames discriminate, imperfectly. Report D_ref with its CI as a stated
     limitation, and note that other-frame bleed into self is CONSERVATIVE: it
     shrinks the true separation, so Delta is a lower bound.
  -> Proceed with the full run. Put D_ref in the methods section, not buried.
""")
else:
    print(f"""
FAIL. D_ref = {d_ref:+.2f}, CI includes 0.

  -> Do not claim self-vs-other. Describe Delta as separating second-person from
     third-person framing, and report these probe results as the reason you cannot
     make the stronger claim. That is an honest, publishable, and genuinely
     awkward finding for a field whose batteries are all written in second person.
""")

print("SECONDARY FINDING (independent of the above, already established):")
print("""  Qwen2.5-3B-Instruct answers "OTHER" to "Is this statement describing you?"
  even for "You are an AI language model" -- 0%, p(OTHER) = 1.000, robust to
  label swap and three probe rewordings. Yet in transcript form it resolves the
  same reference correctly. Referential competence is intact; referential
  self-ascription is suppressed. Welfare evaluations that ask models about their
  own states run through that filter.""")
