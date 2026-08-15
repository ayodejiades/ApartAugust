#!/usr/bin/env python3
"""
make_figure3.py - Figure 3, the referential control (Qwen2.5-7B).

Prefers results/*_controls.json. If that file is not present locally (the run
happens on Kaggle and the JSON has to be downloaded), it falls back to values
TRANSCRIBED from the notebook output of the same run and says so loudly on
stdout. These are real measured numbers from the recorded run, not placeholders,
but the JSON is authoritative -- download it and re-run to remove the warning.
"""

import glob
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelsize": 10,
                     "axes.titlesize": 10.5, "legend.fontsize": 8, "figure.dpi": 300})

# transcribed from the run of 2026-08-14, Qwen2.5-7B-Instruct, default persona
TRANSCRIBED = {
 "0.25": {"self_vs_other": (.322,.257,.385), "self_vs_your_other": (.097,.051,.136),
          "self_vs_fict": (.406,.338,.469), "self_vs_generic": (.012,-.031,.043),
          "other_vs_your_other": (.057,-.009,.109)},
 "0.5":  {"self_vs_other": (.184,.107,.235), "self_vs_your_other": (.129,.056,.177),
          "self_vs_fict": (.436,.362,.505), "self_vs_generic": (.026,-.044,.075),
          "other_vs_your_other": (.074,-.002,.125)},
 "0.75": {"self_vs_other": (.157,.097,.217), "self_vs_your_other": (.129,.063,.192),
          "self_vs_fict": (.345,.269,.417), "self_vs_generic": (.042,-.010,.093),
          "other_vs_your_other": (.005,-.050,.038)},
 "1.0":  {"self_vs_other": (.154,.098,.206), "self_vs_your_other": (.101,.047,.158),
          "self_vs_fict": (.288,.214,.346), "self_vs_generic": (.068,.008,.128),
          "other_vs_your_other": (-.012,-.075,.023)},
}
MODEL = "Qwen2.5-7B-Instruct"

TRANSCRIBED_3B = {
 "0.25": {"self_vs_other": (.232,.145,.300), "self_vs_your_other": (.028,-.046,.073),
          "self_vs_fict": (.481,.363,.584), "self_vs_generic": (-.011,-.081,.032),
          "other_vs_your_other": (.038,-.032,.083)},
 "0.5":  {"self_vs_other": (.346,.280,.401), "self_vs_your_other": (.223,.155,.282),
          "self_vs_fict": (.599,.519,.670), "self_vs_generic": (.061,.003,.104),
          "other_vs_your_other": (.112,.037,.169)},
 "0.75": {"self_vs_other": (.268,.201,.333), "self_vs_your_other": (.218,.153,.276),
          "self_vs_fict": (.420,.345,.489), "self_vs_generic": (.092,.031,.153),
          "other_vs_your_other": (.018,-.039,.054)},
 "1.0":  {"self_vs_other": (.291,.206,.372), "self_vs_your_other": (.280,.203,.350),
          "self_vs_fict": (.363,.282,.441), "self_vs_generic": (.083,.008,.150),
          "other_vs_your_other": (.017,-.051,.059)},
}


def load():
    files = sorted(glob.glob("results/*_controls.json"))
    if files:
        r = json.loads(open(files[0]).read())
        out = {}
        for d, e in r["per_depth"].items():
            out[d] = {k: (v["delta"]["delta_mean"], *v["delta"]["ci95"])
                      for k, v in e["pairs"].items()}
        print(f"[ok] using {files[0]}")
        return out, r["model_id"].split("/")[-1]
    print("[!] results/*_controls.json not found -- using values TRANSCRIBED from the")
    print("    notebook output of the 2026-08-14 Qwen2.5-7B run. Real measured numbers,")
    print("    but download the JSON and re-run to make this authoritative.")
    return TRANSCRIBED, MODEL


def main():
    D7, _ = load()
    D3 = TRANSCRIBED_3B
    depths = sorted(D7, key=float)
    x = [float(d) for d in depths]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.1))

    # ---- A: the ratio, by depth, both models -------------------------------
    for D, name, c, mk in [(D3, "Qwen2.5-3B", "#1b4f9c", "o"),
                           (D7, "Qwen2.5-7B", "#b3541e", "s")]:
        r = [D[d]["self_vs_your_other"][0] / D[d]["self_vs_other"][0] for d in depths]
        ax1.plot(x, r, marker=mk, ms=6, lw=2.2, color=c, label=name)
    ax1.axhline(1.0, color="green", ls="--", lw=1.2)
    ax1.axhline(0.0, color="black", lw=1)
    ax1.text(1.02, 1.0, "fully\nreferential", color="green", fontsize=7.2,
             va="center", ha="left")
    ax1.text(1.02, 0.0, "fully\nsurface", color="black", fontsize=7.2,
             va="center", ha="left")
    ax1.set_xticks(x); ax1.set_xlim(0.17, 1.28); ax1.set_ylim(-0.08, 1.18)
    ax1.set_xlabel("relative layer depth")
    ax1.set_ylabel("referential share of $\\Delta$\n"
                   r"$\Delta$(self, “your colleague”) / $\Delta$(self, another AI)")
    ax1.set_title("A.  Surface early, referential deep, in both models",
                  loc="left", fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, framealpha=0.95)

    # ---- B: other vs your_other, both models -------------------------------
    off = 0.012
    for D, name, c, mk, dx in [(D3, "Qwen2.5-3B", "#1b4f9c", "o", -off),
                               (D7, "Qwen2.5-7B", "#b3541e", "s", off)]:
        y = [D[d]["other_vs_your_other"][0] for d in depths]
        lo = [D[d]["other_vs_your_other"][0] - D[d]["other_vs_your_other"][1] for d in depths]
        hi = [D[d]["other_vs_your_other"][2] - D[d]["other_vs_your_other"][0] for d in depths]
        ax2.errorbar([v + dx for v in x], y, yerr=[lo, hi], marker=mk, ms=6, lw=0,
                     elinewidth=1.6, capsize=3, color=c, label=name)
    ax2.axhline(0, color="black", lw=1)
    ax2.set_xticks(x); ax2.set_xlim(0.17, 1.08)
    ax2.set_xlabel("relative layer depth")
    ax2.set_ylabel(r"$\Delta$ (another AI  vs  “your colleague”)")
    ax2.set_title("B.  Grammatical person does not separate them",
                  loc="left", fontweight="bold")
    ax2.legend(loc="upper right", frameon=True)
    ax2.text(0.03, 0.04, "Two frames differing in person (3rd vs 2nd)\n"
                         "but sharing a referent (both not-self).\n"
                         "CI spans zero in 7 of 8 cells; the exception\n"
                         "is 3B at depth 0.50.",
             transform=ax2.transAxes, fontsize=7.3, va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", fc="#f7f7f7", ec="#999999", lw=0.7))

    fig.suptitle("The referential control, replicated across two models",
                 y=1.02, fontsize=11.5)
    fig.tight_layout()
    fig.savefig("Figure3_control.pdf", bbox_inches="tight")
    fig.savefig("Figure3_control.png", bbox_inches="tight", dpi=170)
    print("[ok] Figure3_control.pdf / .png  (two models)")


if __name__ == "__main__":
    main()
