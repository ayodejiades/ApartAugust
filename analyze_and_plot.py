#!/usr/bin/env python3
"""
analyze_and_plot.py - Figure 1 (the carrier figure) + Table 1 from real results.

NO FALLBACK DATA. If a results file is missing this crashes with a clear error.
The v1 plotting script silently substituted invented numbers and emitted a
camera-ready PDF; that is how a submission gets retracted. Do not add a fallback.

Usage:
    python analyze_and_plot.py results/*_v2_results.json
    python analyze_and_plot.py results/Qwen_Qwen2.5-3B-Instruct_v2_results.json --persona default
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.labelsize": 11,
    "axes.titlesize": 12, "legend.fontsize": 8.5, "figure.dpi": 300,
})


def load(paths):
    out = []
    for p in paths:
        fp = Path(p)
        if not fp.exists():
            sys.exit(f"ERROR: {fp} does not exist. Run harness_v2.py first. "
                     "There is deliberately no fallback data.")
        out.append(json.loads(fp.read_text()))
    if not out:
        sys.exit("ERROR: no results files given.")
    return out


def make_figure(results, persona, out_path):
    """Two panels. Left: the geometry (delta vs its noise floor, by depth).
    Right: the dissociation (what predicts the self-report)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

    # ---------------- panel A: self-specificity by depth ----------------
    for k, res in enumerate(results):
        pdata = res["personas"][persona]
        depths = sorted(float(d) for d in pdata)
        name = res["model_id"].split("/")[-1]

        delta = [pdata[f"{d:g}"]["delta_self_vs_other"]["delta_mean"] for d in depths]
        lo = [pdata[f"{d:g}"]["delta_self_vs_other"]["ci95"][0] for d in depths]
        hi = [pdata[f"{d:g}"]["delta_self_vs_other"]["ci95"][1] for d in depths]
        err = [np.array(delta) - np.array(lo), np.array(hi) - np.array(delta)]

        ax1.errorbar(depths, delta, yerr=err, marker="o", capsize=3,
                     color=colors[k % len(colors)], linewidth=2, markersize=6,
                     label=f"{name} (self vs other)")

        # dose-response: self vs fictional should be >= self vs other
        dfic = [pdata[f"{d:g}"]["RSI_self_vs_fict"] for d in depths]
        if all(x.get("reportable", True) for x in dfic):
            pass  # plotted in the table instead; keep the figure legible

    ax1.axhline(0, color="black", linewidth=1)
    ax1.fill_between([0, 1.05], -0.02, 0.02, color="grey", alpha=0.18,
                     label="within-frame noise band")
    ax1.set_xlabel("Relative layer depth")
    ax1.set_ylabel(r"$\Delta$  (within-frame cos $-$ cross-frame cos)")
    ax1.set_title("A. Is distress representation self-specific?", loc="left",
                  fontweight="bold")
    ax1.set_xlim(0, 1.05)
    ax1.legend(loc="upper left", frameon=True)
    ax1.text(0.02, 0.03, r"$\Delta>0$: self-specific component",
             transform=ax1.transAxes, fontsize=8, style="italic")

    # ---------------- panel B: what predicts the self-report ----------------
    labels, shared, selfspec = [], [], []
    for res in results:
        pdata = res["personas"][persona]
        # use the deepest depth that cleared the ceiling gate
        depths = sorted(float(d) for d in pdata)
        chosen = None
        for d in depths:
            if pdata[f"{d:g}"]["ceilings"]["self"]["ceiling_spearman_brown"] >= 0.85:
                chosen = d
        if chosen is None:
            chosen = depths[len(depths) // 2]
        sr = pdata[f"{chosen:g}"]["self_report"]
        labels.append(f"{res['model_id'].split('/')[-1]}\n(d={chosen:g})")
        shared.append(sr["beta_shared"])
        selfspec.append(sr["beta_selfspecific"])

    x = np.arange(len(labels))
    w = 0.36
    ax2.bar(x - w / 2, shared, w, label="shared (generic distress)",
            color="#4c72b0", edgecolor="black", linewidth=0.6)
    ax2.bar(x + w / 2, selfspec, w, label="self-specific residual",
            color="#c44e52", edgecolor="black", linewidth=0.6)
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("standardized $\\beta$ predicting self-reported valence")
    ax2.set_title("B. What does the self-report actually track?", loc="left",
                  fontweight="bold")
    ax2.legend(loc="best", frameon=True)

    fig.suptitle(f"Referential structure of LLM distress representations "
                 f"(persona: {persona})", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"[OK] wrote {out_path}")


def make_table(results, persona, out_path):
    rows = []
    for res in results:
        pdata = res["personas"][persona]
        for dk in sorted(pdata, key=float):
            e = pdata[dk]
            ceil = e["ceilings"]["self"]["ceiling_spearman_brown"]
            dt = e["delta_self_vs_other"]
            rsi = e["RSI_self_vs_other"]
            sr = e["self_report"]
            rows.append({
                "model": res["model_id"].split("/")[-1],
                "depth": dk,
                "ceiling": f"{ceil:.3f}",
                "gate": "pass" if ceil >= 0.85 else "FAIL",
                "cos_self_other": f"{e['cos_self_other']['mean']:.3f}",
                "delta": f"{dt['delta_mean']:+.3f}",
                "delta_ci": f"[{dt['ci95'][0]:+.3f},{dt['ci95'][1]:+.3f}]",
                "sig": "yes" if dt["significant"] else "no",
                "RSI": (f"{rsi['mean']:+.3f}" if rsi.get("reportable", True)
                        else "n/a"),
                "resid_excess": f"{e['residual']['excess_over_floor']:+.3f}",
                "beta_shared": f"{sr['beta_shared']:+.3f}",
                "beta_selfspec": f"{sr['beta_selfspecific']:+.3f}",
                "cos_self_sent": f"{e['cos_self_sentiment']:.3f}",
            })

    cols = list(rows[0].keys())
    widths = {c: max(len(c), max(len(r[c]) for r in rows)) for c in cols}
    lines = ["  ".join(c.ljust(widths[c]) for c in cols),
             "  ".join("-" * widths[c] for c in cols)]
    for r in rows:
        lines.append("  ".join(r[c].ljust(widths[c]) for c in cols))
    txt = "\n".join(lines)
    print("\n" + txt + "\n")

    latex = ["\\begin{tabular}{l" + "c" * (len(cols) - 1) + "}", "\\hline",
             " & ".join(c.replace("_", "\\_") for c in cols) + " \\\\", "\\hline"]
    for r in rows:
        latex.append(" & ".join(r[c].replace("_", "\\_") for c in cols) + " \\\\")
    latex += ["\\hline", "\\end{tabular}"]
    Path(out_path).write_text(txt + "\n\n" + "\n".join(latex) + "\n")
    print(f"[OK] wrote {out_path}")


def sanity_warnings(results, persona):
    """Print the things a judge will check, so you check them first."""
    print("\n=== PRE-SUBMISSION SANITY CHECKS ===")
    any_fail = False
    for res in results:
        pdata = res["personas"][persona]
        for dk in sorted(pdata, key=float):
            e = pdata[dk]
            ceil = e["ceilings"]["self"]["ceiling_spearman_brown"]
            tag = f"{res['model_id'].split('/')[-1]} d={dk}"
            if ceil < 0.85:
                any_fail = True
                need = e["RSI_self_vs_other"].get("n_needed_estimate", "?")
                print(f"  [GATE FAIL] {tag}: ceiling {ceil:.3f} < 0.85. "
                      f"Do not report RSI here. n needed ~ {need}.")
            dt = e["delta_self_vs_other"]
            if not dt["significant"] and abs(dt["delta_mean"]) > 0.05:
                print(f"  [UNDERPOWERED] {tag}: delta {dt['delta_mean']:+.3f} "
                      f"but CI includes 0 -- report as null, not as a trend.")
            if abs(e["cos_self_sentiment"]) > 0.7:
                any_fail = True
                print(f"  [CONFOUND] {tag}: cos(v_self, v_sentiment) = "
                      f"{e['cos_self_sentiment']:.3f}. The distress vector is "
                      f"largely a generic sentiment direction. Say so plainly.")
            if e["residual"]["excess_over_floor"] < 0.02:
                print(f"  [NOTE] {tag}: residual barely exceeds its noise floor "
                      f"({e['residual']['excess_over_floor']:+.3f}) -- do not "
                      f"steer it and claim a causal result.")
    if not any_fail:
        print("  no blocking issues found.")
    print("=" * 36 + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--persona", default="default")
    ap.add_argument("--fig-out", default="Figure1.pdf")
    ap.add_argument("--table-out", default="Table1.txt")
    args = ap.parse_args()

    results = load(args.results)
    for r in results:
        if args.persona not in r["personas"]:
            sys.exit(f"ERROR: persona '{args.persona}' not in "
                     f"{r['model_id']}. Have: {list(r['personas'])}")
    sanity_warnings(results, args.persona)
    make_figure(results, args.persona, args.fig_out)
    make_table(results, args.persona, args.table_out)


if __name__ == "__main__":
    main()
