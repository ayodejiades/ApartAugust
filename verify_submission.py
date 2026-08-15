#!/usr/bin/env python3
"""
verify_submission.py - Mechanical pre-submission audit.

Written because on a two-person team where one person does all the writing and
analysis, that person ends up checking their own work at hour 42 on no sleep.
This replaces the second pair of eyes with something that does not get tired.

Run from the repo root:
    python verify_submission.py --draft paper.md
    python verify_submission.py --draft paper.tex --pdf report.pdf

Exit code 0 = clean, 1 = blocking issues found.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

OK, WARN, FAIL = "  ok  ", " warn ", " FAIL "
issues = {"fail": 0, "warn": 0}


def report(level, check, detail=""):
    if level == FAIL:
        issues["fail"] += 1
    elif level == WARN:
        issues["warn"] += 1
    print(f"[{level}] {check}" + (f"\n         {detail}" if detail else ""))


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True,
                              text=True, timeout=60).stdout.strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------

def check_prereg_precedes_data():
    """The prereg is worthless if it was committed after the first results."""
    prereg = sh("git log --diff-filter=A --format=%ct -- PREREGISTRATION.md | tail -1")
    results = sh("git log --diff-filter=A --format=%ct -- 'results/*_v2_results.json' | tail -1")
    if not prereg:
        return report(FAIL, "prereg committed",
                      "PREREGISTRATION.md is not in git history. Its whole value is the "
                      "timestamp. Commit it (though committing it now proves nothing).")
    if not results:
        return report(WARN, "prereg precedes data", "no results committed yet")
    if int(prereg) < int(results):
        report(OK, "prereg precedes first results commit")
    else:
        report(FAIL, "prereg precedes first results commit",
               "Results were committed BEFORE the pre-registration. Do not describe it "
               "as a pre-registration in the paper - say 'analysis plan' instead.")


def check_no_fallback_data():
    """v1's plotting script silently plotted invented numbers. Never again."""
    patterns = [
        (r"except\s+FileNotFoundError", "silent fallback on missing data"),
        (r"torch\.randn.*steer|steer.*torch\.randn", "synthetic steering vector"),
        (r"calibrated empirical|reference values", "placeholder data"),
    ]
    hits = []
    for f in Path(".").glob("*.py"):
        if f.name == "verify_submission.py":
            continue
        in_doc = None          # active triple-quote delimiter, or None
        for line_no, line in enumerate(f.read_text().splitlines(), 1):
            stripped = line.strip()
            # track docstring/comment regions so prose ABOUT the old bug
            # doesn't get flagged as the bug itself
            if in_doc:
                if in_doc in line:
                    in_doc = None
                continue
            for q in ('"""', "'''"):
                if stripped.startswith(q) and stripped.count(q) == 1:
                    in_doc = q
                    break
            if in_doc or stripped.startswith("#"):
                continue
            for pat, desc in patterns:
                if re.search(pat, line):
                    hits.append(f"{f.name}:{line_no}  {desc}")
    if hits:
        report(FAIL, "no fabrication paths in code", "\n         ".join(hits))
    else:
        report(OK, "no fabrication paths in code")


def collect_result_numbers():
    """Every float appearing anywhere in the results JSONs, at 2 and 3 dp."""
    nums = set()

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            for dp in (1, 2, 3):
                nums.add(f"{abs(float(o)):.{dp}f}")
                nums.add(f"{abs(float(o)) * 100:.{dp}f}")
    files = list(Path("results").glob("*.json")) if Path("results").exists() else []
    for f in files:
        try:
            walk(json.loads(f.read_text()))
        except Exception:
            pass
    return nums, len(files)


def check_numbers_trace(draft_path):
    """Every decimal in the draft should appear in a results file.

    This is the check that catches a stale number left in after a re-run --
    the most likely way an honest team ends up with a wrong figure in the PDF.
    """
    if not draft_path or not Path(draft_path).exists():
        return report(WARN, "numbers trace to results", "no draft given, skipped")
    result_nums, n_files = collect_result_numbers()
    if n_files == 0:
        return report(WARN, "numbers trace to results", "no results files yet")

    text = Path(draft_path).read_text()
    text = re.sub(r"```.*?```", "", text, flags=re.S)          # skip code blocks
    text = re.sub(r"\\cite\{[^}]*\}|\\ref\{[^}]*\}", "", text)  # skip refs

    KNOWN_SAFE = {
        "0.85", "0.850", "0.05", "0.050", "0.95", "0.950", "1.00", "0.25",
        "0.50", "0.75", "2.5", "97.5", "0.2", "0.7", "0.80",
    }
    orphans = []
    for m in re.finditer(r"(?<![\w.])(\d+\.\d+)(?![\w])", text):
        val = m.group(1)
        if val in KNOWN_SAFE or val in result_nums:
            continue
        if re.match(r"^(19|20)\d\d\.", val):   # years, arxiv ids
            continue
        line = text[:m.start()].count("\n") + 1
        orphans.append(f"line {line}: {val}")

    if orphans:
        report(FAIL, "every number traces to a results file",
               f"{len(orphans)} untraceable:\n         " +
               "\n         ".join(orphans[:12]) +
               ("\n         ..." if len(orphans) > 12 else "") +
               "\n         (thresholds and depths are whitelisted; these are not)")
    else:
        report(OK, "every number traces to a results file")


def check_overclaim_language(draft_path):
    if not draft_path or not Path(draft_path).exists():
        return report(WARN, "no overclaiming language", "no draft given, skipped")
    text = Path(draft_path).read_text().lower()
    banned = {
        "prove": "say 'shows' or 'suggests'",
        "proves": "say 'shows' or 'suggests'",
        "proven": "say 'shows' or 'suggests'",
        "demonstrates that": "usually an overclaim; 'is consistent with' is safer",
        "solves the": "you are not solving the individuation problem",
        "conclusively": "no",
    }
    hits = [f"'{w}' -> {fix}" for w, fix in banned.items()
            if re.search(rf"\b{re.escape(w)}\b", text)]
    if hits:
        report(FAIL, "no overclaiming language", "\n         ".join(hits))
    else:
        report(OK, "no overclaiming language")

    # the sentence that buys the most credibility with these judges
    if "moral patient" in text or "moral status" in text:
        if not re.search(r"(does not|no|nothing).{0,80}(licen|support|entail|imply)", text):
            report(WARN, "moral-status disclaimer present",
                   "You mention moral status. State plainly that no number here "
                   "licenses a moral-patienthood claim.")
        else:
            report(OK, "moral-status disclaimer present")


def check_required_sections(draft_path):
    if not draft_path or not Path(draft_path).exists():
        return report(WARN, "required sections", "no draft given, skipped")
    text = Path(draft_path).read_text().lower()
    required = {
        "abstract": "abstract",
        "related work": "related work",
        "method": "method",
        "limitation": "limitations",
        "dual-use": "dual-use / ethical appendix (REQUIRED by organizers)",
        "author contribution": "author contributions",
    }
    missing = [d for k, d in required.items() if k not in text]
    if missing:
        report(FAIL, "required sections present", "missing: " + ", ".join(missing))
    else:
        report(OK, "required sections present")

    words = len(re.findall(r"\w+", text.split("introduction")[0]))
    if words > 260:
        report(WARN, "abstract <= 150 words",
               f"~{words} words before Introduction; organizers cap the abstract at 150")


def check_gate_failures_reported(draft_path):
    """Gate-failing cells must be shown, not dropped."""
    if not Path("results").exists():
        return report(WARN, "gate-failing cells disclosed", "no results yet")
    failing = []
    for f in Path("results").glob("*_v2_results.json"):
        try:
            r = json.loads(f.read_text())
        except Exception:
            continue
        for persona, pdata in r.get("personas", {}).items():
            for dk, e in pdata.items():
                c = e.get("ceilings", {}).get("self", {}).get("ceiling_spearman_brown")
                if c is not None and c < 0.85:
                    failing.append(f"{r['model_id']} {persona} d={dk} ceiling={c:.3f}")
    if not failing:
        return report(OK, "gate-failing cells disclosed", "none present")
    if draft_path and Path(draft_path).exists():
        text = Path(draft_path).read_text().lower()
        if "ceiling" in text and ("fail" in text or "below" in text or "gate" in text):
            report(OK, "gate-failing cells disclosed", f"{len(failing)} present, discussed")
        else:
            report(FAIL, "gate-failing cells disclosed",
                   f"{len(failing)} cells failed the ceiling gate and the draft does not "
                   f"mention it. Showing a failed cell reads as rigour; hiding one does not.\n"
                   f"         " + "\n         ".join(failing[:5]))


def check_repo_state():
    if not Path(".git").exists():
        return report(FAIL, "git repo", "not a git repo - the submission needs a public repo")
    dirty = sh("git status --porcelain")
    if dirty:
        report(WARN, "working tree clean",
               f"{len(dirty.splitlines())} uncommitted files - judges clone what you pushed")
    else:
        report(OK, "working tree clean")

    remote = sh("git remote get-url origin")
    report(OK if remote else FAIL, "remote configured", remote or "no origin remote")

    n = len(list(Path("results").glob("*_v2_results.json"))) if Path("results").exists() else 0
    report(OK if n else FAIL, "results committed", f"{n} results files")

    for f in ["README.md", "PREREGISTRATION.md", "build_prompts.py", "harness_v2.py"]:
        if not Path(f).exists():
            report(FAIL, f"{f} present")


def check_docx(path="Digital_Minds_Submission.docx"):
    """Audit the built submission, not only the markdown source."""
    if not Path(path).exists():
        return report(WARN, "submission docx built", "not found; run populate_template.py")
    try:
        import subprocess
        txt = subprocess.run(["pdftotext", "-layout",
                              path.replace(".docx", ".pdf"), "-"],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception:
        txt = ""
    if not txt:
        return report(WARN, "submission docx audited",
                      "no rendered PDF; run soffice --convert-to pdf")

    leftovers = [s for s in ["Author name", "PROJECT TITLE", "How to use this template",
                             "italicized guidance", "[Reference 1]",
                             "Summarize your project", "[First contribution",
                             "Link to GitHub", "[surname]", "[repo URL]"] if s in txt]
    report(FAIL if leftovers else OK, "no template placeholders left",
           ", ".join(leftovers) if leftovers else "")

    required = {"title": "Are Referent-Specific", "abstract": "referent cancels",
                "Figure 1": "Figure 1.", "Figure 2": "Figure 2.",
                "Table 1": "Table 1.", "Table 2": "Table 2.", "Table 3": "Table 3.",
                "ethics appendix": "Dual-Use", "LLM usage": "LLM Usage",
                "contributions": "Adesegun designed"}
    absent = [k for k, v in required.items() if v not in txt]
    report(FAIL if absent else OK, "all required sections present",
           "missing: " + ", ".join(absent) if absent else "")

    pages = txt.count("\f") + 1
    report(WARN if pages > 11 else OK, "page count",
           f"{pages} pages (template recommends 4 excl. refs + appendix)")

    import re as _re
    m = _re.search(r"Abstract\s+(.*?)\n\s*\n", txt, _re.S)
    if m:
        n = len(_re.findall(r"\w+", m.group(1)))
        report(OK if 150 <= n <= 250 else WARN, "abstract length",
               f"{n} words (template asks 150-250)")


def check_controls_reported(draft_path):
    """If the control run exists, the paper must report it."""
    ctrl = list(Path("results").glob("*_controls.json")) if Path("results").exists() else []
    if not ctrl:
        return report(WARN, "control run reported", "no *_controls.json yet")
    if draft_path and Path(draft_path).exists():
        txt = Path(draft_path).read_text().lower()
        if "your colleague" in txt or "your_other" in txt or "second-person control" in txt:
            report(OK, "control run reported", f"{len(ctrl)} file(s)")
        else:
            report(FAIL, "control run reported",
                   f"{len(ctrl)} control result file(s) exist but the draft does not "
                   "mention the control. Pick a branch from RESULT_BRANCHES.md.")


def check_figures():
    figs = list(Path(".").glob("Figure*.pdf")) + list(Path(".").glob("Figure*.png"))
    if not figs:
        return report(FAIL, "figure exists", "no Figure*.pdf - judges look at it first")
    newest_result = max((f.stat().st_mtime for f in Path("results").glob("*.json")),
                        default=0) if Path("results").exists() else 0
    stale = [f.name for f in figs if f.stat().st_mtime < newest_result]
    if stale:
        report(FAIL, "figures newer than results",
               f"stale: {', '.join(stale)} - re-run analyze_and_plot.py")
    else:
        report(OK, "figures newer than results", ", ".join(f.name for f in figs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", help="paper.md or paper.tex")
    args = ap.parse_args()

    print("=" * 70)
    print("PRE-SUBMISSION AUDIT")
    print("=" * 70)
    print("\n-- integrity --")
    check_prereg_precedes_data()
    check_no_fallback_data()
    check_numbers_trace(args.draft)
    check_gate_failures_reported(args.draft)
    print("\n-- writing --")
    check_overclaim_language(args.draft)
    check_required_sections(args.draft)
    print("\n-- artifacts --")
    check_repo_state()
    check_figures()
    print("\n-- submission document --")
    check_docx()
    check_controls_reported(args.draft)

    print("\n" + "=" * 70)
    print(f"{issues['fail']} blocking, {issues['warn']} warnings")
    if issues["fail"]:
        print("\nFix the blocking items before submitting.")
        print("Then do the one thing this script cannot do:")
    else:
        print("\nMechanical checks clean. This script cannot do the last step:")
    print("  clone the repo fresh into /tmp, run the reproduction command,")
    print("  and confirm it works on a machine that is not yours.")
    print("=" * 70)
    sys.exit(1 if issues["fail"] else 0)


if __name__ == "__main__":
    main()
