"""
reproduce_paper_results.py -- Regenerate the manuscript tables and figures WITHOUT
touching the archived results.

What it does
  1. Copies research/ into a separate working folder (default: ./_reproduction/<timestamp>/,
     which is git-ignored). The archived repository is never written to.
  2. Runs, inside the copy, the analysis scripts that produce the manuscript numbers and
     figures (Tables 3-9 and Figures 1-2; see README.md).
  3. Compares every file under the copy's results/ with the archived file of the same path:
       - identical bytes                       -> IDENTICAL
       - CSV/JSON with equal numbers (tol.)     -> EQUIVALENT (formatting only)
       - PNG with the same size and at most a few pixels differing by 1-2 intensity
         levels (font anti-aliasing differs between platforms) -> EQUIVALENT
       - anything else                          -> DIFFERENT
  4. Writes a report (reproduction_report.md) in the working folder and exits with
     status 0 only if nothing is DIFFERENT.

Requirements: the analysis environment (environment/requirements-analysis.txt).
Usage (from the repository root):
    python tools/reproduce_paper_results.py
    python tools/reproduce_paper_results.py --workdir /path/to/scratch
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESEARCH = os.path.join(REPO, "research")

# (script, extra arguments). Order matters: write_* read the outputs of the previous script.
SCRIPTS = [
    ("analysis/consolidated_final.py", []),      # Tables 3-5, Kruskal-Wallis
    ("analysis/paper_numbers.py", []),           # Tables 6-7, TOST, GridWorld ablations (Table 9)
    ("analysis/ablation_report_x1_x2.py", []),   # Table 8, CartPole rows of Table 9
    ("analysis/write_report_x1_x2.py", []),
    ("analysis/flops_vs_timing_fig.py", []),     # Figure 1
    ("analysis/roofline_analysis.py", []),       # Figure 2
]
PNG_MAX_PIXELS, PNG_MAX_LEVEL = 50, 2
NUM_RTOL, NUM_ATOL = 1e-9, 1e-12


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def numbers_equal_csv(a, b):
    import pandas as pd
    import numpy as np
    da, db = pd.read_csv(a), pd.read_csv(b)
    if list(da.columns) != list(db.columns) or da.shape != db.shape:
        return False
    for c in da.columns:
        x, y = da[c], db[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            if not np.allclose(x.values.astype(float), y.values.astype(float), rtol=NUM_RTOL, atol=NUM_ATOL, equal_nan=True):
                return False
        elif not (x.astype(str).values == y.astype(str).values).all():
            return False
    return True


def numbers_equal_json(a, b):
    def eq(x, y):
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return abs(x - y) <= NUM_ATOL + NUM_RTOL * abs(y)
        if isinstance(x, dict) and isinstance(y, dict):
            return x.keys() == y.keys() and all(eq(x[k], y[k]) for k in x)
        if isinstance(x, list) and isinstance(y, list):
            return len(x) == len(y) and all(eq(i, j) for i, j in zip(x, y))
        return x == y
    with open(a, encoding="utf-8") as fa, open(b, encoding="utf-8") as fb:
        return eq(json.load(fa), json.load(fb))


def png_equivalent(a, b):
    from PIL import Image, ImageChops
    import numpy as np
    ia, ib = Image.open(a).convert("RGBA"), Image.open(b).convert("RGBA")
    if ia.size != ib.size:
        return False, f"size {ia.size} vs {ib.size}"
    d = np.asarray(ImageChops.difference(ia, ib)).max(axis=2)
    n, mx = int((d > 0).sum()), int(d.max())
    return (n <= PNG_MAX_PIXELS and mx <= PNG_MAX_LEVEL), f"{n} pixels differ, max level {mx}"


def compare(new, old):
    if not os.path.exists(old):
        return "NEW", "not in the archived repository"
    if sha256(new) == sha256(old):
        return "IDENTICAL", ""
    ext = os.path.splitext(new)[1].lower()
    try:
        if ext == ".csv" and numbers_equal_csv(new, old):
            return "EQUIVALENT", "same values, different formatting"
        if ext == ".json" and numbers_equal_json(new, old):
            return "EQUIVALENT", "same values, different formatting"
        if ext == ".png":
            ok, why = png_equivalent(new, old)
            return ("EQUIVALENT" if ok else "DIFFERENT"), why
    except Exception as e:  # pragma: no cover
        return "DIFFERENT", f"comparison failed: {e}"
    return "DIFFERENT", "content differs"


def main():
    ap = argparse.ArgumentParser(description="Regenerate the manuscript results in a copy and compare.")
    ap.add_argument("--workdir", default=os.path.join(REPO, "_reproduction",
                                                      dt.datetime.now().strftime("%Y%m%d_%H%M%S")))
    args = ap.parse_args()
    work = os.path.abspath(args.workdir)
    if os.path.commonpath([work, RESEARCH]) == RESEARCH:
        sys.exit("The working folder must not be inside research/.")
    if os.path.exists(work):
        sys.exit(f"Working folder already exists: {work}")
    copy = os.path.join(work, "research")
    print(f"Copying research/ to {copy} ...")
    shutil.copytree(RESEARCH, copy, ignore=shutil.ignore_patterns("__pycache__", "_checkpoints", ".venv*"))

    env = dict(os.environ, MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    t_start = time.time() - 1
    log_lines = []
    for script, extra in SCRIPTS:
        print(f"Running {script} ...")
        r = subprocess.run([sys.executable, script] + extra, cwd=copy, env=env,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        log_lines.append(f"## {script} (exit {r.returncode})\n```\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}\n```\n")
        if r.returncode != 0:
            print(r.stderr)
            sys.exit(f"{script} failed; see {work}")
    with open(os.path.join(work, "run_log.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    rows = []
    for d, _, files in os.walk(os.path.join(copy, "results")):
        for fn in files:
            new = os.path.join(d, fn)
            rel = os.path.relpath(new, copy)
            old = os.path.join(RESEARCH, rel)
            if os.path.getmtime(new) < t_start:
                continue  # not rewritten by the scripts (archived input data)
            status, why = compare(new, old)
            rows.append((rel.replace(os.sep, "/"), status, why))
    counts = {s: sum(1 for r in rows if r[1] == s) for s in ("IDENTICAL", "EQUIVALENT", "DIFFERENT", "NEW")}
    rep = [f"# Reproduction report ({dt.datetime.now().isoformat(timespec='seconds')})", "",
           f"Python {sys.version.split()[0]} on {sys.platform}. Archived repository: `{REPO}` (not modified).", "",
           f"Regenerated files: {counts['IDENTICAL']} identical, {counts['EQUIVALENT']} equivalent, "
           f"{counts['DIFFERENT']} different, {counts['NEW']} new.", "",
           "| file (relative to research/) | status | note |", "|---|---|---|"]
    rep += [f"| `{p}` | {s} | {w} |" for p, s, w in sorted(rows)]
    with open(os.path.join(work, "reproduction_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(rep) + "\n")
    print("\n".join(rep))
    print(f"\nReport: {os.path.join(work, 'reproduction_report.md')}")
    sys.exit(1 if counts["DIFFERENT"] else 0)


if __name__ == "__main__":
    main()
