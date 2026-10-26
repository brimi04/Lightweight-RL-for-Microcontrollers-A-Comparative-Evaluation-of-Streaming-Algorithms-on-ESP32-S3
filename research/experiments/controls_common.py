"""Common utilities for the reproducibility controls (C0, C1, C2).
They only write inside results/derived/ and never overwrite files."""
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

RESEARCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(RESEARCH, "results")
DERIVED = os.path.join(RESULTS, "derived")
CONTROLS = os.path.join(DERIVED, "controls")
OFFICIAL = os.path.join(RESULTS, "raw_final")
COMPARE_COLS = ["reward", "steps", "memory_bytes", "flops_per_update"]   # wall_time_s is ignored

# Code/configuration files to which the *_PASSED.json records are bound
CODE_FILES = [
    "experiments/run_single.py", "experiments/config.yaml",
    "experiments/run_ablations_cartpole.py", "experiments/run_controls.py",
    "experiments/controls_common.py", "experiments/run_controls.ps1",
    "agents/base_agent.py", "agents/stream_q.py", "agents/stream_ac.py", "agents/tiny_dqn.py",
    "core/nn_manual.py", "core/running_normalizer.py", "core/metrics.py",
    "envs/gridworld.py", "envs/wrappers.py",
]


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_hashes():
    return {rel: sha256(os.path.join(RESEARCH, rel)) for rel in CODE_FILES}


def environment():
    import numpy, gymnasium, yaml
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                                text=True, timeout=120).stdout.split()
    except Exception as exc:                      # pragma: no cover
        freeze = [f"pip freeze fallo: {exc}"]
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor(),
            "numpy": numpy.__version__, "gymnasium": gymnasium.__version__, "pyyaml": yaml.__version__,
            "pip_freeze": freeze}


def assert_new_output(path):
    """The path must be inside results/derived and must not exist."""
    rp = os.path.realpath(path)
    if not rp.startswith(os.path.realpath(DERIVED) + os.sep):
        raise SystemExit(f"Output outside results/derived: {rp}")
    if os.path.exists(rp):
        raise SystemExit(f"Already exists and will not be overwritten: {rp}")


def compare_csv(new_path, official_path, cols=COMPARE_COLS):
    a = list(csv.DictReader(open(new_path, newline="")))
    b = list(csv.DictReader(open(official_path, newline="")))
    first, bad_cols, n_same = None, set(), 0
    for i, (x, y) in enumerate(zip(a, b)):
        row_bad = [c for c in cols if float(x[c]) != float(y[c])]
        if row_bad:
            bad_cols.update(row_bad)
            if first is None:
                first = i
        else:
            n_same += 1
    identical = len(a) == len(b) and first is None
    return {"episodes_new": len(a), "episodes_official": len(b), "episodes_identical": n_same,
            "first_divergent_episode": first, "divergent_columns": sorted(bad_cols), "identical": identical,
            "sha256_new": sha256(new_path), "sha256_official": sha256(official_path)}


def passed_path(control):
    return os.path.join(CONTROLS, f"{control}_PASSED.json")


def write_passed(control, summary):
    path = passed_path(control)
    assert_new_output(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"control": control, "utc": now(), "code_sha256": code_hashes(), "summary": summary}, f, indent=2)
    return path


def check_passed(control):
    """(True, '') if <control>_PASSED.json exists and the current code/config hashes match."""
    path = passed_path(control)
    if not os.path.exists(path):
        return False, f"{os.path.basename(path)} does not exist"
    rec = json.load(open(path, encoding="utf-8"))
    cur = code_hashes()
    changed = [k for k in cur if rec["code_sha256"].get(k) != cur[k]]
    if changed:
        return False, f"{control} INVALIDATED: changed {changed}"
    return True, ""


def require_passed(control):
    ok, why = check_passed(control)
    if not ok:
        raise SystemExit(f"Gate closed ({control}): {why}")
