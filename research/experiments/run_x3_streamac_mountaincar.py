"""
run_x3_streamac_mountaincar.py -- Protocol X3: Stream-AC/MountainCar with
the DOCUMENTED configuration (config.yaml, no override: alpha_actor=0.01,
alpha_critic=0.02, gamma=0.99, entropy_coef=0.05 -- the one returned by
run_single.resolve_params(cfg, "stream_ac", "mountaincar")), 10 seeds,
2000 episodes. Standalone script: it does NOT modify run_single.py; it calls
run_single.run() directly WITHOUT params_override (it uses exactly
config.yaml, like run_controls.run_c0() for the other combinations).

Unlike C0/C1/C2 and X1/X2, X3 is NOT a reproducibility control
with a pass/fail criterion: there is already preserved evidence (see
results/derived/linux_exploratory_verification_20261019/README.md, lines
35-37) that Stream-AC/MountainCar is an edge case whose exact
reproducibility against raw_final is not guaranteed across platforms. The goal
of X3 is to generate, in the Windows .venv already validated by C0/C1/C2, a
set of 10 seeds with complete provenance (hashes, environment, command),
and to report the REAL comparison with raw_final in an INFORMATIVE way: whether
each episode matches or not is recorded, but the script does NOT
stop on discrepancies (because exact reproducibility is not expected;
stopping would apply a criterion that does not fit this case).

Stages:
  --stage run   : runs the 10 seeds and writes the CSV + run_meta.json +
                  the informative comparison with raw_final (report.md,
                  results.json). It has no prior control gate (there is no
                  "C3"), but it DOES check that it does not overwrite anything existing.
  --dry-run     : shows the resolved configuration (parameters, episodes,
                  analytical memory/FLOPs) without running any episode or
                  creating anything.

Outputs (never in raw_*/ or ablations/; never overwrites):
  results/derived/ablations_mountaincar_2000/stream_ac_documented/stream_ac_mountaincar_{seed}.csv
  results/derived/ablations_mountaincar_2000/stream_ac_documented/run_meta.json
  results/derived/controls/X3_streamac_mountaincar/{report.md,results.json}
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import yaml

RESEARCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RESEARCH)

import gymnasium as gym

from experiments.run_single import run, resolve_params, make_agent, CONFIG_PATH
from experiments import controls_common as cc

AGENT, ENV = "stream_ac", "mountaincar"
RESULTS = os.path.join(RESEARCH, "results")
OUT_DIR = os.path.join(RESULTS, "derived", "ablations_mountaincar_2000", "stream_ac_documented")
REPORT_DIR = os.path.join(RESULTS, "derived", "controls", "X3_streamac_mountaincar")
OFFICIAL_DIR = os.path.join(RESULTS, "raw_final")
SEEDS = list(range(10))
EXPECTED_EPISODES, EXPECTED_MAX_STEPS = 2000, 200
EXPECTED_PARAMS = {"alpha_actor": 0.01, "alpha_critic": 0.02, "gamma": 0.99, "entropy_coef": 0.05}

CRIT = ("INFORMATIVE comparison (not pass/fail) of reward, steps, memory_bytes and "
        "flops_per_update episode by episode against results/raw_final/ "
        "(wall_time_s is ignored). It does not stop on discrepancies: exact "
        "reproducibility is not expected in this case, see the script docstring.")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance():
    return {"utc": cc.now(), "argv": sys.argv, "command": os.environ.get("CONTROL_COMMAND", " ".join(sys.argv)),
            "environment": cc.environment(), "code_sha256": cc.code_hashes()}


def guard_out(path):
    rp = os.path.realpath(path)
    if "derived" not in rp:
        raise SystemExit(f"The output must be inside results/derived: {rp}")


def check_config(cfg, episodes):
    params = resolve_params(cfg, AGENT, ENV)
    assert params == EXPECTED_PARAMS, f"documented parameters differ from the expected ones: {params}"
    assert cfg["envs"][ENV]["max_episode_steps"] == EXPECTED_MAX_STEPS
    assert episodes == EXPECTED_EPISODES, f"protocol X3 requires {EXPECTED_EPISODES} episodes"
    return params


def report(rows, t0):
    n_id = sum(r["identical"] for r in rows)
    lines = [f"# Informative control X3 -- Stream-AC/MountainCar, documented configuration", "",
             f"UTC date: {cc.now()} - duration: {time.time() - t0:.0f} s", "",
             f"Criterion: {CRIT}", "",
             "| seed | episodes (new/official) | identical | first divergence | columns | result |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['seed']} | {r['episodes_new']}/{r['episodes_official']} | {r['episodes_identical']} | "
                     f"{r['first_divergent_episode'] if r['first_divergent_episode'] is not None else '-'} | "
                     f"{','.join(r['divergent_columns']) or '-'} | {'IDENTICO' if r['identical'] else 'DIVERGE'} |")
    lines += ["", f"**SUMMARY: {n_id}/{len(rows)} seeds identical to raw_final (informative, not a gate).**", "",
             "See `results.json` for full hashes and environment."]
    with open(os.path.join(REPORT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(REPORT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"n_identical": n_id, "n_total": len(rows), "rows": rows, **provenance()}, f, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["run"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    episodes = cfg["envs"][ENV]["episodes"]
    params = check_config(cfg, episodes)

    env = gym.make("MountainCar-v0")
    agent = make_agent(AGENT, ENV, env, None, cfg, np.random.default_rng(0))
    print(f"X3: agent={AGENT} env={ENV} params={params} episodes={episodes} "
          f"max_steps={cfg['envs'][ENV]['max_episode_steps']} seeds={SEEDS}")
    print(f"memory_bytes={agent.memory_bytes()} flops_per_update={agent.flops_per_update()}")
    if args.dry_run:
        return

    guard_out(OUT_DIR)
    guard_out(REPORT_DIR)
    targets = [os.path.join(OUT_DIR, f"{AGENT}_{ENV}_{s}.csv") for s in SEEDS]
    for p in targets:
        cc.assert_new_output(p)
    cc.assert_new_output(os.path.join(REPORT_DIR, "report.md"))
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    meta = {"protocol": "X3", "params": params, "episodes": episodes, "seeds": SEEDS,
            "reset_seed": "seed + episode", "rng": "numpy.random.default_rng(seed)",
            "provenance_start": provenance()}
    rows, t0 = [], time.time()
    for s, path in zip(SEEDS, targets):
        run(AGENT, ENV, s, episodes, cfg, out_dir=OUT_DIR)
        off = os.path.join(OFFICIAL_DIR, f"{AGENT}_{ENV}_{s}.csv")
        r = {"seed": s, **cc.compare_csv(path, off)}
        rows.append(r)
        print(f"[X3] seed {s}: {r['episodes_identical']}/{r['episodes_official']} "
              f"{'IDENTICO' if r['identical'] else 'DIVERGE'} ({time.time() - t0:.0f}s)", flush=True)
    meta["wall_clock_s"] = round(time.time() - t0, 1)
    meta["outputs_sha256"] = {os.path.basename(p): sha256(p) for p in targets}
    with open(os.path.join(OUT_DIR, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    report(rows, t0)
    n_id = sum(r["identical"] for r in rows)
    print(f"X3 finished: {n_id}/{len(rows)} seeds identical to raw_final (informative).")


if __name__ == "__main__":
    main()
