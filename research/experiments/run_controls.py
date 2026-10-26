"""
run_controls.py -- Reproducibility controls C0 and C2.

C0  Regression of run_single.py after K1/K2: re-executes the 80 official
    reproducible runs (GridWorld and CartPole x 3 agents; MountainCar x Stream-Q and TinyDQN;
    seeds 0-9; official budget) with run_single.run() WITHOUT params_override (uses
    config.yaml + overrides) and compares them with results/raw_final/ on reward, steps,
    memory_bytes and flops_per_update (exact equality). Stream-AC/MountainCar is
    excluded (the official CSVs are known not to be reproducible with the documented
    configuration). It stops at the FIRST discrepancy.
C2  Control for X2: Stream-AC/CartPole beta_e=0.05 through the params_override path that
    X2 will use (alpha_actor 0.005, alpha_critic 0.01), seeds 0-9, 1500 episodes.
    Requires a valid C1_PASSED.

The status strings written to results.json ("C0 SUPERADO ...", "C2 FALLIDO ...", etc.)
are kept verbatim in Spanish: they are machine-read status codes that are compared
literally with the stored evidence (see run_ablations_streamac_beta.py).

Outputs (only in results/derived/, never overwrites):
  controls/C0_regression/{runs/, report.md, results.json}    + controls/C0_PASSED.json
  ablations_cartpole_1500/stream_ac_beta/ref_b005/*.csv,
  controls/C2_streamac_beta_ref/{report.md, results.json}    + controls/C2_PASSED.json
"""
import argparse
import json
import os
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.run_single import run, resolve_params, CONFIG_PATH
from experiments import controls_common as cc

C0_JOBS = ([(e, a) for e in ("gridworld", "cartpole") for a in ("stream_q", "stream_ac", "tiny_dqn")]
           + [("mountaincar", "stream_q"), ("mountaincar", "tiny_dqn")])
SEEDS = list(range(10))
C2_EXPECTED = {"alpha_actor": 0.005, "alpha_critic": 0.01, "gamma": 0.99, "entropy_coef": 0.05}


def report(path_dir, title, criterion, rows, status, extra, t0):
    lines = [f"# {title}", "", f"UTC date: {cc.now()} · duration: {time.time() - t0:.0f} s", "",
             f"Criterion: {criterion}", "",
             "| environment | agent | seed | episodes (new/official) | identical | first divergence | columns | result |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['env']} | {r['agent']} | {r['seed']} | {r['episodes_new']}/{r['episodes_official']} | "
                     f"{r['episodes_identical']} | {r['first_divergent_episode'] if r['first_divergent_episode'] is not None else '—'} | "
                     f"{','.join(r['divergent_columns']) or '—'} | {'OK' if r['identical'] else 'DISCREPANCIA'} |")
    lines += ["", f"**RESULT: {status}**", "", "Output and official hashes: see `results.json`."]
    with open(os.path.join(path_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(path_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"status": status, "rows": rows, **extra}, f, indent=2)


def run_c0(cfg, dry):
    out = os.path.join(cc.CONTROLS, "C0_regression"); runs = os.path.join(out, "runs")
    jobs = [(e, a, s) for e, a in C0_JOBS for s in SEEDS]
    for e, a, s in jobs:
        off = os.path.join(cc.OFFICIAL, f"{a}_{e}_{s}.csv")
        assert os.path.exists(off), off
        cc.assert_new_output(os.path.join(runs, f"{a}_{e}_{s}.csv"))
    print(f"C0: {len(jobs)} runs; episodes per environment: "
          f"{ {e: cfg['envs'][e]['episodes'] for e in ('gridworld', 'cartpole', 'mountaincar')} }")
    for e, a in C0_JOBS:
        print(f"   {e:11s} {a:9s} params={resolve_params(cfg, a, e)}")
    if dry:
        return 0
    cc.assert_new_output(os.path.join(out, "report.md")); os.makedirs(runs)
    extra = {"command": os.environ.get("CONTROL_COMMAND", " ".join(sys.argv)), "argv": sys.argv,
             "start_utc": cc.now(), "environment": cc.environment(), "code_sha256": cc.code_hashes(),
             "params": {f"{e}/{a}": resolve_params(cfg, a, e) for e, a in C0_JOBS}}
    rows, t0 = [], time.time()
    for e, a, s in jobs:
        episodes = cfg["envs"][e]["episodes"]
        run(a, e, s, episodes, cfg, out_dir=runs)
        r = {"env": e, "agent": a, "seed": s, **cc.compare_csv(os.path.join(runs, f"{a}_{e}_{s}.csv"),
                                                               os.path.join(cc.OFFICIAL, f"{a}_{e}_{s}.csv"))}
        rows.append(r)
        print(f"[C0] {e:11s} {a:9s} seed {s}: {r['episodes_identical']}/{r['episodes_official']} "
              f"{'OK' if r['identical'] else 'DISCREPANCIA'} ({time.time() - t0:.0f}s)", flush=True)
        if not r["identical"]:
            report(out, "Control C0 — regression of run_single.py (K1/K2) against raw_final", CRIT, rows,
                   "C0 FALLIDO — secuencia detenida en la primera discrepancia", extra, t0)
            return 1
    report(out, "Control C0 — regression of run_single.py (K1/K2) against raw_final", CRIT, rows,
           "C0 SUPERADO (80/80 corridas idénticas)", extra, t0)
    cc.write_passed("C0", {"runs": len(rows), "all_identical": True})
    return 0


def run_c2(cfg, dry):
    params = dict(resolve_params(cfg, "stream_ac", "cartpole"), entropy_coef=0.05)
    assert params == C2_EXPECTED, params
    out_runs = os.path.join(cc.DERIVED, "ablations_cartpole_1500", "stream_ac_beta", "ref_b005")
    out = os.path.join(cc.CONTROLS, "C2_streamac_beta_ref")
    episodes = cfg["envs"]["cartpole"]["episodes"]; assert episodes == 1500
    for s in SEEDS:
        cc.assert_new_output(os.path.join(out_runs, f"stream_ac_cartpole_{s}.csv"))
    print(f"C2: Stream-AC/CartPole params_override={params} episodes={episodes} seeds={SEEDS}")
    if dry:
        return 0
    cc.require_passed("C1")
    cc.assert_new_output(os.path.join(out, "report.md")); os.makedirs(out); os.makedirs(out_runs)
    extra = {"command": os.environ.get("CONTROL_COMMAND", " ".join(sys.argv)), "argv": sys.argv,
             "start_utc": cc.now(), "environment": cc.environment(), "code_sha256": cc.code_hashes(),
             "params_override": params, "episodes": episodes}
    rows, t0 = [], time.time()
    for s in SEEDS:
        run("stream_ac", "cartpole", s, episodes, cfg, params_override=params, out_dir=out_runs)
        r = {"env": "cartpole", "agent": "stream_ac", "seed": s,
             **cc.compare_csv(os.path.join(out_runs, f"stream_ac_cartpole_{s}.csv"),
                              os.path.join(cc.OFFICIAL, f"stream_ac_cartpole_{s}.csv"))}
        rows.append(r)
        print(f"[C2] seed {s}: {r['episodes_identical']}/{r['episodes_official']} "
              f"{'OK' if r['identical'] else 'DISCREPANCIA'} ({time.time() - t0:.0f}s)", flush=True)
        if not r["identical"]:
            report(out, "Control C2 — Stream-AC/CartPole β_e = 0.05 frente a raw_final", CRIT, rows,
                   "C2 FALLIDO — secuencia detenida en la primera discrepancia", extra, t0)
            return 1
    report(out, "Control C2 — Stream-AC/CartPole β_e = 0.05 frente a raw_final", CRIT, rows,
           "C2 SUPERADO (10/10 seeds idénticas)", extra, t0)
    cc.write_passed("C2", {"runs": len(rows), "all_identical": True})
    return 0


CRIT = ("exact equality of reward, steps, memory_bytes and flops_per_update in every episode against "
        "results/raw_final/ (wall_time_s is ignored).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--control", choices=["C0", "C2"], required=True)
    ap.add_argument("--dry-run", action="store_true", help="only checks configuration and paths; does not run any episode")
    a = ap.parse_args()
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    sys.exit(run_c0(cfg, a.dry_run) if a.control == "C0" else run_c2(cfg, a.dry_run))


if __name__ == "__main__":
    main()
