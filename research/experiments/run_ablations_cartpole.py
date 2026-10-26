"""
run_ablations_cartpole.py  --  Protocol X1: discretization ablation of
Stream-Q in CartPole (1500 episodes, seeds 0-9). Standalone script:
it does NOT modify or replace anything in run_single.py; it imports its functions
make_agent / obs_for_agent / resolve_params and replicates line by line the loop of
run_single.run(), changing only the discretizer.

Stages (gated):
  --stage control : runs only the official [6,6,12,12] configuration (control C1)
                    and compares it with results/raw_final/stream_q_cartpole_{seed}.csv
                    (reward, steps, memory_bytes, flops_per_update; exact equality).
                    Requires a valid controls/C0_PASSED.json. Writes
                    controls/C1_streamq_bins_ref/{report.md,results.json} and, if it passes,
                    controls/C1_PASSED.json bound to the SHA-256 of the code and config.
  --stage grid    : runs the uniform configurations. Refuses to run if
                    C1_PASSED.json does not exist or if any code/config hash changed.
  --dry-run       : shows the resolved configuration and checks the analytical
                    memory/FLOPs, without running any episode.

The status string written to results.json ("C1 SUPERADO ..." / "C1 FALLIDO ...") is
kept verbatim in Spanish, consistently with the stored evidence.

Outputs (never in raw_*/ or ablations/; never overwrites):
  results/derived/ablations_cartpole_1500/stream_q_bins/<config>/stream_q_cartpole_{seed}.csv
  results/derived/ablations_cartpole_1500/stream_q_bins/<config>/run_meta.json
CSV schema identical to raw_final (core.metrics.EpisodeLogger).
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np
import yaml

RESEARCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RESEARCH)

import gymnasium as gym

from envs.wrappers import CARTPOLE_DISCRETIZER, Discretizer
from core.metrics import EpisodeLogger, Timer
from experiments.run_single import make_agent, obs_for_agent, resolve_params, CONFIG_PATH
from experiments import controls_common as cc

AGENT, ENV = "stream_q", "cartpole"
RESULTS = os.path.join(RESEARCH, "results")
OUT_ROOT = os.path.join(RESULTS, "derived", "ablations_cartpole_1500", "stream_q_bins")
CONTROL_DIR = os.path.join(RESULTS, "derived", "controls", "C1_streamq_bins_ref")
OFFICIAL_DIR = os.path.join(RESULTS, "raw_final")
SEEDS = list(range(10))

# Expected official parameters (config.yaml, no override for stream_q).
EXPECTED_PARAMS = {"alpha": 0.1, "gamma": 0.99, "lam": 0.9, "epsilon": 0.1, "epsilon_decay": 0.999}
EXPECTED_EPISODES, EXPECTED_MAX_STEPS = 1500, 500


def make_discretizer(config_id):
    """'ref' returns THE SAME official object used by run_single.make_env('cartpole');
    'uK' creates uniform bins [K,K,K,K] with the same official bounds."""
    if config_id == "ref":
        return CARTPOLE_DISCRETIZER
    k = int(config_id[1:])
    return Discretizer(low=CARTPOLE_DISCRETIZER.low.tolist(),
                       high=CARTPOLE_DISCRETIZER.high.tolist(), bins=[k] * 4)


CONFIGS = {"ref": [6, 6, 12, 12], "u6": [6] * 4, "u9": [9] * 4, "u12": [12] * 4, "u18": [18] * 4, "u24": [24] * 4}


def run_one(seed, episodes, cfg, discretizer, out_path):
    """Exact replica of run_single.run(agent_name='stream_q', env_name='cartpole', ...)
    with the discretizer as the only difference. Same creation order of rng,
    environment and agent; same reset seeds; same handling of terminated /
    truncated; same calls to act()/update()."""
    rng = np.random.default_rng(seed)
    env = gym.make("CartPole-v1")                       # = run_single.make_env("cartpole")[0]
    agent = make_agent(AGENT, ENV, env, discretizer, cfg, rng)
    max_steps = cfg["envs"][ENV]["max_episode_steps"]
    logger = EpisodeLogger(AGENT, ENV, seed, out_path)

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        state = obs_for_agent(AGENT, ENV, obs, discretizer, env)
        total_reward = 0.0
        steps = 0

        with Timer() as t:
            for _ in range(max_steps):
                action = agent.act(state)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_state = obs_for_agent(AGENT, ENV, next_obs, discretizer, env)
                done = terminated or truncated
                agent.update(state, action, reward, next_state, terminated, done)

                state = next_state
                total_reward += reward
                steps += 1
                if done:
                    break

        logger.log_episode(
            episode=episode,
            reward=total_reward,
            steps=steps,
            memory_bytes=agent.memory_bytes(),
            flops_per_update=agent.flops_per_update(),
            wall_time_s=t.elapsed,
        )

    logger.save()
    return logger.rows


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def provenance():
    return {"utc": cc.now(), "argv": sys.argv, "command": os.environ.get("CONTROL_COMMAND", " ".join(sys.argv)),
            "environment": cc.environment(), "code_sha256": cc.code_hashes()}


def check_config(cfg, episodes):
    params = resolve_params(cfg, AGENT, ENV)
    assert params == EXPECTED_PARAMS, f"stream_q parameters differ from the official ones: {params}"
    assert cfg["envs"][ENV]["max_episode_steps"] == EXPECTED_MAX_STEPS
    assert episodes == EXPECTED_EPISODES, f"protocol X1 requires {EXPECTED_EPISODES} episodes"
    assert list(CARTPOLE_DISCRETIZER.bins) == CONFIGS["ref"], "the official discretizer is not [6,6,12,12]"
    return params


def expected_cost(bins):
    n = int(np.prod(bins)) * 2                         # states x 2 actions
    return 2 * n * 4, 4 * n                            # Q + traces in float32; 4 FLOPs per entry


def guard_out(path):
    rp = os.path.realpath(path)
    for forbidden in ("raw_final", "raw_hardware", "raw_tuned", "raw_exploratory", "raw_summaries", "ablations" + os.sep):
        if os.sep + forbidden in rp + os.sep and "derived" not in rp:
            raise SystemExit(f"Forbidden output: {rp}")
    if "derived" not in rp:
        raise SystemExit(f"The output must be inside results/derived: {rp}")


def run_config(config_id, seeds, episodes, cfg, params):
    disc = make_discretizer(config_id)
    out_dir = os.path.join(OUT_ROOT, config_id)
    guard_out(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    targets = [os.path.join(out_dir, f"{AGENT}_{ENV}_{s}.csv") for s in seeds]
    existing = [p for p in targets if os.path.exists(p)]
    if existing:
        raise SystemExit(f"Outputs already exist and will not be overwritten: {existing[:3]}...")
    meta = {"protocol": "X1", "config_id": config_id, "bins": CONFIGS[config_id],
            "n_states": int(disc.n_states), "params": params, "episodes": episodes,
            "max_episode_steps": cfg["envs"][ENV]["max_episode_steps"], "seeds": seeds,
            "reset_seed": "seed + episode", "rng": "numpy.random.default_rng(seed)",
            "provenance_start": provenance()}
    t0 = time.time()
    for s, path in zip(seeds, targets):
        rows = run_one(s, episodes, cfg, disc, path)
        print(f"[{config_id}] seed {s}: {len(rows)} ep, R20={np.mean([r['reward'] for r in rows[-20:]]):.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    meta["wall_clock_s"] = round(time.time() - t0, 1)
    meta["outputs_sha256"] = {os.path.basename(p): sha256(p) for p in targets}
    with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def compare_control(seeds, t0):
    rows = []
    for s in seeds:
        r = {"env": ENV, "agent": AGENT, "seed": s,
             **cc.compare_csv(os.path.join(OUT_ROOT, "ref", f"{AGENT}_{ENV}_{s}.csv"),
                              os.path.join(OFFICIAL_DIR, f"{AGENT}_{ENV}_{s}.csv"))}
        rows.append(r)
        print(f"[C1] seed {s}: {r['episodes_identical']}/{r['episodes_official']} "
              f"{'OK' if r['identical'] else 'DISCREPANCIA'}", flush=True)
    ok = all(r["identical"] for r in rows)
    status = "C1 SUPERADO (10/10 seeds idénticas)" if ok else "C1 FALLIDO — detener X1"
    lines = ["# Control C1 — Stream-Q/CartPole [6,6,12,12] frente a raw_final", "",
             f"UTC date: {cc.now()} · duration: {time.time() - t0:.0f} s", "",
             "Criterion: exact equality of reward, steps, memory_bytes and flops_per_update in every episode "
             "(wall_time_s is ignored).", "",
             "| environment | agent | seed | episodes (new/official) | identical | first divergence | columns | result |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['env']} | {r['agent']} | {r['seed']} | {r['episodes_new']}/{r['episodes_official']} | "
                     f"{r['episodes_identical']} | {r['first_divergent_episode'] if r['first_divergent_episode'] is not None else '—'} | "
                     f"{','.join(r['divergent_columns']) or '—'} | {'OK' if r['identical'] else 'DISCREPANCIA'} |")
    lines += ["", f"**RESULT: {status}**", "", "Output and official hashes, environment and command: see `results.json`."]
    with open(os.path.join(CONTROL_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(CONTROL_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"status": status, "rows": rows, "params": resolve_params(yaml.safe_load(open(CONFIG_PATH)), AGENT, ENV),
                   "bins": CONFIGS["ref"], **provenance()}, f, indent=2)
    if ok:
        cc.write_passed("C1", {"runs": len(rows), "all_identical": True, "bins": CONFIGS["ref"]})
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["control", "grid"], required=True)
    ap.add_argument("--configs", nargs="+", default=None, help="default: ref (control) or u6 u9 u12 u18 u24 (grid)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    episodes = cfg["envs"][ENV]["episodes"]
    params = check_config(cfg, episodes)
    configs = args.configs or (["ref"] if args.stage == "control" else ["u6", "u9", "u12", "u18", "u24"])
    if args.stage == "control" and configs != ["ref"]:
        raise SystemExit("The control stage only runs 'ref'.")

    for c in configs:
        disc = make_discretizer(c)
        agent = make_agent(AGENT, ENV, gym.make("CartPole-v1"), disc, cfg, np.random.default_rng(0))
        exp_mem, exp_fl = expected_cost(CONFIGS[c])
        assert (agent.memory_bytes(), agent.flops_per_update()) == (exp_mem, exp_fl), c
        print(f"{c:4s} bins={CONFIGS[c]} n_states={disc.n_states} memory_bytes={agent.memory_bytes()} "
              f"flops_per_update={agent.flops_per_update()} params={params}")
    print(f"episodes={episodes} max_steps={cfg['envs'][ENV]['max_episode_steps']} seeds={SEEDS} reset=seed+episode")
    if args.dry_run:
        return

    if args.stage == "control":
        cc.require_passed("C0")
        cc.assert_new_output(os.path.join(CONTROL_DIR, "report.md"))
        os.makedirs(CONTROL_DIR)
        t0 = time.time()
        run_config("ref", SEEDS, episodes, cfg, params)
        sys.exit(0 if compare_control(SEEDS, t0) else 1)
    cc.require_passed("C1")
    for c in configs:
        run_config(c, SEEDS, episodes, cfg, params)


if __name__ == "__main__":
    main()
