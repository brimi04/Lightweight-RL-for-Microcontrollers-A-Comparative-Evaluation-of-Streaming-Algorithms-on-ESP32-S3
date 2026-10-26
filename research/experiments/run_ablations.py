"""
run_ablations.py
=============================================================
Controlled ablation studies — one variable at a time.

Goal: justify the design decisions of each agent
by showing the sensitivity to the key hyperparameters.
Answers the reviewer's questions:
  Why λ=0.9 in Stream-Q?
  Why 18 bins in CartPole?
  Why buffer_size=32 in TinyDQN?
  Why entropy_coef=0.05 in Stream-AC?

Implemented ablations:
  1. Stream-Q  — sensitivity to λ (trace decay):  {0.0, 0.5, 0.7, 0.9}
  2. Stream-Q  — sensitivity to bins in CartPole: {6, 9, 12, 18, 24}
  3. TinyDQN   — sensitivity to buffer_size:      {0, 8, 16, 32, 64}
  4. Stream-AC — sensitivity to entropy_coef:     {0.0, 0.01, 0.05, 0.1, 0.3}

Protocol:
  - 5 seeds per configuration (seeds 0–4).
  - All other hyperparameters fixed (universal values from config.yaml).
  - Environments: GridWorld and CartPole for all ablations.
  - Metric: mean reward over the last 20 episodes + normalized AUC.
  - Save CSVs in results/ablations/ and a summary in results/ablations/summary.md

Usage:
  python experiments/run_ablations.py               # all ablations
  python experiments/run_ablations.py --study lambda # a single ablation
  python experiments/run_ablations.py --seeds 3     # 3 seeds (fast)
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym

from envs.gridworld import GridWorldEnv
from envs.wrappers import Discretizer, CARTPOLE_DISCRETIZER, MOUNTAINCAR_DISCRETIZER
from agents.stream_q import StreamQAgent
from agents.stream_ac import StreamACAgent
from agents.tiny_dqn import TinyDQNAgent

CONFIG_PATH  = os.path.join(os.path.dirname(__file__), "config.yaml")
RESULTS_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "ablations")
LAST_N       = 20    # final episodes used to compute the mean reward

# ── Ablation grids ──────────────────────────────────────────────────────
LAMBDA_GRID      = [0.0, 0.5, 0.7, 0.9]
BINS_GRID        = [6, 9, 12, 18, 24]   # bins per dimension in CartPole
BUFFER_GRID      = [0, 8, 16, 32, 64]
ENTROPY_GRID     = [0.0, 0.01, 0.05, 0.1, 0.3]

ENVS_ABLATION    = ["gridworld", "cartpole"]


# ─────────────────────────────────────────────────────────────────────────────
# Execution infrastructure
# ─────────────────────────────────────────────────────────────────────────────

def make_cartpole_disc(n_bins):
    """Creates a Discretizer for CartPole with n_bins uniform bins per dimension."""
    return Discretizer(
        low= [-2.4, -3.0, -0.21, -3.0],
        high=[ 2.4,  3.0,  0.21,  3.0],
        bins=[n_bins] * 4,
    )


def make_env_and_dims(env_name, n_bins=18):
    """Returns (env, discretizer, n_features, n_actions, n_states_for_q)."""
    if env_name == "gridworld":
        env = GridWorldEnv(size=5)
        n_features = env.observation_size
        n_actions  = env.n_actions
        n_states   = env.size * env.size
        disc       = None
    elif env_name == "cartpole":
        env  = gym.make("CartPole-v1")
        disc = make_cartpole_disc(n_bins)
        n_features = env.observation_space.shape[0]
        n_actions  = env.action_space.n
        n_states   = disc.n_states
    else:
        raise ValueError(env_name)
    return env, disc, n_features, n_actions, n_states


def obs_to_state(env_name, obs, env, disc, agent_type):
    """Converts an observation to the agent's format."""
    if agent_type == "stream_q":
        if env_name == "gridworld":
            return env.state_index(obs)
        return disc.to_index(obs)
    return np.asarray(obs, dtype=np.float32)


def run_episode(env_name, env, disc, agent, agent_type, max_steps, seed_ep):
    obs, _ = env.reset(seed=seed_ep)
    state  = obs_to_state(env_name, obs, env, disc, agent_type)
    total  = 0.0
    for _ in range(max_steps):
        action = agent.act(state)
        next_obs, reward, term, trunc, _ = env.step(action)
        done       = term or trunc
        next_state = obs_to_state(env_name, next_obs, env, disc, agent_type)
        agent.update(state, action, reward, next_state, term, done)
        state  = next_state
        total += reward
        if done:
            break
    return total


def run_config(agent_factory, agent_type, env_name, n_episodes, max_steps,
               n_seeds, n_bins=18):
    """
    Runs one configuration with n_seeds seeds.
    Returns: (tail_means, auc_means, memory_bytes, flops_per_update)
    """
    env, disc, _, _, _ = make_env_and_dims(env_name, n_bins)
    tail_list = []
    auc_list  = []
    mem_b     = None
    flops     = None

    for seed in range(n_seeds):
        rng    = np.random.default_rng(seed)
        agent  = agent_factory(rng, env_name, n_bins)
        if mem_b is None:
            mem_b = agent.memory_bytes()
            flops = agent.flops_per_update()

        rewards = []
        for ep in range(n_episodes):
            r = run_episode(env_name, env, disc, agent, agent_type,
                            max_steps, seed * 10000 + ep)
            rewards.append(r)

        tail_list.append(float(np.mean(rewards[-LAST_N:])))
        # Normalized AUC: area under the curve / (n_episodes * max_possible_reward)
        # It is normalized by episodes to be independent of the budget.
        auc_list.append(float(np.mean(rewards)))  # AUC = mean over all episodes

    return tail_list, auc_list, mem_b, flops


# ─────────────────────────────────────────────────────────────────────────────
# Ablation definitions
# ─────────────────────────────────────────────────────────────────────────────

def ablation_lambda(cfg, n_seeds, n_episodes):
    """Stream-Q: sensitivity to λ (trace decay)."""
    base = dict(cfg["agents"]["stream_q"])
    rows = []

    for env_name in ENVS_ABLATION:
        max_steps = cfg["envs"][env_name]["max_episode_steps"]
        _, _, _, n_actions, n_states = make_env_and_dims(env_name)

        for lam in LAMBDA_GRID:
            params = {**base, "lam": lam}

            def factory(rng, env_n=env_name, nb=18, _n_states=n_states,
                        _na=n_actions, _p=params):
                _, disc_tmp, _, _, ns = make_env_and_dims(env_n)
                return StreamQAgent(_n_states if env_n == "gridworld" else disc_tmp.n_states,
                                    _na, rng=rng, **_p)

            tail, auc, mem, fl = run_config(
                factory, "stream_q", env_name, n_episodes, max_steps, n_seeds
            )
            for s, (t, a) in enumerate(zip(tail, auc)):
                rows.append(dict(
                    study="lambda", agent="stream_q", env=env_name,
                    param_name="lam", param_value=lam, seed=s,
                    tail_mean=t, auc_mean=a,
                    memory_bytes=mem, flops_per_update=fl,
                ))
            print(f"  [λ={lam}] {env_name:10s} → last20={np.mean(tail):6.2f}±{np.std(tail):.2f} "
                  f"| AUC={np.mean(auc):6.2f} | mem={mem}B")
    return rows


def ablation_bins(cfg, n_seeds, n_episodes):
    """Stream-Q: sensitivity to the number of bins per dimension (CartPole)."""
    base = dict(cfg["agents"]["stream_q"])
    rows = []
    env_name  = "cartpole"
    max_steps = cfg["envs"][env_name]["max_episode_steps"]
    _, _, _, n_actions, _ = make_env_and_dims(env_name)

    for n_bins in BINS_GRID:
        disc = make_cartpole_disc(n_bins)
        n_states = disc.n_states
        mem_agent = StreamQAgent(n_states, n_actions, rng=np.random.default_rng(0), **base)
        mem_ref   = mem_agent.memory_bytes()
        fl_ref    = mem_agent.flops_per_update()

        def factory(rng, env_n=env_name, nb=n_bins, _ns=n_states, _na=n_actions, _p=base):
            return StreamQAgent(_ns, _na, rng=rng, **_p)

        tail, auc, mem, fl = run_config(
            factory, "stream_q", env_name, n_episodes, max_steps, n_seeds, n_bins=n_bins
        )
        for s, (t, a) in enumerate(zip(tail, auc)):
            rows.append(dict(
                study="bins", agent="stream_q", env=env_name,
                param_name="bins_per_dim", param_value=n_bins, seed=s,
                tail_mean=t, auc_mean=a,
                memory_bytes=mem_ref, flops_per_update=fl_ref,
            ))
        print(f"  [bins={n_bins}] n_states={n_states:7d} mem={mem_ref:7d}B "
              f"→ last20={np.mean(tail):6.2f}±{np.std(tail):.2f}")
    return rows


def ablation_buffer(cfg, n_seeds, n_episodes):
    """TinyDQN: sensitivity to buffer_size."""
    base = dict(cfg["agents"]["tiny_dqn"])
    rows = []

    for env_name in ENVS_ABLATION:
        max_steps = cfg["envs"][env_name]["max_episode_steps"]
        _, _, n_features, n_actions, _ = make_env_and_dims(env_name)

        for buf in BUFFER_GRID:
            params = {**base, "buffer_size": buf}

            def factory(rng, env_n=env_name, nb=None, _nf=n_features, _na=n_actions, _p=params):
                return TinyDQNAgent(_nf, _na, rng=rng, **_p)

            tail, auc, mem, fl = run_config(
                factory, "tiny_dqn", env_name, n_episodes, max_steps, n_seeds
            )
            for s, (t, a) in enumerate(zip(tail, auc)):
                rows.append(dict(
                    study="buffer", agent="tiny_dqn", env=env_name,
                    param_name="buffer_size", param_value=buf, seed=s,
                    tail_mean=t, auc_mean=a,
                    memory_bytes=mem, flops_per_update=fl,
                ))
            print(f"  [buf={buf:2d}] {env_name:10s} mem={mem:5d}B "
                  f"→ last20={np.mean(tail):6.2f}±{np.std(tail):.2f}")
    return rows


def ablation_entropy(cfg, n_seeds, n_episodes):
    """Stream-AC: sensitivity to entropy_coef."""
    base = dict(cfg["agents"]["stream_ac"])
    rows = []

    for env_name in ENVS_ABLATION:
        max_steps = cfg["envs"][env_name]["max_episode_steps"]
        _, _, n_features, n_actions, _ = make_env_and_dims(env_name)

        for ent in ENTROPY_GRID:
            params = {**base, "entropy_coef": ent}

            def factory(rng, env_n=env_name, nb=None, _nf=n_features, _na=n_actions, _p=params):
                return StreamACAgent(_nf, _na, rng=rng, **_p)

            tail, auc, mem, fl = run_config(
                factory, "stream_ac", env_name, n_episodes, max_steps, n_seeds
            )
            for s, (t, a) in enumerate(zip(tail, auc)):
                rows.append(dict(
                    study="entropy", agent="stream_ac", env=env_name,
                    param_name="entropy_coef", param_value=ent, seed=s,
                    tail_mean=t, auc_mean=a,
                    memory_bytes=mem, flops_per_update=fl,
                ))
            print(f"  [ent={ent}] {env_name:10s} "
                  f"→ last20={np.mean(tail):6.2f}±{np.std(tail):.2f}")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report
# ─────────────────────────────────────────────────────────────────────────────

def write_summary(all_rows, n_seeds):
    """Generates a Markdown report with the results of all ablations."""
    path = os.path.join(RESULTS_DIR, "ablation_summary.md")

    import pandas as pd
    df = pd.DataFrame(all_rows)

    lines = ["# Ablation Study — Results\n"]
    lines.append(f"Seeds: {n_seeds} | Metric: reward over the last {LAST_N} episodes\n")

    for study in df["study"].unique():
        sub = df[df["study"] == study]
        param = sub["param_name"].iloc[0]
        agent = sub["agent"].iloc[0]
        lines.append(f"\n## {study.upper()} — {agent} (parameter: `{param}`)\n")

        for env_name in sub["env"].unique():
            sub_env = sub[sub["env"] == env_name]
            lines.append(f"\n### Environment: {env_name}\n")
            lines.append(f"| `{param}` | memory (B) | FLOPs/upd | R̄ (last20) | σ | mean AUC |")
            lines.append("|" + "|".join(["---"]*6) + "|")
            for val in sub_env["param_value"].unique():
                g = sub_env[sub_env["param_value"] == val]
                mem  = int(g["memory_bytes"].iloc[0])
                fl   = int(g["flops_per_update"].iloc[0])
                tail = g["tail_mean"].mean()
                std  = g["tail_mean"].std()
                auc  = g["auc_mean"].mean()
                lines.append(f"| {val} | {mem} | {fl} | {tail:.2f} | {std:.2f} | {auc:.2f} |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSummary saved to: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

STUDIES = {
    "lambda":  ablation_lambda,
    "bins":    ablation_bins,
    "buffer":  ablation_buffer,
    "entropy": ablation_entropy,
}


def main():
    parser = argparse.ArgumentParser(
        description="Controlled ablations: one variable at a time."
    )
    parser.add_argument("--study", nargs="+", default=list(STUDIES.keys()),
                        choices=list(STUDIES.keys()),
                        help="Ablations to run (default: all)")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Seeds per configuration (default: 5)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Episodes per run (default: the per-environment value in config.yaml)")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # The number of episodes per environment is taken from config.yaml if not
    # given on the CLI, but it is overridden with a reduced value
    # for the ablations (we want fast results with a clear trend).
    if args.episodes:
        for env_name in cfg["envs"]:
            cfg["envs"][env_name]["episodes"] = args.episodes

    all_rows = []
    t_total  = time.time()

    for study_name in args.study:
        print(f"\n{'='*60}")
        print(f"ABLATION: {study_name.upper()}")
        print(f"{'='*60}")

        n_ep_gw = args.episodes or cfg["envs"]["gridworld"]["episodes"]
        n_ep_cp = args.episodes or cfg["envs"]["cartpole"]["episodes"]

        # Each ablation uses the environment's number of episodes from config.yaml
        # (reduced for speed in ablations vs. the main experiment).
        n_ep = min(n_ep_gw, n_ep_cp) if not args.episodes else args.episodes
        rows = STUDIES[study_name](cfg, args.seeds, n_ep)
        all_rows.extend(rows)

        # CSV per ablation
        csv_path = os.path.join(RESULTS_DIR, f"ablation_{study_name}.csv")
        fields = ["study","agent","env","param_name","param_value","seed",
                  "tail_mean","auc_mean","memory_bytes","flops_per_update"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"CSV saved: {csv_path}")

    if all_rows:
        write_summary(all_rows, args.seeds)

    elapsed = time.time() - t_total
    print(f"\nTiempo total: {elapsed:.1f}s")
    print(f"Results in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
