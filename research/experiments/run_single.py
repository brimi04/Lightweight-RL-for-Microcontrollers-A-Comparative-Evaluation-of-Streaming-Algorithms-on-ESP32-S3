import argparse
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym

from envs.gridworld import GridWorldEnv
from envs.wrappers import CARTPOLE_DISCRETIZER, MOUNTAINCAR_DISCRETIZER
from agents.stream_q import StreamQAgent
from agents.stream_ac import StreamACAgent
from agents.tiny_dqn import TinyDQNAgent
from core.metrics import EpisodeLogger, Timer

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
# results/raw no longer exists (reorganized 2026-06-29): the runs with
# universal hyperparameters (this runner, and run_all.py, which uses it without
# passing out_dir) live in raw_final/, next to baseline_dqn and comparison_table.
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "raw_final")


def make_env(env_name):
    if env_name == "gridworld":
        return GridWorldEnv(size=5), None
    if env_name == "cartpole":
        return gym.make("CartPole-v1"), CARTPOLE_DISCRETIZER
    if env_name == "mountaincar":
        return gym.make("MountainCar-v0"), MOUNTAINCAR_DISCRETIZER
    raise ValueError(f"unknown environment: {env_name}")


def env_dims(env_name, env):
    if env_name == "gridworld":
        return env.observation_size, env.n_actions
    return env.observation_space.shape[0], env.action_space.n


def resolve_params(cfg, agent_name, env_name):
    """Universal hyperparameters from config.yaml (cfg["agents"][agent]),
    updated with the explicit per-environment overrides
    (cfg["overrides"][env][agent]) if they exist. Without an `overrides` block, or without
    an entry for (env, agent), it returns exactly cfg["agents"][agent]."""
    params = dict(cfg["agents"][agent_name])
    params.update(cfg.get("overrides", {}).get(env_name, {}).get(agent_name, {}))
    return params


def make_agent(agent_name, env_name, env, discretizer, cfg, rng, params_override=None):
    n_features, n_actions = env_dims(env_name, env)
    # params_override (if given) still takes full precedence, as before:
    # it is not combined with the per-environment overrides.
    if params_override is not None:
        params = params_override
    else:
        params = resolve_params(cfg, agent_name, env_name)

    if agent_name == "stream_q":
        if env_name == "gridworld":
            n_states = env.size * env.size
        else:
            n_states = discretizer.n_states
        return StreamQAgent(n_states, n_actions, rng=rng, **params)

    if agent_name == "stream_ac":
        return StreamACAgent(n_features, n_actions, rng=rng, **params)

    if agent_name == "tiny_dqn":
        return TinyDQNAgent(n_features, n_actions, rng=rng, **params)

    raise ValueError(f"unknown agent: {agent_name}")


def obs_for_agent(agent_name, env_name, obs, discretizer, env):
    if agent_name == "stream_q":
        if env_name == "gridworld":
            return env.state_index(obs)
        return discretizer.to_index(obs)
    return np.asarray(obs, dtype=np.float32)


def run(agent_name, env_name, seed, episodes, cfg, max_steps_override=None,
        params_override=None, out_dir=None, out_suffix=""):
    rng = np.random.default_rng(seed)
    env, discretizer = make_env(env_name)
    agent = make_agent(agent_name, env_name, env, discretizer, cfg, rng, params_override=params_override)
    max_steps = max_steps_override or cfg["envs"][env_name]["max_episode_steps"]

    out_dir = out_dir or RESULTS_DIR
    out_path = os.path.join(out_dir, f"{agent_name}_{env_name}_{seed}{out_suffix}.csv")
    logger = EpisodeLogger(agent_name, env_name, seed, out_path)

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        state = obs_for_agent(agent_name, env_name, obs, discretizer, env)
        total_reward = 0.0
        steps = 0

        with Timer() as t:
            for _ in range(max_steps):
                action = agent.act(state)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_state = obs_for_agent(agent_name, env_name, next_obs, discretizer, env)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, choices=["stream_q", "stream_ac", "tiny_dqn"])
    parser.add_argument("--env", required=True, choices=["gridworld", "cartpole", "mountaincar"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    episodes = args.episodes or cfg["envs"][args.env].get("episodes", cfg["episodes"])
    t0 = time.time()
    rows = run(args.agent, args.env, args.seed, episodes, cfg)
    elapsed = time.time() - t0

    first_avg = np.mean([r["reward"] for r in rows[:20]])
    last_avg = np.mean([r["reward"] for r in rows[-20:]])
    print(f"agent={args.agent} env={args.env} episodes={episodes} time={elapsed:.1f}s")
    print(f"mean reward first 20: {first_avg:.2f} | last 20: {last_avg:.2f}")


if __name__ == "__main__":
    main()
