"""Self-contained runner for BaselineDQNAgent in the 3 environments.

Usage:
    python run_baseline.py --env gridworld --seed 0
    python run_baseline.py --env cartpole  --seed 0
    python run_baseline.py --env mountaincar --seed 0
    python run_baseline.py --all   # runs everything (slow)

Writes CSVs to results/raw_final/ with the same schema as run_single.py,
so that make_comparison_table.py picks them up automatically.

Fixed parameters (not in config.yaml because baseline_dqn is not a lightweight
agent -- documented here for reproducibility):
    hidden = 64              # 2 hidden layers of 64 units each
    lr = 0.001
    gamma = 0.99
    epsilon = 1.0            # maximum initial exploration
    epsilon_decay = 0.995    # per episode
    epsilon_min = 0.01
    buffer_size = 10000
    batch_size = 32
    target_update_freq = 100 # environment steps

Episode budget (same as the other agents):
    gridworld:   300
    cartpole:   1500
    mountaincar: 2000  -- but cut to 1000 because of time constraints;
                          in any case baseline_dqn stays at -200, like
                          the others in MountainCar without reward shaping.
"""
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
from agents.baseline_dqn import BaselineDQNAgent
from core.metrics import EpisodeLogger, Timer

# results/raw no longer exists (reorganized 2026-06-29); baseline_dqn lives in
# raw_final/ next to the lightweight agents and comparison_table.csv.
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "raw_final")

# Episode budget per environment
EPISODES = {
    "gridworld":   300,
    "cartpole":   1500,
    "mountaincar": 1000,   # reduced for time; still enough to see -200
}

MAX_STEPS = {
    "gridworld":   100,
    "cartpole":    500,
    "mountaincar": 200,
}

# Fixed baseline hyperparameters
BASELINE_PARAMS = dict(
    hidden=64,
    lr=0.001,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01,
    buffer_size=10000,
    batch_size=32,
    target_update_freq=100,
)


def make_env(env_name):
    if env_name == "gridworld":
        return GridWorldEnv(size=5)
    if env_name == "cartpole":
        return gym.make("CartPole-v1")
    if env_name == "mountaincar":
        return gym.make("MountainCar-v0")
    raise ValueError(env_name)


def env_dims(env_name, env):
    if env_name == "gridworld":
        return env.observation_size, env.n_actions
    return env.observation_space.shape[0], env.action_space.n


def run_one(env_name, seed):
    rng = np.random.default_rng(seed)
    env = make_env(env_name)
    n_features, n_actions = env_dims(env_name, env)
    agent = BaselineDQNAgent(n_features, n_actions, rng=rng, **BASELINE_PARAMS)

    episodes = EPISODES[env_name]
    max_steps = MAX_STEPS[env_name]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"baseline_dqn_{env_name}_{seed}.csv")
    logger = EpisodeLogger("baseline_dqn", env_name, seed, out_path)

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        state = np.asarray(obs, dtype=np.float32)
        total_reward = 0.0
        steps = 0

        with Timer() as t:
            for _ in range(max_steps):
                action = agent.act(state)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_state = np.asarray(next_obs, dtype=np.float32)
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
    tail = [r["reward"] for r in logger.rows[-20:]]
    return sum(tail) / len(tail)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["gridworld", "cartpole", "mountaincar"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        for env_name in ["gridworld", "cartpole", "mountaincar"]:
            for seed in [0, 1, 2]:
                t0 = time.time()
                last20 = run_one(env_name, seed)
                print(f"baseline_dqn x {env_name} seed={seed} "
                      f"last20={last20:.2f}  ({time.time()-t0:.1f}s)")
    else:
        if args.env is None:
            parser.error("--env is required unless --all is used")
        t0 = time.time()
        last20 = run_one(args.env, args.seed)
        print(f"baseline_dqn x {args.env} seed={args.seed} "
              f"last20={last20:.2f}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
