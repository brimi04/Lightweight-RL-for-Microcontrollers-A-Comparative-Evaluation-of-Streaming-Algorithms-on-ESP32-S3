"""
Baseline-DQN runner with checkpointing, designed for short, repeated
calls (the remote shell kills any background process when the call
ends, so plain nohup cannot be used). Each invocation runs
for up to `--budget_s` seconds and then saves a checkpoint (pickle of the agent +
logger rows + next episode) to continue in the next call.

Usage:
  python run_resumable.py --mode cartpole    --seed 0 --episodes 1500 --budget_s 170
  python run_resumable.py --mode mountaincar --seed 0 --episodes 2000 --budget_s 170
  python run_resumable.py --mode gridworld_probe --seed 0 --budget_s 170

Prints "STATUS=PAUSED" if it must be called again, or "STATUS=DONE" when finished.
"""
import argparse, os, sys, time, pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gymnasium as gym

from envs.gridworld import GridWorldEnv
from agents.baseline_dqn import BaselineDQNAgent
from core.metrics import EpisodeLogger, Timer

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "raw_final")
CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "_checkpoints")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

BASELINE_PARAMS = dict(
    hidden=64, lr=0.001, gamma=0.99,
    epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
    buffer_size=10000, batch_size=32, target_update_freq=100,
)

GW_CONV_THRESHOLD = 1.5
GW_SUSTAIN = 50
GW_MAX_EPISODES = 5000
GW_MAX_STEPS = 100

MAX_STEPS = {"cartpole": 500, "mountaincar": 200}
ENV_NAME = {"cartpole": "cartpole", "mountaincar": "mountaincar", "gridworld_probe": "gridworld"}
SUFFIX = {"cartpole": "full", "mountaincar": "full", "gridworld_probe": "probe"}


def make_env(mode):
    if mode == "cartpole":
        return gym.make("CartPole-v1")
    if mode == "mountaincar":
        return gym.make("MountainCar-v0")
    if mode == "gridworld_probe":
        return GridWorldEnv(size=5)
    raise ValueError(mode)


def env_dims(mode, env):
    if mode == "gridworld_probe":
        return env.observation_size, env.n_actions
    return env.observation_space.shape[0], env.action_space.n


def ckpt_path(mode, seed):
    return os.path.join(CKPT_DIR, f"ckpt_{mode}_{seed}.pkl")


def out_path(mode, seed):
    return os.path.join(RESULTS_DIR, f"baseline_dqn_{ENV_NAME[mode]}_{seed}_{SUFFIX[mode]}.csv")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["cartpole", "mountaincar", "gridworld_probe"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--budget_s", type=float, default=170.0)
    args = p.parse_args()

    mode = args.mode
    seed = args.seed
    target_episodes = args.episodes or GW_MAX_EPISODES
    cpath = ckpt_path(mode, seed)
    opath = out_path(mode, seed)

    if os.path.exists(cpath):
        with open(cpath, "rb") as f:
            state = pickle.load(f)
        agent = state["agent"]
        rows = state["rows"]
        next_ep = state["next_ep"]
        sustain_count = state.get("sustain_count", 0)
        converged_at = state.get("converged_at", None)
        env = make_env(mode)
        print(f"RESUME mode={mode} seed={seed} from episode {next_ep} "
              f"(already has {len(rows)} rows)")
    else:
        env = make_env(mode)
        n_features, n_actions = env_dims(mode, env)
        rng = np.random.default_rng(seed)
        agent = BaselineDQNAgent(n_features, n_actions, rng=rng, **BASELINE_PARAMS)
        rows = []
        next_ep = 0
        sustain_count = 0
        converged_at = None
        print(f"START mode={mode} seed={seed} target_episodes={target_episodes}")

    max_steps = GW_MAX_STEPS if mode == "gridworld_probe" else MAX_STEPS[mode]
    t_start = time.time()
    episode = next_ep

    while episode < target_episodes:
        if time.time() - t_start > args.budget_s:
            break

        obs, _ = env.reset(seed=seed + episode)
        state_vec = np.asarray(obs, dtype=np.float32)
        total_reward, steps = 0.0, 0
        with Timer() as t:
            for _ in range(max_steps):
                action = agent.act(state_vec)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_state = np.asarray(next_obs, dtype=np.float32)
                done = terminated or truncated
                agent.update(state_vec, action, reward, next_state, terminated, done)
                state_vec = next_state
                total_reward += reward
                steps += 1
                if done:
                    break

        rows.append({
            "agent": "baseline_dqn", "env": ENV_NAME[mode], "seed": seed,
            "episode": episode, "reward": total_reward, "steps": steps,
            "memory_bytes": agent.memory_bytes(),
            "flops_per_update": agent.flops_per_update(),
            "wall_time_s": t.elapsed,
        })

        if mode == "gridworld_probe" and episode >= 19:
            tail = [r["reward"] for r in rows[-20:]]
            r20 = sum(tail) / len(tail)
            if r20 >= GW_CONV_THRESHOLD:
                sustain_count += 1
                if sustain_count >= GW_SUSTAIN and converged_at is None:
                    converged_at = episode - GW_SUSTAIN + 1
            else:
                sustain_count = 0

        episode += 1

        if converged_at is not None and mode == "gridworld_probe" and episode >= converged_at + GW_SUSTAIN + 150:
            print(f"  gridworld_probe seed={seed}: convergence confirmed at ep={converged_at}, stopping at ep={episode}")
            break

    tail = [r["reward"] for r in rows[-20:]] if rows else [0.0]
    r20_now = sum(tail) / len(tail)
    elapsed = time.time() - t_start

    done = (episode >= target_episodes) or (
        mode == "gridworld_probe" and converged_at is not None and episode >= converged_at + GW_SUSTAIN + 150
    )

    if done:
        logger = EpisodeLogger("baseline_dqn", ENV_NAME[mode], seed, opath)
        logger.rows = rows
        logger.save()
        if os.path.exists(cpath):
            try:
                os.remove(cpath)
            except OSError:
                pass  # the device sandbox may block the deletion; not critical
        extra = f" converged_at_episode={converged_at}" if mode == "gridworld_probe" else ""
        print(f"STATUS=DONE mode={mode} seed={seed} episodios_corridos={episode} "
              f"R20_final={r20_now:.2f}{extra} -> {opath}")
    else:
        with open(cpath, "wb") as f:
            pickle.dump({
                "agent": agent, "rows": rows, "next_ep": episode,
                "sustain_count": sustain_count, "converged_at": converged_at,
            }, f)
        print(f"STATUS=PAUSED mode={mode} seed={seed} episodio_actual={episode}/{target_episodes} "
              f"R20={r20_now:.2f} tiempo_bloque={elapsed:.0f}s converged_at={converged_at}")


if __name__ == "__main__":
    main()
