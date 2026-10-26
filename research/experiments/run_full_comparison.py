"""
Re-runs of the baseline DQN with the same episode budget as the lightweight
agents (fairness fix), plus a convergence probe in GridWorld to
estimate how many episodes the DQN would actually need to converge.

Outputs (they do not overwrite the official CSVs currently used in the paper):
  results/raw_final/baseline_dqn_cartpole_{seed}_full.csv     (1500 ep, seeds 0-4)
  results/raw_final/baseline_dqn_mountaincar_{seed}_full.csv  (2000 ep, seeds 0-4)
  results/raw_final/baseline_dqn_gridworld_{seed}_probe.csv   (up to 5000 ep, seeds 0-4)
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gymnasium as gym

from envs.gridworld import GridWorldEnv
from agents.baseline_dqn import BaselineDQNAgent
from core.metrics import EpisodeLogger, Timer

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "raw_final")
os.makedirs(RESULTS_DIR, exist_ok=True)

BASELINE_PARAMS = dict(
    hidden=64, lr=0.001, gamma=0.99,
    epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
    buffer_size=10000, batch_size=32, target_update_freq=100,
)

GW_CONV_THRESHOLD = 1.5   # convergence_reward from config.yaml for gridworld
GW_SUSTAIN = 50           # consecutive episodes with R20 >= threshold needed to declare convergence
GW_MAX_EPISODES = 5000

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def run_fixed(env_name, seed, episodes, max_steps, make_env, dims, suffix):
    rng = np.random.default_rng(seed)
    env = make_env()
    n_features, n_actions = dims(env)
    agent = BaselineDQNAgent(n_features, n_actions, rng=rng, **BASELINE_PARAMS)
    out_path = os.path.join(RESULTS_DIR, f"baseline_dqn_{env_name}_{seed}_{suffix}.csv")
    logger = EpisodeLogger("baseline_dqn", env_name, seed, out_path)

    t0 = time.time()
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        state = np.asarray(obs, dtype=np.float32)
        total_reward, steps = 0.0, 0
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
        logger.log_episode(episode=episode, reward=total_reward, steps=steps,
                            memory_bytes=agent.memory_bytes(),
                            flops_per_update=agent.flops_per_update(),
                            wall_time_s=t.elapsed)
        if (episode + 1) % 200 == 0:
            tail = [r["reward"] for r in logger.rows[-20:]]
            log(f"  {env_name} seed={seed} ep={episode+1}/{episodes} R20={sum(tail)/len(tail):.2f} "
                f"elapsed={time.time()-t0:.0f}s")
    logger.save()
    tail = [r["reward"] for r in logger.rows[-20:]]
    log(f"DONE {env_name} seed={seed} -> R20_final={sum(tail)/len(tail):.2f} "
        f"total_time={time.time()-t0:.0f}s -> {out_path}")


def run_gridworld_probe(seed):
    rng = np.random.default_rng(seed)
    env = GridWorldEnv(size=5)
    agent = BaselineDQNAgent(env.observation_size, env.n_actions, rng=rng, **BASELINE_PARAMS)
    out_path = os.path.join(RESULTS_DIR, f"baseline_dqn_gridworld_{seed}_probe.csv")
    logger = EpisodeLogger("baseline_dqn", "gridworld", seed, out_path)

    converged_at = None
    sustain_count = 0
    t0 = time.time()
    for episode in range(GW_MAX_EPISODES):
        obs, _ = env.reset(seed=seed + episode)
        state = np.asarray(obs, dtype=np.float32)
        total_reward, steps = 0.0, 0
        with Timer() as t:
            for _ in range(100):
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
        logger.log_episode(episode=episode, reward=total_reward, steps=steps,
                            memory_bytes=agent.memory_bytes(),
                            flops_per_update=agent.flops_per_update(),
                            wall_time_s=t.elapsed)

        if episode >= 19:
            tail = [r["reward"] for r in logger.rows[-20:]]
            r20 = sum(tail) / len(tail)
            if r20 >= GW_CONV_THRESHOLD:
                sustain_count += 1
                if sustain_count >= GW_SUSTAIN and converged_at is None:
                    converged_at = episode - GW_SUSTAIN + 1
            else:
                sustain_count = 0

        if (episode + 1) % 200 == 0:
            tail = [r["reward"] for r in logger.rows[-20:]]
            log(f"  gridworld-probe seed={seed} ep={episode+1}/{GW_MAX_EPISODES} "
                f"R20={sum(tail)/len(tail):.2f} converged_at={converged_at} "
                f"elapsed={time.time()-t0:.0f}s")

        if converged_at is not None and episode >= converged_at + 200:
            log(f"  gridworld-probe seed={seed}: sustained convergence confirmed, stopping at ep={episode+1}")
            break

    logger.save()
    log(f"DONE gridworld-probe seed={seed} -> converged_at_episode={converged_at} "
        f"(episodes run={episode+1}) total_time={time.time()-t0:.0f}s -> {out_path}")
    return converged_at


def main():
    log("=== PHASE 1: CartPole, full budget (1500 ep x 5 seeds) ===")
    for seed in range(5):
        run_fixed("cartpole", seed, 1500, 500,
                  lambda: gym.make("CartPole-v1"),
                  lambda e: (e.observation_space.shape[0], e.action_space.n),
                  "full")

    log("=== PHASE 2: MountainCar, full budget (2000 ep x 5 seeds) ===")
    for seed in range(5):
        run_fixed("mountaincar", seed, 2000, 200,
                  lambda: gym.make("MountainCar-v0"),
                  lambda e: (e.observation_space.shape[0], e.action_space.n),
                  "full")

    log("=== PHASE 3: GridWorld, convergence probe (up to 5000 ep x 5 seeds) ===")
    results = []
    for seed in range(5):
        results.append(run_gridworld_probe(seed))
    log(f"Convergence episodes per seed (GridWorld probe): {results}")

    log("=== ALL DONE ===")

if __name__ == "__main__":
    main()
