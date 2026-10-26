import time
import csv
import os


class EpisodeLogger:
    """Accumulates per-episode metrics and dumps them to CSV. Schema shared
    with tools/serial_logger.py so that PC and ESP32 runs can be combined."""

    FIELDS = ["agent", "env", "seed", "episode", "reward", "steps",
              "memory_bytes", "flops_per_update", "wall_time_s"]

    def __init__(self, agent_name, env_name, seed, out_path):
        self.agent_name = agent_name
        self.env_name = env_name
        self.seed = seed
        self.out_path = out_path
        self.rows = []

    def log_episode(self, episode, reward, steps, memory_bytes, flops_per_update, wall_time_s):
        self.rows.append({
            "agent": self.agent_name,
            "env": self.env_name,
            "seed": self.seed,
            "episode": episode,
            "reward": reward,
            "steps": steps,
            "memory_bytes": memory_bytes,
            "flops_per_update": flops_per_update,
            "wall_time_s": wall_time_s,
        })

    def save(self):
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        with open(self.out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)


class Timer:
    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._t0
