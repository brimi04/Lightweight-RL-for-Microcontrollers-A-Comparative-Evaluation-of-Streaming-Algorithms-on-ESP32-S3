import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_single import run

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

AGENTS = ["stream_q", "stream_ac", "tiny_dqn"]
ENVS = ["gridworld", "cartpole", "mountaincar"]


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    seeds = cfg["seeds"]

    total = len(AGENTS) * len(ENVS) * len(seeds)
    done = 0
    for agent_name in AGENTS:
        for env_name in ENVS:
            episodes = cfg["envs"][env_name].get("episodes", cfg["episodes"])
            for seed in seeds:
                run(agent_name, env_name, seed, episodes, cfg)
                done += 1
                print(f"[{done}/{total}] {agent_name} x {env_name} seed={seed} episodes={episodes} OK")


if __name__ == "__main__":
    main()
