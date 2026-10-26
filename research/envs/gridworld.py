import numpy as np

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right


class GridWorldEnv:
    """NxN grid, state = linear position, gymnasium-style API."""

    def __init__(self, size=5):
        self.size = size
        self.goal = (size - 1, size - 1)
        self.observation_size = 2
        self.n_actions = len(ACTIONS)
        self.pos = (0, 0)

    def reset(self, seed=None):
        self.pos = (0, 0)
        return self._obs(), {}

    def step(self, action):
        dr, dc = ACTIONS[action]
        r = min(max(self.pos[0] + dr, 0), self.size - 1)
        c = min(max(self.pos[1] + dc, 0), self.size - 1)
        self.pos = (r, c)
        terminated = self.pos == self.goal
        reward = 10.0 if terminated else -1.0
        truncated = False
        return self._obs(), reward, terminated, truncated, {}

    def _obs(self):
        return np.array([self.pos[0], self.pos[1]], dtype=np.float32)

    def state_index(self, obs):
        return int(obs[0]) * self.size + int(obs[1])
