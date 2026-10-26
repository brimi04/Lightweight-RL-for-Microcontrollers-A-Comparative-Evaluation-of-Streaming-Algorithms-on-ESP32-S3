import numpy as np
from agents.base_agent import BaseAgent


class StreamQAgent(BaseAgent):
    """Tabular Q-learning with eligibility traces (TD(lambda)), immediate online
    update at every step, no replay buffer."""

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 lam=0.9, epsilon=0.1, epsilon_decay=0.999, rng=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.rng = rng or np.random.default_rng()
        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)
        self.traces = np.zeros((n_states, n_actions), dtype=np.float32)

    def act(self, state_idx):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.Q[state_idx]))

    def update(self, state_idx, action, reward, next_state_idx, terminated, episode_end):
        best_next = 0.0 if terminated else np.max(self.Q[next_state_idx])
        td_error = reward + self.gamma * best_next - self.Q[state_idx, action]

        self.traces *= self.gamma * self.lam
        self.traces[state_idx, action] = 1.0

        self.Q += self.alpha * td_error * self.traces

        if episode_end:
            self.traces[:] = 0.0
            self.epsilon *= self.epsilon_decay

    def n_params(self):
        return self.Q.size

    def memory_bytes(self):
        return self.Q.nbytes + self.traces.nbytes

    def flops_per_update(self):
        return 4 * self.Q.size
