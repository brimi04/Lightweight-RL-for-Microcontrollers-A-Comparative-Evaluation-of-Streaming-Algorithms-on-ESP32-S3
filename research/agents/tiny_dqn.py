import numpy as np
from collections import deque
from agents.base_agent import BaseAgent
from core.nn_manual import DenseLayer
from core.running_normalizer import RunningNormalizer


class TinyDQNAgent(BaseAgent):
    """Network with small hidden layers (manual forward/backward), discounted TD
    target, optional reduced mini-buffer to provide a fair contrast against
    purely streaming methods."""

    def __init__(self, n_features, n_actions, hidden=12, lr=0.01, gamma=0.99,
                 epsilon=0.1, epsilon_decay=0.999, buffer_size=32, rng=None):
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.rng = rng or np.random.default_rng()
        self.normalizer = RunningNormalizer(n_features)

        self.l1 = DenseLayer(n_features, hidden, activation="relu", rng=self.rng)
        self.l2 = DenseLayer(hidden, n_actions, activation="linear", rng=self.rng)

        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size) if buffer_size > 0 else None

    def _q_values(self, x):
        """Forward WITH cache (l1._x, l2._x are stored for backward()).
        Use only right before calling backward()."""
        h = self.l1.forward(x)
        return self.l2.forward(h)

    def _q_values_inference(self, x):
        """Forward WITHOUT cache, for uses not followed by backward()
        (act() and the TD-target computation on next_obs)."""
        h = self.l1.forward_inference(x)
        return self.l2.forward_inference(h)

    def act(self, obs):
        self.normalizer.update(obs)
        x = self.normalizer.normalize(obs)
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        q = self._q_values_inference(x)
        return int(np.argmax(q))

    def _train_step(self, obs, action, reward, next_obs, terminated):
        x = self.normalizer.normalize(obs)
        x_next = self.normalizer.normalize(next_obs)

        # q_next is computed first and without cache: it is only used for the target,
        # no backward is done through it. If it were computed with _q_values()
        # it would overwrite the l1/l2 cache before the backward below.
        q_next = self._q_values_inference(x_next)
        target = reward + (0.0 if terminated else self.gamma * np.max(q_next))

        q = self._q_values(x)  # caches x and h(x) for the following backward
        td_error = np.clip(q[action] - target, -10.0, 10.0)
        grad_out = np.zeros(self.n_actions, dtype=np.float32)
        grad_out[action] = td_error

        grad_h = self.l2.backward(grad_out, self.lr)
        self.l1.backward(grad_h, self.lr)

    def update(self, obs, action, reward, next_obs, terminated, episode_end):
        self._train_step(obs, action, reward, next_obs, terminated)

        if self.buffer is not None:
            self.buffer.append((obs, action, reward, next_obs, terminated))
            if len(self.buffer) >= 4:
                idx = self.rng.integers(len(self.buffer))
                self._train_step(*self.buffer[idx])

        if episode_end:
            self.epsilon *= self.epsilon_decay

    def n_params(self):
        return self.l1.n_params() + self.l2.n_params()

    def memory_bytes(self):
        weight_bytes = (self.l1.n_params() + self.l2.n_params()) * 4
        buffer_bytes = 0
        if self.buffer is not None:
            buffer_bytes = self.buffer_size * (self.n_features * 2 + 3) * 4
        return weight_bytes + buffer_bytes + self.normalizer.nbytes()

    def flops_per_update(self):
        forward = self.l1.flops_forward() + self.l2.flops_forward()
        backward = self.l1.flops_backward() + self.l2.flops_backward()
        train_step_flops = 2 * forward + backward
        if self.buffer is not None:
            # Once learning is active, update() performs this _train_step twice per
            # environment step: once for the current transition (always) and once
            # for a transition sampled from the mini-buffer (once len(buffer) >= 4,
            # which holds for virtually the entire run). flops_per_update() reports
            # this steady-state cost, consistent with memory_bytes()/flops_per_update()
            # representing the update procedure once learning is active.
            return 2 * train_step_flops
        return train_step_flops
