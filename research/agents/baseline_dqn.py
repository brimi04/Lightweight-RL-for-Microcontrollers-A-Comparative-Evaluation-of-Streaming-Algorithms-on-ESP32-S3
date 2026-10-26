import numpy as np
from collections import deque
from agents.base_agent import BaseAgent
from core.nn_manual import DenseLayer
from core.running_normalizer import RunningNormalizer


class BaselineDQNAgent(BaseAgent):
    """DQN-style reference (replay + target network as in Mnih et al. 2015, without their full protocol) with a 64-64 network, a large buffer (10 K) and a
    target network. It serves as a 'heavy' baseline to quantify the resource
    difference (memory, FLOPs) with respect to the lightweight agents.

    It is NOT intended for deployment on the ESP32: the goal is to have a
    quantitative comparison point that justifies the project's lightweight methods.

    Key differences vs TinyDQN:
      - Architecture: n_in -> 64 -> 64 -> n_actions  (vs n_in -> 12 -> n_actions)
      - Buffer:       10 000 transitions              (vs 32)
      - Target net:   copy of the weights updated every target_update_freq steps
      - Updates:      batch_size=32 sequential SGD updates per environment step, each
                      on ONE transition sampled uniformly with replacement
                      (not a minibatch)

    memory_bytes() and flops_per_update() are exact analytical counts,
    consistent with the methodology used in the rest of the project.
    """

    def __init__(self, n_features, n_actions, hidden=64, lr=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 buffer_size=10000, batch_size=32, target_update_freq=100,
                 rng=None):
        self.n_features = n_features
        self.n_actions = n_actions
        self.hidden = hidden
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.rng = rng or np.random.default_rng()
        self.normalizer = RunningNormalizer(n_features)

        # ── Online network (3 layers: n_in -> hidden -> hidden -> n_actions) ──────
        self.l1 = DenseLayer(n_features, hidden, activation="relu", rng=self.rng)
        self.l2 = DenseLayer(hidden, hidden, activation="relu", rng=self.rng)
        self.l3 = DenseLayer(hidden, n_actions, activation="linear", rng=self.rng)

        # ── Target network (copy of the weights, no gradient cache) ──────────────
        self._copy_to_target()

        # ── Replay buffer ─────────────────────────────────────────────────────
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self._steps = 0  # total environment steps (not episodes)

    # ── Target network ────────────────────────────────────────────────────────

    def _copy_to_target(self):
        """Copy the online-network weights into the target network."""
        self._t1_W = self.l1.W.copy()
        self._t1_b = self.l1.b.copy()
        self._t2_W = self.l2.W.copy()
        self._t2_b = self.l2.b.copy()
        self._t3_W = self.l3.W.copy()
        self._t3_b = self.l3.b.copy()

    def _q_target(self, x):
        """Forward pass of the target network (copied weights, no backward)."""
        z1 = self._t1_W @ x + self._t1_b
        h1 = np.maximum(z1, 0.0)
        z2 = self._t2_W @ h1 + self._t2_b
        h2 = np.maximum(z2, 0.0)
        return self._t3_W @ h2 + self._t3_b

    # ── Online network ────────────────────────────────────────────────────────

    def _q_online(self, x):
        """Forward pass of the online network WITH cache (for backward)."""
        h1 = self.l1.forward(x)
        h2 = self.l2.forward(h1)
        return self.l3.forward(h2)

    def _q_online_inference(self, x):
        """Forward pass of the online network WITHOUT cache (act and target computation)."""
        h1 = self.l1.forward_inference(x)
        h2 = self.l2.forward_inference(h1)
        return self.l3.forward_inference(h2)

    # ── Public interface ──────────────────────────────────────────────────────

    def act(self, obs):
        self.normalizer.update(obs)
        x = self.normalizer.normalize(obs)
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self._q_online_inference(x)))

    def _train_step(self, obs, action, reward, next_obs, terminated):
        """One SGD step on a single transition."""
        x = self.normalizer.normalize(obs)
        x_next = self.normalizer.normalize(next_obs)

        # TD target computed with the target network (not the online one) for stability
        q_next = self._q_target(x_next)
        target = reward + (0.0 if terminated else self.gamma * float(np.max(q_next)))

        # Forward with cache + backward
        q = self._q_online(x)
        td_error = float(np.clip(q[action] - target, -10.0, 10.0))
        grad_out = np.zeros(self.n_actions, dtype=np.float32)
        grad_out[action] = td_error

        grad_h2 = self.l3.backward(grad_out, self.lr)
        grad_h1 = self.l2.backward(grad_h2, self.lr)
        self.l1.backward(grad_h1, self.lr)

    def update(self, obs, action, reward, next_obs, terminated, episode_end):
        self.buffer.append((obs, action, reward, next_obs, terminated))
        self._steps += 1

        # Train only once the buffer holds enough transitions
        if len(self.buffer) >= self.batch_size:
            idxs = self.rng.integers(len(self.buffer), size=self.batch_size)
            for idx in idxs:
                self._train_step(*self.buffer[idx])

        # Periodically synchronize the target network
        if self._steps % self.target_update_freq == 0:
            self._copy_to_target()

        if episode_end:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── Analytical metrics ────────────────────────────────────────────────────

    def n_params(self):
        return self.l1.n_params() + self.l2.n_params() + self.l3.n_params()

    def memory_bytes(self):
        """Exact analytical count (bytes, float32):
          - Online network:  (l1 + l2 + l3) parameters * 4
          - Target network:  same (full copy of the weights)
          - Replay buffer: buffer_size * (n_features*2 + 3) * 4
              (obs, next_obs: n_features floats each; action, reward, terminated: 1 float each)
          - Normalizer: mean + var (n_features floats each)
        """
        params = self.l1.n_params() + self.l2.n_params() + self.l3.n_params()
        online_bytes = params * 4
        target_bytes = params * 4            # identical copy of the online network
        buffer_bytes = self.buffer_size * (self.n_features * 2 + 3) * 4
        return online_bytes + target_bytes + buffer_bytes + self.normalizer.nbytes()

    def flops_per_update(self):
        """Analytical FLOP count per call to update():
          For each of the batch_size samples:
            - Target forward (for q_next): flops_forward x3 layers
            - Online forward with cache + backward:         flops_forward + flops_backward x3
          Total = batch_size * (2 * forward + backward) per sample

        Note: DenseLayer.flops_forward() = 2*W.size (mults+adds of the matmul)
              DenseLayer.flops_backward() = 4*W.size (grad_W, grad_b, grad_x, update)
        """
        fwd = self.l1.flops_forward() + self.l2.flops_forward() + self.l3.flops_forward()
        bwd = self.l1.flops_backward() + self.l2.flops_backward() + self.l3.flops_backward()
        # 2026-10-19 correction: _train_step() runs 2 forwards (target on x_next
        # + online with cache on x) + 1 backward. The previous count used
        # 3 forwards (a 25% over-count); the raw CSV files in raw_final/ generated
        # before this date keep that old value in flops_per_update
        # (1433600 GW/CP, 1413120 MC). Correct values: 1146880 / 1130496.
        return self.batch_size * (2 * fwd + bwd)
