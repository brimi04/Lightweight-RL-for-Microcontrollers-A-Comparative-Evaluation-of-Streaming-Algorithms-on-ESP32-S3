import numpy as np
from agents.base_agent import BaseAgent
from core.running_normalizer import RunningNormalizer

# Gradient and weight clipping to avoid numerical divergence (Bug 7).
# Without it, with high learning rates (alpha_actor >= 0.05 in CartPole) the
# weights overflow: the TD bootstrap + an unbounded linear policy
# enter the same feedback loop as TinyDQN's "deadly triad", although
# here without ReLU (the instability comes from the softmax gradient itself
# multiplied by alpha when td_error is large). Same pattern as
# GRAD_CLIP/WEIGHT_CLIP in core/nn_manual.py.
GRAD_CLIP = 5.0
WEIGHT_CLIP = 5.0


class StreamACAgent(BaseAgent):
    """One-step online actor-critic (TD(0)), linear softmax policy,
    no experience replay. Weights: W_actor (n_actions x n_features), w_critic (n_features,)."""

    def __init__(self, n_features, n_actions, alpha_actor=0.05, alpha_critic=0.1,
                 gamma=0.99, entropy_coef=0.02, rng=None):
        self.n_features = n_features
        self.n_actions = n_actions
        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.rng = rng or np.random.default_rng()
        self.normalizer = RunningNormalizer(n_features)
        # +1 feature: constant bias (always 1), needed because the optimal
        # action can be almost constant regardless of the position.
        self.W_actor = np.zeros((n_actions, n_features + 1), dtype=np.float32)
        self.w_critic = np.zeros(n_features + 1, dtype=np.float32)

    def _augment(self, x):
        return np.concatenate([x, [1.0]]).astype(np.float32)

    def _policy(self, x):
        logits = self.W_actor @ x
        logits -= np.max(logits)
        exp = np.exp(logits)
        return exp / np.sum(exp)

    def act(self, obs):
        self.normalizer.update(obs)
        x = self._augment(self.normalizer.normalize(obs))
        probs = self._policy(x)
        action = int(self.rng.choice(self.n_actions, p=probs))
        return action

    def update(self, obs, action, reward, next_obs, terminated, episode_end):
        x = self._augment(self.normalizer.normalize(obs))
        x_next = self._augment(self.normalizer.normalize(next_obs))

        v = self.w_critic @ x
        v_next = 0.0 if terminated else self.w_critic @ x_next
        td_error = reward + self.gamma * v_next - v

        delta_critic = self.alpha_critic * td_error * x
        delta_critic = np.clip(delta_critic, -GRAD_CLIP, GRAD_CLIP)
        self.w_critic += delta_critic
        np.clip(self.w_critic, -WEIGHT_CLIP, WEIGHT_CLIP, out=self.w_critic)

        probs = self._policy(x)
        grad_log = -probs[:, None] * x[None, :]
        grad_log[action] += x

        # entropy bonus: pushes the logits to keep the policy
        # more spread out and prevent it from collapsing onto a single action too
        # early. dH/dz_j = -p_j * (log(p_j) + H), H = -sum p log p.
        log_probs = np.log(probs + 1e-8)
        entropy = -np.sum(probs * log_probs)
        grad_entropy_z = -probs * (log_probs + entropy)
        grad_entropy = grad_entropy_z[:, None] * x[None, :]

        delta_actor = self.alpha_actor * (td_error * grad_log + self.entropy_coef * grad_entropy)
        delta_actor = np.clip(delta_actor, -GRAD_CLIP, GRAD_CLIP)
        self.W_actor += delta_actor
        np.clip(self.W_actor, -WEIGHT_CLIP, WEIGHT_CLIP, out=self.W_actor)

    def n_params(self):
        return self.W_actor.size + self.w_critic.size

    def memory_bytes(self):
        return self.W_actor.nbytes + self.w_critic.nbytes + self.normalizer.nbytes()

    def flops_per_update(self):
        return 6 * self.W_actor.size + 4 * self.w_critic.size
