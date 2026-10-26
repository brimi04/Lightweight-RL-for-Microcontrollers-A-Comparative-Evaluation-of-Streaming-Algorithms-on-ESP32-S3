import numpy as np


class Discretizer:
    """Discretizes a vector of continuous observations into bins, for tabular agents (Stream-Q)."""

    def __init__(self, low, high, bins):
        self.low = np.array(low, dtype=np.float32)
        self.high = np.array(high, dtype=np.float32)
        self.bins = np.array(bins, dtype=np.int64)

    def discretize(self, obs):
        obs = np.clip(obs, self.low, self.high)
        ratios = (obs - self.low) / (self.high - self.low + 1e-8)
        idx = (ratios * (self.bins - 1)).astype(np.int64)
        return idx

    def to_index(self, obs):
        idx = self.discretize(obs)
        flat = 0
        for i, b in zip(idx, self.bins):
            flat = flat * b + i
        return int(flat)

    @property
    def n_states(self):
        return int(np.prod(self.bins))


CARTPOLE_DISCRETIZER = Discretizer(
    low=[-2.4, -3.0, -0.21, -3.0],
    high=[2.4, 3.0, 0.21, 3.0],
    bins=[6, 6, 12, 12],
)

MOUNTAINCAR_DISCRETIZER = Discretizer(
    low=[-1.2, -0.07],
    high=[0.6, 0.07],
    bins=[18, 14],
)


def mountaincar_height(position):
    """MountainCar terrain height as a function of position (same formula
    that gymnasium uses internally for rendering). Useful as the
    potential Phi(s) for potential-based reward shaping (Ng, Harada &
    Russell, 1999): r' = r + gamma*Phi(s') - Phi(s). This kind of shaping does
    not change the optimal policy (it is an invariant transformation), unlike
    arbitrary shaping.

    Diagnostic (see the project notes MEMORIA_PROYECTO.md, section 11, not included
    in this repository): with the original reward (-1/step, 0 only when reaching
    the goal) a random policy NEVER reaches the goal in 200 episodes of 200 steps --
    the signal is too flat/late for stream_ac and tiny_dqn to learn anything
    (stream_q learns partially because its tabular Q starts at 0, an "optimistic"
    value that drives state-directed exploration). Applying this scaled potential
    (tested with a factor of ~150x on this function) to stream_q showed a real
    improvement: from ~-161 to ~-145 average reward (last 20 episodes, 3 seeds,
    2000 episodes), with >50% of the final episodes reaching the goal vs. ~0-3%
    without shaping. Stream-AC and TinyDQN still did not solve the environment
    even with this shaping (left as future work, see the notes) -- the most likely
    cause is not reward sparsity but an exploration/representation limitation
    of those agents in this continuous environment.
    """
    return np.sin(3.0 * position) * 0.45 + 0.55
