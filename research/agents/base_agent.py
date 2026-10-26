class BaseAgent:
    """Common interface for online RL agents (sample-by-sample, no batches)."""

    def act(self, obs):
        raise NotImplementedError

    def update(self, obs, action, reward, next_obs, terminated, episode_end):
        """terminated: true end of the MDP (e.g. the pole falls) -> cuts the bootstrap.
        episode_end: terminated or truncated -> triggers the reset of traces/epsilon."""
        raise NotImplementedError

    def n_params(self):
        raise NotImplementedError

    def memory_bytes(self):
        """Model size (weights + tables + traces) in bytes, computed analytically."""
        raise NotImplementedError

    def flops_per_update(self):
        """Analytical count of multiplications+additions per call to update()."""
        raise NotImplementedError
