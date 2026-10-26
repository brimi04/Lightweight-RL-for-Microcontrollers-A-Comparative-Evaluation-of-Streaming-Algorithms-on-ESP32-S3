import numpy as np


class RunningNormalizer:
    """Streaming normalization (Welford), without storing the observation history."""

    def __init__(self, n_features):
        self.mean = np.zeros(n_features, dtype=np.float32)
        self.var = np.ones(n_features, dtype=np.float32)
        self.count = 1e-4

    def update(self, x):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.var += (delta * (x - self.mean) - self.var) / self.count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-6)

    def nbytes(self):
        return self.mean.nbytes + self.var.nbytes
