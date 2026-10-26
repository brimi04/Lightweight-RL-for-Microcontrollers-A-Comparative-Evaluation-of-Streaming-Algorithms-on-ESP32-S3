import numpy as np

# Bound on the gradient entering each layer, and bound on the weights themselves after
# the update. Without it, the semi-gradient bootstrap (TD + non-linear function,
# no target network) enters a feedback loop: W grows -> the activation
# h=relu(Wx) grows (unbounded, ReLU does not bound from above) -> grad_W of the
# next layer (outer(grad_out, h)) grows even if grad_out is bounded ->
# W grows further. This is the classic "deadly triad" instability (bootstrap +
# off-policy-like update + function approximation, see Sutton & Barto ch. 11).
# Bounding only the gradient is not enough because what diverges is the activation,
# not the gradient itself; that is why W is also bounded directly. Simple, cheap
# clips that are easy to port to C (a couple of comparisons per weight).
GRAD_CLIP = 5.0
WEIGHT_CLIP = 10.0


class DenseLayer:
    """Dense layer with manual forward/backward (no autograd), designed to be
    translated line by line into C (matrix-vector multiplication + SGD)."""

    def __init__(self, n_in, n_out, activation="relu", rng=None):
        rng = rng or np.random.default_rng()
        limit = np.sqrt(6.0 / (n_in + n_out))
        self.W = rng.uniform(-limit, limit, size=(n_out, n_in)).astype(np.float32)
        self.b = np.zeros(n_out, dtype=np.float32)
        self.activation = activation
        self._x = None
        self._z = None

    def forward(self, x):
        self._x = x
        z = self.W @ x + self.b
        self._z = z
        if self.activation == "relu":
            return np.maximum(z, 0.0)
        return z  # linear

    def forward_inference(self, x):
        """Forward pass without storing the cache. Use it when backward() will not be
        called next (e.g. to compute the TD target), so as not to overwrite the
        cache needed by the next backward()."""
        z = self.W @ x + self.b
        if self.activation == "relu":
            return np.maximum(z, 0.0)
        return z  # linear

    def backward(self, grad_out, lr):
        grad_out = np.clip(grad_out, -GRAD_CLIP, GRAD_CLIP)
        if self.activation == "relu":
            grad_out = grad_out * (self._z > 0)
        grad_W = np.outer(grad_out, self._x)
        grad_b = grad_out
        grad_x = self.W.T @ grad_out
        self.W -= lr * grad_W
        self.b -= lr * grad_b
        np.clip(self.W, -WEIGHT_CLIP, WEIGHT_CLIP, out=self.W)
        np.clip(self.b, -WEIGHT_CLIP, WEIGHT_CLIP, out=self.b)
        return grad_x

    def n_params(self):
        return self.W.size + self.b.size

    def flops_forward(self):
        return 2 * self.W.size

    def flops_backward(self):
        return 4 * self.W.size
