import numpy as np


class MultilayerPerceptron:
    """Neural network: 3 -> 5 -> 5 -> 2 -> 2."""

    def __init__(self, seed=42):
        rng = np.random.default_rng(seed)

        # Weights: (number of inputs, number of outputs)
        self.W1 = rng.normal(0, 0.5, (3, 5))
        self.W2 = rng.normal(0, 0.5, (5, 5))
        self.W3 = rng.normal(0, 0.5, (5, 2))
        self.W4 = rng.normal(0, 0.5, (2, 2))

        # One bias for each neuron in the corresponding layer.
        self.b1 = np.zeros(5)
        self.b2 = np.zeros(5)
        self.b3 = np.zeros(2)
        self.b4 = np.zeros(2)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    def forward(self, inputs, return_layers=False):
        x = np.asarray(inputs, dtype=float)

        # Accept one input with shape (3,) or a batch with shape (n, 3).
        if x.ndim not in (1, 2) or x.shape[-1] != 3:
            raise ValueError("Input must have shape (3,) or (n, 3).")

        # Hidden layer 1: 3 inputs -> 5 neurons
        z1 = x @ self.W1 + self.b1
        a1 = self.sigmoid(z1)

        # Hidden layer 2: 5 inputs -> 5 neurons
        z2 = a1 @ self.W2 + self.b2
        a2 = self.sigmoid(z2)

        # Hidden layer 3: 5 inputs -> 2 neurons
        z3 = a2 @ self.W3 + self.b3
        a3 = self.sigmoid(z3)

        # Output layer: 2 inputs -> 2 neurons
        z4 = a3 @ self.W4 + self.b4
        a4 = self.relu(z4)

        if return_layers:
            return a1, a2, a3, a4

        return a4