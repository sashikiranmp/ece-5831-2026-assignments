import numpy as np


class LogicGate:
    """Implement two-input logic gates using NumPy."""

    def and_gate(self, x1, x2):
        """Return 1 only when both inputs are 1."""
        inputs = np.array([x1, x2])
        weights = np.array([0.5, 0.5])
        bias = -0.7

        result = np.sum(inputs * weights) + bias
        return int(result > 0)

    def nand_gate(self, x1, x2):
        """Return 0 only when both inputs are 1."""
        inputs = np.array([x1, x2])
        weights = np.array([-0.5, -0.5])
        bias = 0.7

        result = np.sum(inputs * weights) + bias
        return int(result > 0)

    def or_gate(self, x1, x2):
        """Return 1 when at least one input is 1."""
        inputs = np.array([x1, x2])
        weights = np.array([0.5, 0.5])
        bias = -0.2

        result = np.sum(inputs * weights) + bias
        return int(result > 0)

    def nor_gate(self, x1, x2):
        """Return 1 only when both inputs are 0."""
        inputs = np.array([x1, x2])
        weights = np.array([-0.5, -0.5])
        bias = 0.2

        result = np.sum(inputs * weights) + bias
        return int(result > 0)

    def xor_gate(self, x1, x2):
        """Return 1 when the inputs are different."""
        nand_result = self.nand_gate(x1, x2)
        or_result = self.or_gate(x1, x2)

        return self.and_gate(nand_result, or_result)


if __name__ == "__main__":
    gate = LogicGate()

    input_pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]

    print("x1 x2 | AND NAND OR NOR XOR")

    for x1, x2 in input_pairs:
        print(
            f"{x1}  {x2}  | "
            f"{gate.and_gate(x1, x2)}   "
            f"{gate.nand_gate(x1, x2)}    "
            f"{gate.or_gate(x1, x2)}  "
            f"{gate.nor_gate(x1, x2)}   "
            f"{gate.xor_gate(x1, x2)}"
        )