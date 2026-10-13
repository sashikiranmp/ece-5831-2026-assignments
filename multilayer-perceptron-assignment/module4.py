import numpy as np
from multilayer_perceptron import MultilayerPerceptron


def main():
    np.set_printoptions(precision=6, suppress=True)

    model = MultilayerPerceptron(seed=42)

    test_inputs = np.array([
        [1.0, 0.5, -1.0],
        [0.0, 0.0, 0.0],
        [2.0, -0.5, 1.0]
    ])

    print("Architecture: 3 -> 5 -> 5 -> 2 -> 2")

    for number, inputs in enumerate(test_inputs, start=1):
        layers = model.forward(inputs, return_layers=True)

        print(f"\nExample {number}")
        print("Input:", inputs)

        for layer_number, values in enumerate(layers, start=1):
            activation = "Sigmoid" if layer_number < 4 else "ReLU"
            print(f"Layer {layer_number} ({activation}): {values}")

        # Check the required number of neurons in each layer.
        assert [layer.shape for layer in layers] == [
            (5,), (5,), (2,), (2,)
        ]

        # Sigmoid outputs should be between zero and one.
        for layer in layers[:3]:
            assert np.all((layer >= 0) & (layer <= 1))

        # ReLU outputs should be nonnegative.
        assert np.all(layers[3] >= 0)

    # Test the activation functions using known values.
    np.testing.assert_allclose(
        model.sigmoid(np.array([0.0])),
        [0.5]
    )

    np.testing.assert_allclose(
        model.relu(np.array([-2.0, 0.0, 3.0])),
        [0.0, 0.0, 3.0]
    )

    # Processing a batch should match processing inputs individually.
    batch_output = model.forward(test_inputs)
    individual_outputs = np.array([
        model.forward(inputs) for inputs in test_inputs
    ])

    np.testing.assert_allclose(batch_output, individual_outputs)

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()