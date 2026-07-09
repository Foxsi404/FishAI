"""
FishAI

Copyright (c) 2026 Foxsi

Licensed under the MIT License.
"""

from random import uniform


class Neuron:
    def __init__(self, input_count: int):
        self.weights = [uniform(-1.0, 1.0) for _ in range(input_count)]
        self.bias = uniform(-1.0, 1.0)

     def predict(self, inputs: list[float]) -> float:
        if len(inputs) != len(self.weights):
            raise ValueError(
                f"Expected {len(self.weights)} inputs, got {len(inputs)}."
            )

        output = self.bias

        for weight, value in zip(self.weights, inputs):
            output += weight * value

        return output