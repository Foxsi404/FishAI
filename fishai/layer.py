from fishai.neuron import Neuron


class Layer:
    def __init__(self, neuron_count: int, input_count: int, activation):
        self.neurons = [
            Neuron(input_count, activation)
            for _ in range(neuron_count)
        ]
    def predict(self, inputs: list[float]) -> list[float]:
        return [neuron.predict(inputs) for neuron in self.neurons]
