from fishai.neuron import Neuron

brain = Neuron(2)

result = brain.predict([2, 3])

print(result)

from fishai.activation import Activation
from fishai.layer import Layer

layer = Layer(
    neuron_count=3,
    input_count=2,
    activation=Activation.sigmoid
)

result = layer.predict([2, 3])

print(result)
