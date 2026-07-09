from math import exp


class Activation:
    @staticmethod
    def linear(x: float) -> float:
        return x

    @staticmethod
    def sigmoid(x: float) -> float:
        return 1 / (1 + exp(-x))
