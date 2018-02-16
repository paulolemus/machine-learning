"""
Contains the Perceptron class which can be trained and then used to linearly classify a dataset.
"""

import numpy as np

class Perceptron:

    def __init__(self, dimensions: int, learning_rate):
        # Using w[0] as bias weight
        self.bias = 1
        self.threshold = 0
        self.dimensions = dimensions
        self.learning_rate = learning_rate
        self.weights = np.zeros(dimensions+1)

    def _sign(self, scalar):
        """
        Return 1 if the given value is greater than threshold, -1 otherwise.
        """
        return 1 if scalar >= 0 else -1

    def classify(self, input_list):
        """
        Binary classification of the given inputs.
        Note that this does NOT train this perceptron.
        """
        input_vec = np.insert(input_list, 0, self.bias)
        return self._sign(self.weights.dot(input_vec))

    def train(self, dataset, labels):

        for coordinates, label in zip(dataset, labels):
            guess = self.classify(coordinates)
            input_vec = np.insert(coordinates, 0, self.bias)

            adjustments = self.learning_rate * (label-guess) * input_vec
            self.weights = self.weights + adjustments


