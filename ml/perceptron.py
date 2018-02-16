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
        self.weights = np.zeroes(dimensions+1)

    def sign(self, inputs):
        """
        Classify a single coordinate.
        """
        # Guard mismatched vector size
        if len(inputs) != self.dimensions:
            raise ValueError

        input_vector = [self.bias] + inputs
        total = self.weights.dot(input_vector)
        return 1 if total > self.threshold else -1


    def train(self, dataset):
        """
        Provided a dataset, where a dataset consists of a list of coordinates paired
        with their proper classification , update the perceptron weights using each
        coordinate.

        Acceptable format:
        dataset: List[coordinates, label]
        """
        for coordinates, label in dataset:
            guess = self.sign(coordinates)
            input_vec = [self.bias] + coordinates

            adjustments = [self.learning_rate*(label-guess)*x_in for x_in in input_vec]
            self.weights = self.weights * adjustments


