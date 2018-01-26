"""
Contains the Perceptron class which can be trained and then used to linearly classify a dataset.
"""

class Perceptron:

    def __init__(self, dimensions: int):
        # Using w[0] as bias weight
        self.bias = 1
        self.threshold = 0
        self.dimensions = dimensions
        self.weights = [0 for __ in range(dimensions+1)]

    def sign(self, *inputs):
        # Guard mismatched vector size
        if len(inputs) != self.dimensions:
            raise ValueError

        inputs.insert(self.bias, 0)
        total = sum([x*w for x, w in zip(inputs, self.weights)])

        return 1 if total > self.threshold else -1


    def train(self, coordinates):
        """
        Provided a list of coordinates, where a coordinate consists of n input values
        and a correct value, update the perceptron weights using each coordinate.
        """


