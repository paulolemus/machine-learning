"""
Example on how to train your perceptron and then how to animate its separation line as it learns.
"""

from perceptron import Perceptron

from random import uniform
import matplotlib.pyplot as plt

def generate_dataset(n, x_s, x_e, y_s, y_e, margin):
    """
    Generate a dataset with the given constraints for the separation line y = x
    """
    dataset = []

    for __ in range(n):
        x = uniform(x_s, x_e)
        y = uniform(y_s, y_e)

        while abs(x - y) < margin:
            x = uniform(x_s, x_e)
            y = uniform(y_s, y_e)

        dataset.append([[x, y], 1 if y > x else -1])

    return dataset


def main():
    start = 0
    end = 100
    dataset_size = 100
    margin = 2

    dataset = generate_dataset(dataset_size, start, end, start, end, margin)

    dimensions = 2
    learning_rate = 0.3
    p = Perceptron(dimensions, learning_rate)

    # Plot of all data points colored according to their label
    plt.title('Visualization of Training Set Points')
    plt.axis([start, end, start, end])
    plt.plot([start, end], [start, end], 'g--')
    for coordinate, label in dataset:
        if label == 1:
            plt.plot(*coordinate, 'yd')
        else:
            plt.plot(*coordinate, 'b.')
    plt.show()

    # Plot the guesses of the untrained perceptron
    plt.figure()
    plt.title('Untrained Perceptron Guesses on Training Set')
    plt.axis([start, end, start, end])
    plt.plot([start, end], [start, end], 'g--')
    plt.show()

if __name__ == '__main__':
    main()


