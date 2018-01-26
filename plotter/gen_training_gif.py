"""
Example on how to train your perceptron and then how to animate its separation line as it learns.
"""

from .perceptron import Perceptron

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import uniform

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
    dataset_size = 1000
    margin = 0.3

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
    plt.savefig('visualization_of_training_set_points.png')

    # Plot the guesses of the untrained perceptron
    plt.figure()
    plt.title('Untrained Perceptron Guesses on Training Set')
    plt.axis([start, end, start, end])
    plt.plot([start, end], [start, end], 'g--')

    for coordinate, label in dataset:
        guess = p.sign(coordinate)
        appearance = ''
        appearance += 'g' if guess == label else 'r'
        appearance += 'd' if label == 1 else '.'
        plt.plot(*coordinate, appearance)

    plt.savefig('untrained_perceptron_guesses_on_training_set.png')

    # Plot perceptron guesses after training once
    p = Perceptron(dimensions, learning_rate)
    p.train(dataset)

    plt.figure()
    plt.title('Singly trained Perceptron Guesses on Training Set')
    plt.axis([start, end, start, end])
    plt.plot([start, end], [start, end], 'g--')

    for coordinate, label in dataset:
        guess = p.sign(coordinate)
        appearance = ''
        appearance += 'g' if guess == label else 'r'
        appearance += 'd' if label == 1 else '.'
        plt.plot(*coordinate, appearance)

    plt.show()


if __name__ == '__main__':
    main()


