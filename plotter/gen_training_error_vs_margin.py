"""
Generate training error vs margin.
"""

import matplotlib.pyplot as plt
import statistics

from random import uniform
try:
    from .perceptron import Perceptron
except:
    from perceptron import Perceptron

def generate_dataset(n, x_bounds, y_bounds, margin):
    """
    Generate dataset with constraints for separation line y=x
    """
    dataset = []

    for __ in range(n):
        x = uniform(*x_bounds)
        y = uniform(*y_bounds)
        distance = abs(x-y)/2**0.5

        while distance < margin:
            x = uniform(*x_bounds)
            y = uniform(*y_bounds)
            distance = abs(x-y)/2**0.5

        dataset.append([[x, y], 1 if y > x else -1])

    return dataset



def main():
    """
    Goal is to generate a graph showing the relationship margin vs training error.
    Large training and testing datasets are used.


    Procedure:
    1. train a perceptron with a large training set then evaluate using
       multiple testing sets for a fixed small margin.
    2. Store each error percentage as a list in a list.
    3. Repeat the previous two steps with increasing margin sizes.

    Storage format:
    List[List[margin, average, stdev]]
    """
    start = -400
    end = 400
    training_size = 1000
    testing_size = 1000
    margin_start = 0
    margin_end = 120
    margin_step = 1
    samples_per_margin = 100

    dimensions = 2
    learning_rate = 0.3

    experiment_data = []

    for margin in range(margin_start, margin_end, margin_step):
        print('margin', margin, '/', margin_end)

        training_errors = []

        for __ in range(samples_per_margin):
            training_set = generate_dataset(training_size, [start, end], [start, end], margin)
            testing_set  = generate_dataset(testing_size, [start, end], [start, end], margin)
            p = Perceptron(dimensions, learning_rate)
            p.train(training_set)

            incorrect_count = 0
            for coordinate, label in testing_set:
                if p.sign(coordinate) != label:
                    incorrect_count += 1

            training_error = incorrect_count / testing_size
            training_errors.append(training_error)


        average = statistics.mean(training_errors)
        std = statistics.pstdev(training_errors)
        experiment_data.append([margin, average, std])

    # Preprocess
    max_training_error = max([item[1] for item in experiment_data])

    # Plot without standard deviation
    plt.title('Training Error vs Margin for Perceptron')
    plt.xlabel('Margin')
    plt.ylabel('Training Error')
    plt.grid(True)
    #plt.axis([margin_start, margin_end, 0, max_training_error])
    plt.plot(
        [item[0] for item in experiment_data],
        [item[1] for item in experiment_data],
        'b.'
    )
    plt.text(
        0.05,
        0.9,
        'learning rate: {}\ntraining size: {}\ntesting size: {}\n samp/margin: {}'.format(learning_rate, training_size, testing_size, samples_per_margin),
        fontsize=14,
        verticalalignment='top',
    )
    #plt.savefig('training_error_vs_margin_for_perceptron.png')

    # plot with standard deviation
    plt.figure()
    plt.title('Training Error vs Margin for Perceptron with Deviation')
    plt.xlabel('Margin')
    plt.ylabel('Training Error')
    plt.errorbar(
        [item[0] for item in experiment_data],
        [item[1] for item in experiment_data],
        [item[2] for item in experiment_data],
        linestyle='None',
        ecolor='r',
        marker='^'
    )
    #plt.savefig('training_error_vs_margin_for_perceptron_with_deviation.png')

    plt.show()



if __name__ == '__main__':
    main()
