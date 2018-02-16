"""
Contains functions used for generating data sets.
"""


from random import uniform

def make_uniform(n, x_s, x_e, y_s, y_e, margin):
    """Generate points that meet the margin requirement for the arbitrary
    and optimal separation line y = x.
    """
    dataset = []

    for __ in range(n):
        x = uniform(x_s, x_e)
        y = uniform(y_s, y_e)
        distance = abs(x-y) / 2**0.5
        while distance < margin:
            x = uniform(x_s, x_e)
            y = uniform(y_s, y_e)
            distance = abs(x-y) / 2**0.5
        dataset.append([x, y])

    labels = [1 if y > x else -1 for x, y in dataset]

    return dataset, labels
