# dissecting-perceptrons
ICS 435 - Machine Learning, Assignment #2

## Assignment

Examine the training error of a perceptron as a function of:

* Size of training data (N points)
* Ratio of training data to test data
* Margin (distance from optimal division to closest point)

The goal of the exercise is to find a quantitative measure of perceptron training. 


## Graphs

Must generate the following graphs:

* Margin vs training error
* Number of training points vs training error for fixed testing set 3x
* Ratio of size of training set to test set vs training error 
* Learning rate vs training error
* Perceptron separation error from optimal separation vs margin for accurate perceptron


## Building graphs

Type the following snippet of code from the root of this repo:

```
python3 -m plotter
```
Alternatively if you are using pipenv, use the following:
```
pipenv install
pipenv run python -m plotter
```
