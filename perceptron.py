from numpy import array, dot, random
from random import choice


class Perceptron(object):
    def __init__(self):
        """
        - Initializes an array(1,3) of weights with random values from 0 to 1
        - alpha is a learning rate
        - n the number of training steps
        - errors is a list to collect all errors
        """
        self.weights = random.rand(3)
        self.alpha = 0.3
        self.n = 10000
        self.errors = []

    def step_function(self, x):
        """
        Step function implementation
       """
        if x < 0:
            return 0
        else:
            return 1

    def train(self, train_data):
        for _ in range(self.n):
            inputs, expected = choice(train_data)
            result = dot(inputs, self.weights)
            error = expected - self.step_function(result)
            self.errors.append(error)
            self.weights += self.alpha * error * inputs

        for inputs, _ in train_data:
            result = dot(inputs, self.weights)
            print("{}: {} -> {}".format(inputs[1:],
                                        result,
                                        self.step_function(result)))

train_data = [
    (array([1, 0, 0]), 0),
    (array([1, 0, 1]), 1),
    (array([1, 1, 0]), 1),
    (array([1, 1, 1]), 1),
]
p = Perceptron()
p.train(train_data)
