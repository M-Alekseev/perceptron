from numpy import array, dot, random, exp
from random import choice


class Perceptron(object):
    def __init__(self):
        """
        - Initializes an array(1,3) of weights with random values from 0 to 1
        - eta is a learning rate
        - n the number of training steps
        - errors is a list to collect all errors
        """
        self.weights = random.rand(3)
        self.eta = 0.3
        self.n = 10000
        self.errors = []

    def step_function(self, x):
        if x < 0:
            return 0
        else:
            return 1

    def sigmoid(x):
        return 1 / (1 + exp(-x))

    def train(self, train_data):
        for _ in range(self.n):
            inputs, expected = choice(train_data)
            result = dot(inputs, self.weights)
            error = expected - self.step_function(result)
            self.errors.append(error)
            self.weights += self.eta * error * inputs

    def test(self, test_data):
        for inputs, _ in test_data:
            result = dot(inputs, self.weights)
            print("{}: {} -> {}".format(inputs[1:],
                                        result,
                                        self.step_function(result)))

train_data = [
    # i is input
    # b is a bias
    # O - desired output
    #       b  i  i    O
    (array([1, 0, 0]), 0),
    (array([1, 0, 1]), 1),
    (array([1, 1, 0]), 1),
    (array([1, 1, 1]), 1),
]

if __name__ == "__main__":
    p = Perceptron()
    p.train(train_data)
    p.test(train_data)
