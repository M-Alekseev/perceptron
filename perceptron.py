from numpy import array, dot, random  # , exp
import matplotlib.pyplot as plt


class Perceptron(object):
    def __init__(self, *args):
        """
        - Initializes an array(1,10) of weights with random values from 0 to 1
        - eta is a learning rate
        - n the number of training steps
        - errors is a list to collect all errors, u may use it for plotting
        - selected contains number of inputs that we want to recognize
        """
        self.weights = random.rand(10)
        self.eta = 0.003
        self.n = 2000
        self.errors = []
        self.selected = random.randint(1, size=10)
        # check if agrs fit the size of the weights
        for arg in args:
            assert (arg in range(len(self.weights))), "Out of range!"
            self.selected[arg] = 1

    def step_function(self, x):
        if x < 0:
            return 0
        else:
            return 1

    # def sigmoid(self, x):
    #     return 1 / (1 + exp(-x))

    def train(self, train_data):
        learned = False
        iteration = 0
        while not learned:
            for inputs, expected in list(zip(train_data, self.selected)):
                result = dot(inputs, self.weights)
                error = expected - self.step_function(result)
                self.errors.append(error)
                self.weights += self.eta * error * inputs
            iteration += 1
            if iteration >= self.n:
                learned = True
                print("iteration - {}".format(iteration))

    def test(self, test_data):
        for inputs in test_data:
            result = dot(inputs, self.weights)
            print("{}: {} -> {}".format(inputs[1:],
                                        result,
                                        self.step_function(result)))

    def plot_errors(self):
        plt.ylim([-1, 1])
        plt.xlim([0, 2500])
        plt.plot(self.errors)
        plt.show()


"""
    Each array in the list represents a 3x3 pixel image
    where 1 is a white color and 0 accordingly black
    | 1 | 0 | 0 |
    | 0 | 1 | 0 |
    | 0 | 0 | 1 |

    First column is used for a bias value
"""
train_data = [
    #     |b |        |        |       |
    # 1
    array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
    # 2   |b |        |        |       |
    array([1, 0, 0, 0, 1, 1, 1, 0, 0, 0]),
    # 3   |b |        |        |       |
    array([1, 0, 0, 0, 0, 0, 0, 1, 1, 1]),
    # 4   |b |        |        |       |
    array([1, 1, 0, 0, 1, 0, 0, 1, 0, 0]),
    # 5   |b |        |        |       |
    array([1, 0, 1, 0, 0, 1, 0, 0, 1, 0]),
    # 6   |b |        |        |       |
    array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
    # 7   |b |        |        |       |
    array([1, 1, 0, 0, 0, 1, 0, 0, 0, 1]),
    # 8   |b |        |        |       |
    array([1, 0, 0, 1, 0, 1, 0, 1, 0, 0]),
    # 9   |b |        |        |       |
    array([1, 1, 0, 1, 0, 1, 0, 0, 0, 0]),
    # 10  |b |        |        |       |
    array([1, 0, 0, 1, 0, 1, 0, 0, 0, 1]),
]

train_data = array(train_data)
# train_data[train_data > 0] = 255

if __name__ == "__main__":
    # Class takes arguments in range [0-9]
    p = Perceptron(8, 9)
    p.train(train_data)
    p.test(train_data)

    p.plot_errors()
