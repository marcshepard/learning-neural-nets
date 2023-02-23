"""
nn.py - test out building a more general neural network from basic principals
It was built using the ideas from back_prop.py, as well as this most excellent
article: https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795.

Note: pylint has been configured to ignore/allow 1 and 2 letter variable names (as god intended),
since x = input, y = output, w = weight, b = bias, err = error to be backprop'ed, dW = weight
gradient, dB = bias gradient, nn = neural net, are all common variable names in neural nets, so
artificially forcing longer names would be silly.
"""
from typing import Tuple
import numpy as np

# BackpropOutput is a tuple of (backprop_error, dW, db)
BackpropOutput = Tuple[np.ndarray, np.ndarray, np.ndarray]

def trace (*args):
    """Print out the arguments if debug tracing is enabled"""
    if False:           # pylint: disable=using-constant-test
        print (*args)

class Layer:
    """Base class for a neural network layer - each method should be overridden"""

    def forward(self, x: np.ndarray) -> np.ndarray:       # pylint: disable=unused-argument
        """Forward pass of the layer for x"""
        return None

    def backward(self, x: np.ndarray, err: np.ndarray) -> BackpropOutput:  # pylint: disable=unused-argument
        """Backward pass of the layer, given x_input and backprop'ed output error err
        Returns: (back_err, dw, db) where back_err is the error that should be backproped to the
        previous layer, dw is the gradient of the weights, and db is the gradient of the bias"""
        return None, None, None


class Sigmoid (Layer):
    """Sigmoid activation function"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of the sigmoid layer"""
        return 1 / (1 + np.exp(-x))

    def backward(self, x: np.ndarray, err: np.ndarray) -> BackpropOutput:
        """Backward pass of the sigmoid layer"""
        return err * x * (1 - x), None, None


class ReLU (Layer):
    """ReLU activation function"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of the ReLU layer"""
        return np.maximum(0, x)

    def backward(self, x: np.ndarray, err: np.ndarray) -> BackpropOutput:
        """Backward pass of the ReLU layer"""
        return err * (x > 0), None, None


class Linear (Layer):
    """Linear layer"""

    def __init__(self, num_inputs: int, num_outputs: int):
        """Initialize weights and bias with random values"""
        self.w = np.random.randn(num_outputs, num_inputs)
        self.b = np.random.randn(num_outputs, 1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass"""
        return np.matmul(self.w, x) + self.b

    def backward(self, x: np.ndarray, err: np.ndarray) -> BackpropOutput:
        """Backward pass"""
        num_records = x.shape[1]
        dw = np.matmul(err, x.T) / num_records
        db = np.sum (err, axis = 1, keepdims=True) / num_records
        backprop_error = np.matmul(self.w.T, err)
        return backprop_error, dw, db

class NeuralNetwork:
    """Neural network class"""

    def __init__(self, seed = 99):
        self.layers = []
        self.inputs = []
        self.dws = []
        self.dbs = []
        self.training_log = []
        self.learning_rate = 0.05
        np.random.seed(seed)

    def add_layer(self, layer: Layer):
        """Add a layer to the neural network"""
        self.layers.append(layer)
        self.inputs.append(None)
        self.dws.append(None)
        self.dbs.append(None)

    def forward(self, x_input: np.ndarray) -> np.ndarray:
        """Forward pass of the neural network"""
        for i, layer in enumerate(self.layers):
            self.inputs[i] = x_input
            x_input = layer.forward(x_input)
        return x_input

    def backward(self, err: np.ndarray):
        """Backward pass of the neural network"""
        for i in range(len(self.layers) - 1, -1, -1):
            err, self.dws[i], self.dbs[i] = self.layers[i].backward(self.inputs[i], err)
            trace ("backprop dW", self.dws[i], "dB", self.dbs[i])

    def train(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int,
              batch_size : int= 0):
        """Train the neural network"""
        self.training_log = []

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        if batch_size == 0:
            batch_size = x_train.shape[1]

        for i in range(epochs):
            for j in range(0, len(x_train), batch_size):
                x_batch = x_train[:, j:j+batch_size]
                y_batch = y_train[:, j:j+batch_size]
                y_hat = self.forward(x_batch)
                err = y_hat - y_batch
                loss = self.loss(y_batch, y_hat)
                self.training_log.append (f"Epoch {i}, Batch {j}, Loss {loss}")
                trace ("Before backprop, weights are", self.layers[0].w, self.layers[0].b)
                self.backward(err)
                self.update_weights(self.learning_rate)
                trace ("After backprop, weights are", self.layers[0].w, self.layers[0].b)
                y_hat = self.forward(x_batch)
                new_loss = self.loss(y_batch, y_hat)
                last_learning_rate = self.learning_rate
                while new_loss > loss:
                    self.update_weights(-last_learning_rate)
                    last_learning_rate /= 2
                    self.update_weights(last_learning_rate)
                    trace ("That was worse - trying weights", self.layers[0].w, self.layers[0].b)
                    y_hat = self.forward(x_batch)
                    new_loss = self.loss(y_batch, y_hat)

    def loss(self, y_hat: np.ndarray, y_train: np.ndarray) -> float:
        """Calculate the loss of the neural network"""
        return np.sum(np.square(y_hat - y_train))/y_train.shape[0]

    def update_weights(self, learning_rate: float):
        """Update the weights of the neural network"""
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Linear):
                layer.w -= learning_rate * self.dws[i]
                layer.b -= learning_rate * self.dbs[i]

    def predict(self, x_input: np.ndarray) -> np.ndarray:
        """Predict the output of the neural network"""
        return self.forward(x_input)


def test_1_variable_identity():
    """1 variable identity (1 input, 1 output)"""
    test_name = "1 neuron, 1 input identity"
    nn = NeuralNetwork()
    nn.add_layer(Linear(1, 1))
    x = np.random.randint(20, size=(1, 10))
    y = x
    nn.train(x, y, 100)
    x = [92]
    y = [92]
    y_hat = nn.predict(x)
    pct_error = abs((y - y_hat)/y)
    if pct_error > .1:
        print(test_name, "with input =", x, ", expected output =", y, \
                ", actual output =", y_hat)
        print ("Weight/bias =", nn.layers[0].w, nn.layers[0].b)
        print("Test failed: error percent = ", pct_error, ">.1")
    else:
        print (test_name, "passed")

def test_1_layer_sum():
    """1 layer sum (2 inputs, 1 output)"""
    test_name = "1 neuron, 2 input sum"
    nn = NeuralNetwork()
    nn.add_layer(Linear(2, 1))
    x = np.random.randint(20, size=(2, 10))
    y = x.sum(axis=0, keepdims=True)
    nn.train(x, y, 100)
    x = np.ndarray((2, 1))
    x[0][0] = 92
    x[1][0] = 17
    y = [109]
    y_hat = nn.predict(x)
    pct_error = abs((y - y_hat)/y)
    if pct_error > .1:
        print(test_name, "with input = ", x, "expected output = ", y, \
                "actual output = ", y_hat)
        print("Test failed: error percent = ", pct_error, ">.1")
    else:
        print (test_name, "passed")

def test_2_layer_2_variable_identity():
    """2d identity test with 2 layer"""
    test_name = "2 layer, 2 input identity"
    nn = NeuralNetwork()
    nn.add_layer(Linear(2, 3))
    nn.add_layer(Linear(3, 2))
    x = np.random.randint(20, size=(2, 10))
    y = np.copy(x)
    nn.train(x, y, 100)
    x = np.ndarray((2, 1))
    x[0][0] = 92
    x[1][0] = -17
    y = x
    y_hat = nn.predict(x)
    pct_error = nn.loss(y_hat, y)/np.sum(np.square(y))
    if pct_error > .1:
        print(test_name, "with input = ", x, "expected output = ", y, \
                "actual output = ", y_hat)
        print("Test failed: error percent = ", pct_error, ">.1")
    else:
        print (test_name, "passed")

def test_activation():
    """ 2d identity with 1 layer of 2 neurons and ReLU activation"""
    nn = NeuralNetwork()
    nn.add_layer(Linear(2, 2))
    nn.add_layer(ReLU())
    nn.train([[0, 0], [0, 1], [1, 0], [1, 1]], [[0, 0], [0, 1], [1, 0], [1, 1]], 0, 0.05, 0)
    x = [92, 94]
    y = nn.predict(x)
    if np.sum(np.square(x - y)) > .1:
        print("2 layer 2d identity test w ReLU activation failed: input = ", x, "predicted output = ", y, "error = ", np.sum(np.square(x - y)))
        print(nn.layers[0].weights)
        print(nn.training_log)

    # 2d identity with 1 layer of 2 neurons and ReLU activation
    nn = NeuralNetwork()
    nn.add_layer(Linear(2, 2))
    nn.add_layer(ReLU())
    nn.train([[0, 0], [0, 1], [1, 0], [1, 1]], [[0, 0], [0, 1], [1, 0], [1, 1]], 0, 0.05, 0)
    x = [92, 94]
    y = nn.predict(x)
    if np.sum(np.square(x - y)) > .1:
        print("2 layer 2d identity test w ReLU activation failed: input = ", x, "predicted output = ", y, "error = ", np.sum(np.square(x - y)))
        print(nn.layers[0].weights)
        print(nn.training_log)

def test():
    """Test the neural network"""
    test_1_variable_identity()
    test_1_layer_sum()
    test_2_layer_2_variable_identity()

test()


