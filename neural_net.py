"""
nn.py - test out building a more general neural network from basic principals

For details, see the README.md file.

Note: pylint has been configured to ignore/allow 1 and 2 letter variable names (as
god intended), since x = input, y = output, w = weight, b = bias, err = error to be
backprop'ed, dw = weight gradient, db = bias gradient, nn = neural net, are all
common variable names in neural nets, so artificially forcing longer names would be
silly.
"""
from typing import Tuple
import numpy as np

# BackpropOutput is a tuple of (backprop_error, dW, db)
BackpropOutput = Tuple[np.ndarray, np.ndarray, np.ndarray]

class LossFunction:
    """Base class for a loss function - each method should be overridden"""

    def loss(self, y_hat: np.ndarray, y: np.ndarray) -> float:  # pylint: disable=unused-argument
        """Forward pass of the loss function for y_hat and y"""
        return None

    def backward(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:  # pylint: disable=unused-argument
        """Backward pass of the loss function for y_hat and y"""
        return None

class MSE (LossFunction):
    """Mean Squared Error loss function"""
    def loss(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        """Forward pass of the MSE loss function"""
        return np.sum((y_hat - y) ** 2) / y_hat.shape[1]

    def backward(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Backward pass of the MSE loss function"""
        return 2 * (y_hat - y) / y_hat.shape[1]

class CrossEntropy (LossFunction):
    """Cross Entropy loss function"""
    def loss(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        """Forward pass of the Cross Entropy loss function"""
        return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / y_hat.shape[1]

    def backward(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Backward pass of the Cross Entropy loss function"""
        return (y_hat - y) / (y_hat * (1 - y_hat))

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
        """Initialize weights and bias with random values between 0 and 1"""
        self.w = np.random.rand(num_outputs, num_inputs)
        self.b = np.random.rand(num_outputs, 1)

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

class NeuralNet:
    """Neural network class"""

    def __init__(self, loss_function = MSE(), seed = 0):
        self.debug_trace = False
        self.layers = []
        self.inputs = []
        self.dws = []
        self.dbs = []
        self.loss_function = loss_function
        self.loss_per_epoch = None
        self.learning_rate = 0.05
        if seed != 0:
            np.random.seed(seed)

    def trace (self, *args):
        """Print out the arguments if debug tracing is enabled"""
        if self.debug_trace:
            print (*args)

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
            if self.dws[i] is not None:
                self.trace ("backprop layer", i, "w/b =", self.layers[i].w, self.layers[i].b, ", dw/db =", self.dws[i], self.dbs[i])

    def train(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int,
              batch_size : int= 0):
        """Train the neural network"""
        self.loss_per_epoch = []

        if batch_size == 0:
            batch_size = x_train.shape[1]

        for _ in range(epochs):
            y_hat = self.forward(x_train)
            self.loss_per_epoch.append (self.loss_function.loss(y_hat, y_train))
            for j in range(0, len(x_train), batch_size):
                x_batch = x_train[:, j:j+batch_size]
                y_batch = y_train[:, j:j+batch_size]
                y_hat = self.forward(x_batch)
                loss = self.loss_function.loss(y_hat, y_batch)
                err = self.loss_function.backward(y_hat, y_batch)
                self.backward(err)
                self.update_weights(self.learning_rate)
                y_hat = self.forward(x_batch)
                new_loss = self.loss_function.loss(y_hat, y_batch)
                last_learning_rate = self.learning_rate
                while new_loss > loss and last_learning_rate > 0.0001:
                    last_learning_rate /= 2
                    self.update_weights(-last_learning_rate)
                    y_hat = self.predict(x_batch)
                    new_loss = self.loss_function.loss(y_hat, y_batch)

    def update_weights(self, learning_rate: float):
        """Update the weights of the neural network"""
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Linear):
                layer.w -= learning_rate * self.dws[i]
                layer.b -= learning_rate * self.dbs[i]

    def predict(self, x_input: np.ndarray) -> np.ndarray:
        """Predict the output of the neural network"""
        return self.forward(x_input)
