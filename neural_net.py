"""
nn.py - a neural network built on numpy from basic principals (so I could learn)

For details, see the README.md file.

Note: pylint has been configured to ignore/allow 1 and 2 letter variable names (as
god intended), since x = input, y = output, w = weight, b = bias, err = error to be
backprop'ed, dw = weight gradient, db = bias gradient, nn = neural net, are all
common variable names in neural nets, so artificially forcing longer names would be
silly.
"""
import numpy as np
import matplotlib.pyplot as plt

class LossFunction:
    """Base class for a loss function - each method should be overridden"""

    def loss(self, y: np.ndarray, y_train: np.ndarray) -> float:  # pylint: disable=unused-argument
        """Loss function for y vs y_train (actual vs expected output)"""
        return None

    def backward(self, y: np.ndarray, y_train: np.ndarray) -> np.ndarray:  # pylint: disable=unused-argument
        """The gradient of loss with respect to y, used for backprop"""
        return None

class MSE (LossFunction):
    """Mean squared error of y vs y_train (actual vs expected output)"""

    def loss(self, y: np.ndarray, y_train: np.ndarray) -> float:
        """Forward pass of the MSE loss function"""
        return np.sum((y - y_train) ** 2) / y.shape[1]

    def backward(self, y: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """The gradient of loss with respect to y, used for backprop"""
        return 2 * (y - y_train) / y.shape[1]

class CrossEntropy (LossFunction):
    """Cross Entropy loss function of y vs y_train (actual vs expected output)
    See https://machinelearningmastery.com/cross-entropy-for-machine-learning"""
    def loss(self, y: np.ndarray, y_train: np.ndarray) -> float:
        """Forward pass of the Cross Entropy loss function"""
        return -np.sum(y_train * np.log(y) + (1 - y_train) * np.log(1 - y)) / y.shape[1]

    def backward(self, y: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """The gradient of loss with respect to y, used for backprop"""
        return -y_train/y + (1-y_train)/(1-y)


class Layer:
    """Base class for a neural network layer - each method should be overridden"""

    def forward(self, x: np.ndarray) -> np.ndarray:       # pylint: disable=unused-argument
        """Output of layer given input x from the previous layer"""
        return None

    def backward(self, x: np.ndarray, dy: np.ndarray) -> np.ndarray:  # pylint: disable=unused-argument
        """Backward pass of the layer: given x_input and backprop'ed output error dy
        from the next layer, compute the backprop'ed error for the previous layer"""
        return None

    def adjust_weights (self, learning_rate) -> None: # pylint: disable=unused-argument
        """Adjust the weights - linear layer overrides"""

class Sigmoid (Layer):
    """Sigmoid activation function"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid output input x from the previous layer"""
        return 1 / (1 + np.exp(-x))

    def backward(self, x: np.ndarray, dy: np.ndarray) -> np.ndarray:
        """Backward pass of the sigmoid layer"""
        sig = self.forward(x)
        return dy * sig * (1 - sig)

class ReLU (Layer):
    """ReLU activation function"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass"""
        return np.maximum(0, x)

    def backward(self, x: np.ndarray, dy: np.ndarray) -> np.ndarray:
        """Backward pass"""
        return dy * (x > 0)

class ReLU2 (Layer):
    """ReLU2 activation function. Like ReLU, but negative values just decrease (but not to zero)
    to see if it helps by eleminating the 'dead neuron' problem RELU creates. Unfortunately, this
    does worse on the test data"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass"""
        return np.maximum(.1*x, x)

    def backward(self, x: np.ndarray, dy: np.ndarray) -> np.ndarray:
        """Backward pass of the ReLU2 layer"""
        return dy * (x > 0) + .1 * dy * (x <= 0)

class Linear (Layer):
    """Linear layer"""

    def __init__(self, num_inputs: int, num_outputs: int):
        """Initialize weights and bias with small random values"""
        self.w = .2 * np.random.rand(num_outputs, num_inputs) - .1
        self.b = .2 * np.random.rand(num_outputs, 1) - .1
        self.dw = None
        self.db = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass"""
        return np.matmul(self.w, x) + self.b

    def backward(self, x: np.ndarray, dy: np.ndarray) -> np.ndarray:
        """Backward pass"""
        num_records = x.shape[1]
        self.dw = np.matmul(dy, x.T) / num_records
        self.db = np.sum (dy, axis = 1, keepdims=True) / num_records
        backprop_error = np.matmul(self.w.T, dy)
        return backprop_error

    def adjust_weights (self, learning_rate) -> None:
        """adjust weights based on gradients computed in backwards pass"""
        self.w -= self.dw * learning_rate
        self.b -= self.db * learning_rate

class NeuralNet:
    """Neural network class"""

    def __init__(self, loss_function = MSE(), seed = 0):
        self.debug_trace = False
        self.layers = []
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

    def forward(self, x_train: np.ndarray) -> list:
        """Forward pass; similar to the "predict" method, but tracks (and returns) a list of
        inputs at each layer for backprop. The last element of inputs is the output"""
        x_input = x_train
        inputs = [x_input]
        for layer in self.layers:
            x_input = layer.forward(x_input)
            inputs.append(x_input)
        return inputs

    def backward(self, dy: np.ndarray, inputs : list):
        """Backward pass of the neural network
        Takes the gradient of the loss function (dy) as input, and "back propogates" it to to
        each layer by iteratively computing the gradient of each layers output as a function
        of it's input
        """
        for i in range(len(self.layers) - 1, -1, -1):
            dy = self.layers[i].backward(inputs[i], dy)

    def train(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int,
              batch_size : int = 0):
        """Train the neural network"""
        self.loss_per_epoch = []

        if batch_size == 0:
            batch_size = x_train.shape[1]

        for _ in range(epochs):
            y_hat = self.predict(x_train)
            self.loss_per_epoch.append (self.loss_function.loss(y_hat, y_train))
            for j in range(0, x_train.shape[1], batch_size):
                # Update weights after processing a mini-batch
                x_batch = x_train[:, j:j+batch_size]
                y_batch = y_train[:, j:j+batch_size]
                inputs = self.forward(x_batch)
                y = inputs[-1]
                dy = self.loss_function.backward(y, y_batch)
                self.backward(dy, inputs)
                self.update_weights(self.learning_rate)

    def auto_train(self, x_train: np.ndarray, y_train: np.ndarray, \
                   x_valid: np.ndarray, y_valid: np.ndarray, target_loss : int): # pylint: disable=too-many-arguments, too-many-locals
        """Train the neural network without any hyper parameters"""
        self.loss_per_epoch = []

        # Defaults
        batch_size = x_train.shape[1]
        if batch_size > 32:
            batch_size = 16
        learning_rate = self.learning_rate  # Initial learning rate; decays with exponential average
        decay = .95                         # learning_rate's exponential average decay rate per batch
        max_epochs = 50000                  # Terminates training if we hit this many epochs

        # Start training
        while True:
            # Log the loss for this epoch.
            y = self.predict(x_valid)
            loss = self.loss(y, y_valid)
            self.loss_per_epoch.append (loss)
            
            # Stop training if we hit the target_loss, or if we've completed max_epochs
            if loss < target_loss:
                return
            if len(self.loss_per_epoch) > max_epochs:
                return

            # Run training over mini-batches of batch_size
            for j in range(0, x_train.shape[1], batch_size):
                # Update weights after processing a mini-batch
                x_batch = x_train[:, j:j+batch_size]
                y_batch = y_train[:, j:j+batch_size]
                inputs = self.forward(x_batch)
                y = inputs[-1]
                dy = self.loss_function.backward(y, y_batch)
                self.backward(dy, inputs)
                self.update_weights(learning_rate)

                # If new weights cause a loss increase, halve the learning rate and try again
                last_learning_rate = learning_rate
                loss = self.loss(y, y_batch)
                while True:
                    y = self.predict(x_batch)
                    new_loss = self.loss(y, y_batch)
                    if new_loss < loss or last_learning_rate < 0.00001:
                        break
                    last_learning_rate /= 2
                    self.update_weights(-last_learning_rate)
                learning_rate = learning_rate * decay + last_learning_rate * (1 - decay)

    def update_weights(self, learning_rate: float):
        """Update the weights of the neural network"""
        for layer in self.layers:
            layer.adjust_weights (learning_rate)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """The neural net's prediction of the output for x_input"""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def loss (self, y_hat : np.ndarray, y : np.ndarray) -> float:
        """Compute loss"""
        return self.loss_function.loss(y_hat, y)

    def plot_loss(self):
        """Show loss per epoch"""
        plt.plot(self.loss_per_epoch)
        plt.title("Loss per epoch", loc = "left")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    def print_weights(self):
        """Print the weights for diagnostics"""
        layer_num = 1
        for layer in self.layers:
            if isinstance(layer, Linear):
                print (f"Weights and biases for linear layer {layer_num}:")
                print (layer.w, "\n", layer.b)
            layer_num += 1
