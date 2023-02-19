"""
nn.py - test out building a neural network from basic principals
"""
import numpy as np

class Layer:
    """Base class for a neural network layer"""

    def forward(self, x_input: np.ndarray) -> np.ndarray:
        """Forward pass of the layer for x_input"""
        return x_input

    def backward(self, x_input: np.ndarray, err: np.ndarray) -> np.ndarray:
        """Backward pass of the layer, given input x and backprop'ed error err"""
        return x_input - err


class Sigmoid (Layer):
    """Sigmoid activation function"""

    def forward(self, x_input: np.ndarray) -> np.ndarray:
        """Forward pass of the sigmoid layer"""
        return 1 / (1 + np.exp(-x_input))

    def backward(self, x_input: np.ndarray, err: np.ndarray) -> np.ndarray:
        """Backward pass of the sigmoid layer"""
        return err * x_input * (1 - x_input)


class ReLU (Layer):
    """ReLU activation function"""

    def forward(self, x_input: np.ndarray) -> np.ndarray:
        """Forward pass of the ReLU layer"""
        return np.maximum(0, x_input)

    def backward(self, x_input: np.ndarray, err: np.ndarray) -> np.ndarray:
        """Backward pass of the ReLU layer"""
        return err * (x_input > 0)


class Linear (Layer):
    """Linear layer"""

    def __init__(self, num_inputs: int, num_outputs: int):
        """Initialize the weights of the linear layer"""
        self.weights = np.random.randn(num_inputs, num_outputs)

    def forward(self, x_input: np.ndarray) -> np.ndarray:
        """Forward pass of the linear layer"""
        return np.matmul(x_input, self.weights)

    def backward(self, x_input: np.ndarray, err: np.ndarray) -> np.ndarray:
        """Backward pass of the linear layer"""
        return np.matmul(err, x_input.T)
        # return np.matmul(err, self.weights.T) - old code
        #return np.matmul((np.matmul (W, X) - Y), X.transpose()) - back_prop.py algo



class NeuralNetwork:
    """Neural network class"""

    def __init__(self):
        self.layers = []
        self.inputs = []
        self.errs = []
        self.training_log = []

    def add_layer(self, layer: Layer):
        """Add a layer to the neural network"""
        self.layers.append(layer)
        self.inputs.append(None)
        self.errs.append(None)

    def forward(self, x_input: np.ndarray) -> np.ndarray:
        """Forward pass of the neural network"""
        for i, layer in enumerate(self.layers):
            self.inputs[i] = x_input
            x_input = layer.forward(x_input)
        return x_input

    def backward(self, err: np.ndarray):
        """Backward pass of the neural network"""
        for i in range(len(self.layers) - 1, -1, -1):
            self.errs[i] = err
            err = self.layers[i].backward(self.inputs[i], err)

    def train(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int,
              learning_rate: float, batch_size: int):
        """Train the neural network"""
        self.training_log = []

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        if batch_size == 0:
            batch_size = len(x_train)

        for i in range(epochs):
            for j in range(0, len(x_train), batch_size):
                x_batch = x_train[j:j+batch_size]
                y_batch = y_train[j:j+batch_size]
                y_hat = self.forward(x_batch)
                err = y_batch - y_hat
                loss = self.loss(y_batch, y_hat)
                self.training_log.append (f"Epoch {i}, Batch {j}, Loss {loss}")
                self.backward(err)
                self.update_weights(learning_rate)

    def loss(self, y_hat: np.ndarray, y_train: np.ndarray) -> float:
        """Calculate the loss of the neural network"""
        return np.sum(np.square(y_hat - y_train))/y_train.shape[0]

    def update_weights(self, learning_rate: float):
        """Update the weights of the neural network"""
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Linear):
                layer.weights -= learning_rate * \
                    np.matmul(self.inputs[i].transpose(), self.errs[i])

    def predict(self, x_input: np.ndarray) -> np.ndarray:
        """Predict the output of the neural network"""
        return self.forward(x_input)

import random
def test():
    """Test the neural network"""

    # 1d identity with 1 neuron
    nn = NeuralNetwork()
    nn.add_layer(Linear(1, 1))
    x = []
    y = []
    for _ in range (10):
        x.append([random.randint(0, 10)])
        y.append(x[-1])

    nn.train(x, y, 10, 0.05, 0)
    #nn.train([[0], [1], [2], [3]], [[0], [1], [2], [3]], 10, 0.05, 0)
    x = [92]
    y = nn.predict(x)
    if abs(x - y)/x > .1:
        print("1 neuron identity test failed: input = ", x, "predicted output = ", y, "error = ", np.sum(np.square(x - y)))
        print(nn.layers[0].weights)
        print(nn.training_log)

    """
    # 2d identity with 1 layer of 2 neurons
    nn = NeuralNetwork()
    nn.add_layer(Linear(2, 2))
    
    x = []
    y = []
    for _ in range (5):
        x.append([random.randint(0, 100), random.randint(0, 100)])
        y.append(x[-1])
    nn.train(x, y, 10, 0.05, 0)
    x = [92, 97]
    y = nn.predict(x)
    print (x)
    print (y)
    percent_error = np.sum(np.square(x - y))/np.sum(np.square(x))
    if percent_error > .1 :
        print("1 layer 2d identity test failed: input = ", x, "predicted output = ", y, "error = ", percent_error)
        print(nn.layers[0].weights)
        print(nn.training_log)
    
    # 2d identity with 2 layers of 2 neurons
    nn = NeuralNetwork()
    nn.add_layer(Linear(2, 2))
    nn.add_layer(Linear(2, 2))
    x = []
    y = []
    for _ in range (100):
        x.append([random.randint(0, 100), random.randint(0, 100)])
        y.append(x[-1])
    nn.train(x, y, 500, 0.05, 10)
    x = [92, -17]
    y = nn.predict(x)
    if np.sum(np.square(x - y)) > .1:
        print("2 layer 2d identity test failed: input = ", x, "predicted output = ", y, "error = ", np.sum(np.square(x - y)))
        print(nn.layers[0].weights)
        print(nn.layers[1].weights)
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
    """

test()


