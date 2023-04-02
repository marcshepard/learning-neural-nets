"""
backprop.py - a neural network built on numpy from basic principals (so I could learn)

Prereqs: numpy, matplotlib

Goals and design:
Initially, I sketeched out the math to train a single linear layer on scratch paper and tested it,
and then generalized with the help of these
articles:
* https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795
* https://en.wikipedia.org/wiki/Backpropagation
Although the implementation here has some significant differences from those articles.

Here is a quick overview of the architecture. A full understanding requires that you understand
derivatives and matrix multiplication and have read the README.md file in this directory.

The neural network is a collection of layers, where linear layers and non-linear activation
functions are separate (ideally alternating) layers.

The linear layers are modeled as:
    f(x) = w*x + b      (matrix multiplication and addition, which greatly simplifies the math)
Where if:
* n is the number of neurons in the previous layer (so the dimension of a single input value)
* m is the number of neurons in this layer (and so the dimension of a single output value)
* k is the number of input records being processed at a time.
Then:
* x is an nxk matrix of inputs; each column is an input record
* w is an mxn weight matrix; each row are the weights of a single neuron
* b is a  mx1 bias matrix, where each row is the bias for the corresponding neuron
* f(x) is an mxk matrix of outputs; each column is the output for the corresponding input column

Here's more intuition and math behind the design of the linear layer:
1. A neuron takes n inputs and produces a single output using a linear function. So a neuron that
   takes n inputs can be modeled as a weight vector w=(w1, ..., wn) and a bias, and produces, for
   a given input vector x=(x1, x2, ..., xn), an output of w1*x1 + ... + wn*xn + b.
2. Matrix multiplication simplies the math. We can model a neuron's weight as a length-n row-vector
   (a 1xn matrix) and a bias, and the input as a length-n column vector (an nx1 matrix). Then the
   calculation for the output is f(x) = w*x + b, where "*" refers to matrix multiplication
3. An entire linear layer of m neurons can therefor be modeled as an mxn weight matrix (where each
   row represents the weights of the corresponding neuron), and a mx1 bias matrix. In this case, we
   can still model the output f(x) = w*x + b. Note that the output, f(x), is a (1xm) column matrix
   of outputs from each neuron.
4. An entire batch of k inputs training inputs can therefore be modeled as an nxk matrix, where n is
   the number of inputs in the first layer, and each column is a single training input. We can stil
   model the output as f(x) = w*x + b. When doing training, the expected output, y, is modeled as a
   (kxm) matrix, where each column is the expected output from the corresponding column of x.
5. How do we pick the initial weights and bias? It turns out they can't all have the same value or
   else training with backprop will always keep them the same. So we start off with initial random
   small values. This initialization requires more discussion in the issues section at the end.
6. Some papers/models on nerural nets use f(x) = x*w + b (so inputs and outputs are rows, weights
   are columns); equally valid, but I had to pick one.
7. Some math papers say biases don't need to be modeled as one can get the same effect by adding a
   column of "1s" to the weight matrix. While mathematically true, this technique can be
   computationally inefficient during backprop/training compared to explicitly modeling the biases.

Next, let's discuss the non-linear (activation) layers in the network
1. Without non-linear layers, there is no point in having more than 1 layer, since if f(x) and g(x)
   are linear functions, then so is f(g(x)). So a neural net with no non-linear layers can't model
   anything more complex than a linear function of it's inputs, which is extremely limited.
2. But what type of non-linear function is needed? As described in the first section, real neurons
   have what we can call "activation thresholds", where the output (axon) only fires if the weighted
   inputs are above a given threshold. So artificial neural nets, inspired by real neurons, follow
   this pattern by providing non-decreasing activation functions that just transform each neurons
   output in a manner that doesn't depend on any of the other neurons. In other words, an activation
   function a(x) is a non-decreasing function applied element-wise to it's input matrix.
3. As a test case, I've implemented two common activation layers:
a. ReLU; x -> max (x, 0). A very common activation function between internal layers. 
b. Sigmoid; x -> 1/(1 + e^-x). A common activation function for the output layer of binary
   classification network (e.g, is the input a picture of a boat or not?) as it produces
   a value between 0 and 1, so the model is > .5 means "yes" else "no".

So now we can create a neural network by chaining together layers; alternating linear layers with
non-linear activation layers. But how do we train it to produce the right answers? That requires:
1. Training data, which consisting of a set of inputs and their corresponding expected outputs.
   Implemented as matrices x and y, whose columns are input and expected output vectors.
2. Defining a "loss function" that measures how close the actual output (f(x)) is to the expected
   output (y). I've implemented two common loss functions:
   a. MSE - mean squared error; the average squared distance between each column of y and f(x)
   b. CrossEntropy - a standard loss function for classification networks
2. Perform a "forward pass" of the input data in batches. A given training session is typically
   divided into "epochs" (how many times all the training data is processed), and "mini-batches"
   (the number of records processed at a time within each epoch; as a hueristic, processing 10-30 at
   a time typically produces good fit and performance).
3. After each forward pass, compute the loss and then do a backward pass (backprop) to adjust the
   weights and biases so as to make the loss smaller the next time around (more on this critical
   step later). After each epoch, I also compute the loss over some validation set to help
   understand the progress of training per epoch.
4. Finally, run the neural net on some validation data it has not seen before to see how effective
   it is in making future predictions. The average loss on this data will generally be higher than
   the loss on the training data as training results in some degree of overfiting.

Here's a bit of intuition behind backprop before we get to the algo:
* First, recall the neural net is basically f(x) = f1(f2(...(fk(x)...) where each fi is either an
  activation or linear layer.
* Each linear layer has a weight matix and bias vector that need to be adjusted to improve the
  network. But how should we adjust them? Here's where calculus kicks in. We can pretend each
  weight matrix, W, is actually a matrix of variables (w11, w12, ...) instead of fixed values. We
  can then compute the gradient of the loss function with respect to each of these variables and
  evaluate that gradient at the given current values for the weights. Subtracting a "small multiple"
  (aka learning rate) of the gradient should decrease loss by moving towards the minima (although if
  the multiplier is too large we can go past the minimum and make the network worse - see issues
  section for more info).
* But how do we compute the gradients of the loss function with respect to the weights? Here the
  chain rule kicks in. We iteratively compute the gradient of the loss function with respect to the
  inputs of each layer, from back to front. First we compute the gradient of the loss function
  itself with respect to it's input, f(x):
    dl/dy = gradient of loss function with respect to y=f(x)
  For the standard loss functions, this is a very simple calculation (see the code - it's one line
  of Python each and the deriviative formula is known to anyone who has taken calculus).

Backprop works backwards one layer at a time using the chain rule. In the code, each layer gets:
* x - the input it got during the forward pass
* dy - the gradient of the loss function with respect to the output it produced; this is computed by
  the next layer and passed back to it (which is why this is called "back prop", as this layer needs
  to do the same). The initial dy (passed to the last layer) is the gradient of the loss function
  with respect to it's input.
The layer then needs to produce a "dy" value for the previous layer (which is the same thing as the
it's "dx" value - the gradient of it's input with respect to it's dy). For the activation layers,
this is a trivial calculation (see the code - it's one line of python and how to get the derivative
will be known to anyone who has taken calculus).
For the linear layers, it's a bit more complex, but if you work through the math with a couple of
examples, you will see:
    dw = dy * x.transpose       # The gradient of the weight; for weight adjustment
    db = avg_over_cols(dy)      # The gradient of the bias; for bias adjustment
    dx = w.T * dy               # The gradient to backprop - to become dy for the previous layer
The first can be worked out by hand (tricky, but that's how I originally did it). The second is
obvious if you think of what would happen if the bias was included in the matrix w as a final column
vector and x having an added final row vector of 1's to pick up the biases. The last also requires
some thought/examples.

Issues/thoughts:
1. I've implemented an auto_train method eliminates a few hyper parameters and results in more
efficient training:
 * Dynamic learning rate. Starts with a relatively high rate (.05), but after adjusting weights for
   each mini-batch, if the loss function doesn't decrease for that mini-batch, cut the weight
   adjustment in half and try again. Idea is to make sure gradient decent didn't bypass the local
   minima by so much as to make loss increase. Then adjust the learning rate using exponential
   average of the new and old rates; the next mini-batch will start with that.
 * Specify target loss rather than number of epochs. Makes training much easier, since loss rate is
   the goal (not number of epochs)
 * Randomly shuffle the training data each epoch so we don't overfit on the mini-batches. This made
   a huge difference in some of the test cases (going from thousands of epochs needed to reach the
   target to a much smaller handful).
2. At some point should make the linear layer's weight function configurable so folks can override
   the default, as it seems like different initializations yeild very different results.
3. ReLU activation can sometimes make everything go to 0 depending on the weight initialization.
   Training with both negative and positive values helps fix this.
4. The right loss function depents on the problem space. MSE OK for regressions, but when training a
   linear model like f(x) = 3x it will give wildly different gradients/errors for the same percent
   error depending on if the training data is near zero or far away.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# Disable some pylint warnings; I like to use x, y, dx, dy as variable names in this file
# pylint: disable=invalid-name

# Loss functions are how we measure how well the network is doing
# There is an abstract base class, as well as two concrete classes; MSE (mean-squared error; a
# popular loss function for regression), and CrossEntropy (popular for binary classification)

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

# A neural net is composed of layers (ideally alternating linear layers and activation functions).
# There is an abstract layer base class and concrete classes for the linear layer and two popular
# activation functions; ReLU (a popular activation between layers), and Sigmoid (popular on the
# output layer for binary classification networks)

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

# The NeuralNet class is the main class for the neural network. It is initialized with a loss
# function and a set of layers. The forward and backward methods are used to train the network,
# and the predict method is used to make predictions on new data.
# Auto_train is a convenience method that will train the network until the loss function is
# below a given threshold

class NeuralNet:
    """Neural network class"""

    def __init__(self, loss_function = MSE(), seed = 0):
        self.debug_trace = False
        self.layers = []
        self.loss_function = loss_function
        self.loss_per_epoch = None
        self.learning_rate = 0.05   # Initial learning rate
        self.decay = .99            # For auto_train's exponential avg of learning_rate
        self.max_epochs = 50000     # Auto_train terminates if loss goal not reached by then.
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

    def auto_train(self, x_train: np.ndarray, y_train: np.ndarray,  # pylint: disable=too-many-arguments, too-many-locals
                   x_valid: np.ndarray, y_valid: np.ndarray, target_loss : int) \
                   -> bool:
        """Train the neural network without any hyper parameters"""
        self.loss_per_epoch = []

        # Defaults
        batch_size = x_train.shape[1]
        if batch_size > 32:
            batch_size = 16
        learning_rate = self.learning_rate  # Initial learning rate; decays with exponential average

        # Start training
        while True:
            # Log the loss for this epoch.
            y = self.predict(x_valid)
            loss = self.loss(y, y_valid)
            self.loss_per_epoch.append (loss)

            # Stop training if we hit the target_loss, or if we've completed max_epochs
            if loss < target_loss:
                return True
            if len(self.loss_per_epoch) > self.max_epochs:
                return False

            # Run training over mini-batches of batch_size; shffle training data each epoch
            # to avoid overfitting caused by using the same mini-batches each epoch
            permutation = np.random.permutation(x_train.shape[1])
            x_train = x_train[:, permutation]
            y_train = y_train[:, permutation]
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
                learning_rate = learning_rate * self.decay + last_learning_rate * (1 - self.decay)

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
                self.trace (f"Weights and biases for linear layer {layer_num}:")
                self.trace (layer.w, "\n", layer.b)
            layer_num += 1

# Testing
np.random.seed(13)

def train_and_test (nn : NeuralNet, test_name : str, x_train : np.ndarray, y_train : np.ndarray, # pylint: disable=too-many-arguments
                    x_valid : np.ndarray, y_valid : np.ndarray, target_loss : int = .05): 
    """Train the nn and validate if it meets the target loss"""
    # Regular training is much less efficient and lots of trial/error to get the right learning rate
    #nn.learning_rate = .001
    #nn.train(x_train, y_train, 5000, 10)
    #y = nn.predict(x_valid)
    #loss = nn.loss(y, y_valid) * 10000 // 1 / 10000

    nn.auto_train(x_train, y_train, x_valid, y_valid, target_loss)
    loss = nn.loss_per_epoch[-1] * 10000 // 1 / 10000

    summary = "Average "
    if isinstance (nn.loss_function, MSE):
        summary += "MSE "
    elif isinstance (nn.loss_function, CrossEntropy):
        summary += "CrossEntropy "
    summary += "loss is " + str(loss) + " after " + str(len(nn.loss_per_epoch)) + " epochs"

    if math.isnan(loss) or loss > target_loss:
        print(f"FAIL: {test_name}. {summary}")
        nn.plot_loss()
        nn.print_weights()
    else:
        print(f"PASS: {test_name}. {summary}")

def test_1_variable_identity():
    """1 variable identity (1 input, 1 output)"""
    test_name = "1 neuron, 1 input identity"
    nn = NeuralNet()
    nn.add_layer(Linear(1, 1))

    # Training set: 100 records; each col of x is a random ints between 0 and 100, y = x
    x_train = np.random.randint(100, size=(1, 100))
    y_train = x_train

    # Validation set: 20 such records
    x_valid = np.random.randint(100, size=(1, 20)) - 50
    y_valid = x_valid

    train_and_test (nn, test_name, x_train, y_train, x_valid, y_valid)

def test_1_layer_sum():
    """1 layer sum (2 inputs, 1 output)"""
    test_name = "1 neuron, 2 input sum"
    nn = NeuralNet()
    nn.add_layer(Linear(2, 1))

    # Training set: 100 records; each col of x is two random ints between 0 and 20, y = sum
    x_train = np.random.randint(20, size=(2, 100))
    y_train = x_train.sum(axis=0, keepdims=True)

    # Validation set: 20 such records
    x_valid = np.random.randint(20, size=(2, 20))
    y_valid = x_valid.sum(axis=0, keepdims=True)

    train_and_test (nn, test_name, x_train, y_train, x_valid, y_valid)

def test_2_layer_2_variable_identity():
    """2d identity test with 2 layer"""
    test_name = "2 layer, 2 input identity"
    nn = NeuralNet()
    nn.add_layer(Linear(2, 3))
    nn.add_layer(Linear(3, 2))

    # Training set: 100 records; each col of x are two random ints between 0 and 100, y = x
    x_train = np.random.randint(100, size=(2, 100))
    y_train = x_train

    # Validation set: 20 such records
    x_valid = np.random.randint(100, size=(2, 20))
    y_valid = x_valid

    train_and_test (nn, test_name, x_train, y_train, x_valid, y_valid)

def test_relu_actication():
    """ReLU activation function test"""
    test_name = "2 layers + ReLU activation, 2 input diff"
    nn = NeuralNet()
    nn.add_layer(Linear(2, 6))
    nn.add_layer(ReLU())
    nn.add_layer(Linear(6, 1))

    # Training set: 1000 records; each col of x are two random ints between -50 and 50, y = x
    x_train = np.random.randint(100, size=(2, 1000)) - 50
    y_train = np.max(x_train, axis=0, keepdims=True) - np.min(x_train, axis=0, keepdims=True)

    # Validation set: 100 such records with x between -10 and 10
    x_valid = np.random.randint(20, size=(2, 100)) - 10
    y_valid = np.max(x_valid, axis=0, keepdims=True) - np.min(x_valid, axis=0, keepdims=True)

    train_and_test (nn, test_name, x_train, y_train, x_valid, y_valid)

def test_simple_classification():
    """Simple classification test - if number above/below threshold"""
    test_name = "1 layer, 1 input classification if input over threshold"
    nn = NeuralNet(CrossEntropy())
    nn.add_layer(Linear(1, 1))
    nn.add_layer(Sigmoid())

    # Training set: 1000 records; each col of x is a random ints between 0 and 20, y = x > 10
    threshold = 10
    x_train = np.random.randint(2 * threshold, size=(1, 1000))
    y_train = (1 * (x_train.sum(axis=0, keepdims=True) > threshold)).reshape(1, 1000)

    # Validation set: 100 such records
    x_valid = np.random.randint(2 * threshold, size=(1, 100))
    y_valid = (1 * (x_valid.sum(axis=0, keepdims=True) > threshold)).reshape(1, 100)

    train_and_test (nn, test_name, x_train, y_train, x_valid, y_valid, .05)

def test_classification():
    """Test network w 2 linear layers and 2 activation functions for classification"""
    test_name = "2 layer, 2 input classification if sum over threshold"
    nn = NeuralNet(CrossEntropy())
    nn.add_layer(Linear(2, 6))
    nn.add_layer(ReLU())
    nn.add_layer(Linear(6, 1))
    nn.add_layer(Sigmoid())

    # Training set: 1000 records; each col of x are two random ints between -40 and 40
    # y = if the sum of the ints is > 20
    threshold = 20
    x_train = np.random.randint(4 * threshold, size=(2, 1000)) - 2 * threshold
    y_train = (1 * (x_train.sum(axis=0, keepdims=True) > threshold)).reshape(1, 1000)

    # Validation set: 200 such records
    x_valid = np.random.randint(4 * threshold, size=(2, 200)) - 2 * threshold
    y_valid = (1 * (x_valid.sum(axis=0, keepdims=True) > threshold)).reshape(1, 200)

    train_and_test (nn, test_name, x_train, y_train, x_valid, y_valid, .05)

def test():
    """Test the neural network"""
    test_1_variable_identity()
    test_1_layer_sum()
    test_2_layer_2_variable_identity()
    test_relu_actication()
    test_simple_classification()
    test_classification()

test()
