"""
backprop_tests.py - test backprop.py

This library tests backprop.py and also shows how to use that library.
"""

import math
import numpy as np
from backprop import NeuralNet, Linear, Sigmoid, ReLU, CrossEntropy, MSE

# Disable some pylint warnings; I like to use x, y, dx, dy, nn as variable names in this file
# pylint: disable=invalid-name

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
