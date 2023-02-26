"""test.py - test neural_net.py"""

import math
import numpy as np
from neural_net import NeuralNet, Linear, Sigmoid, ReLU, CrossEntropy

np.random.seed(13)

def test_1_variable_identity():
    """1 variable identity (1 input, 1 output)"""
    test_name = "1 neuron, 1 input identity"
    nn = NeuralNet()
    nn.add_layer(Linear(1, 1))

    x = np.random.randint(100, size=(1, 100))
    y = x
    nn.train(x, y, 500, 10)

    x = np.random.randint(100, size=(1, 10)) - 50
    y = x
    y_hat = nn.predict(x)
    loss = nn.loss(y_hat, y)

    if math.isnan(loss) or loss > .1:
        print(f"FAIL: {test_name} with loss = {loss}, input = {x}", \
              "output =", y_hat, ", weight/bias = ", nn.layers[0].w, nn.layers[0].b)
        nn.print_weights()
        nn.plot_loss()
    else:
        print (f"PASS: {test_name} with average MSE loss {loss}")

def test_1_layer_sum():
    """1 layer sum (2 inputs, 1 output)"""
    test_name = "1 neuron, 2 input sum"
    nn = NeuralNet()
    nn.add_layer(Linear(2, 1))

    x = np.random.randint(20, size=(2, 100))
    y = x.sum(axis=0, keepdims=True)
    nn.train(x, y, 500, 10)

    x = np.random.randint(20, size=(2, 10))
    y = x.sum(axis=0, keepdims=True)
    y_hat = nn.predict(x)
    loss = nn.loss(y_hat, y)

    if math.isnan(loss) or loss > .1:
        print (f"FAIL: {test_name} with loss = {loss}, input = {x}", \
              "output =", y_hat, ", weight/bias = ", nn.layers[0].w, nn.layers[0].b)
        nn.print_weights()
        nn.plot_loss()
    else:
        print (f"PASS: {test_name} average MSE loss {loss}")


def test_2_layer_2_variable_identity():
    """2d identity test with 2 layer"""
    test_name = "2 layer, 2 input identity"
    nn = NeuralNet()
    nn.add_layer(Linear(2, 3))
    nn.add_layer(Linear(3, 2))

    x = np.random.randint(20, size=(2, 100))
    y = x
    nn.train(x, y, 500, 10)

    x = np.random.randint(20, size=(2, 10))
    y = x
    y_hat = nn.predict(x)
    loss = nn.loss(y_hat, y)

    if math.isnan(loss) or loss > .1:
        print(f"FAIL: {test_name} with loss = {loss}, input = {x}", \
              "output =", y_hat)
        nn.print_weights()
        nn.plot_loss()
    else:
        print (f"PASS: {test_name} with average MSE loss {loss}")


def test_relu_actication():
    """ReLU activation function test"""
    test_name = "2 layers + ReLU activation, 2 input diff"
    nn = NeuralNet()
    nn.add_layer(Linear(2, 6))
    nn.add_layer(ReLU())
    nn.add_layer(Linear(6, 1))

    x = np.random.randint(100, size=(2, 100))
    y = np.max(x, axis=0, keepdims=True) - np.min(x, axis=0, keepdims=True)
    nn.train(x, y, 500, 20)

    x = np.random.randint(20, size=(2, 10))
    y = np.max(x, axis=0, keepdims=True) - np.min(x, axis=0, keepdims=True)
    y_hat = nn.predict(x)
    loss = nn.loss(y_hat, y)

    if math.isnan(loss) or loss > .1:
        print(f"FAIL {test_name} with loss = {loss}")
        nn.print_weights()
        nn.plot_loss()
    else:
        print (f"PASS: {test_name} with average MSE loss {loss}")

def test_simple_classification():
    """Simple classification test - if number above/below threshold"""
    test_name = "Classification input is over threshold"
    nn = NeuralNet(CrossEntropy())
    nn.add_layer(Linear(1, 1))
    nn.add_layer(Sigmoid())

    threshold = 10
    x = np.random.randint(2 * threshold, size=(1, 100))
    y = (1 * (x.sum(axis=0, keepdims=True) > threshold)).reshape(1, 100)
    nn.train(x, y, 10000, 10)

    x = np.random.randint(2 * threshold, size=(1, 10))
    y = (1 * (x.sum(axis=0, keepdims=True) > threshold)).reshape(1, 10)

    y_hat = nn.predict(x)
    loss = nn.loss(y_hat, y)
    if math.isnan(loss) or loss > .1:
        print(f"FAIL: {test_name} with CrossEntropy loss = {loss}")
        print ("predicted value of 0:", nn.predict([[0]]))
        print (f"predicted value of threshold {threshold}:", nn.predict([[threshold]]))
        print (f"predicted value of {2*threshold}:", nn.predict([[2*threshold]]))
        nn.print_weights()
        nn.plot_loss()
    else:
        print (f"PASS: {test_name} passed with average CrossEntropy loss {loss}")


def test_classification():
    """Test network w 2 linear layers and 2 activation functions for classification"""
    test_name = "Classification if sum of inuts over a threshold, 2 layers"
    nn = NeuralNet(CrossEntropy())
    nn.add_layer(Linear(2, 6))
    nn.add_layer(ReLU())
    nn.add_layer(Linear(6, 1))
    nn.add_layer(Sigmoid())

    threshold = 20
    x = np.random.randint(threshold, size=(2, 100))
    y = (1 * (x.sum(axis=0, keepdims=True) > threshold)).reshape(1, 100)
    nn.train(x, y, 4000, 10)

    x = np.random.randint(2 * threshold, size=(2, 10))
    y = (1 * (x.sum(axis=0, keepdims=True) > threshold)).reshape(1, 10)
    y_hat = nn.predict(x)
    loss = nn.loss(y_hat, y)

    if math.isnan(loss) or loss > .1:
        print(f"FAIL: {test_name} with CrossEntropy loss = {loss}")
        nn.print_weights()
        nn.plot_loss()
    else:
        print (f"PASS: {test_name} passed with average CrossEntropy loss {loss}")

def test():
    """Test the neural network"""
    test_1_variable_identity()
    test_1_layer_sum()
    test_2_layer_2_variable_identity()
    test_relu_actication()
    test_simple_classification()
    test_classification()

test()
