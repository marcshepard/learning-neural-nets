"""test.py - test neural_net.py"""

import numpy as np
from neural_net import NeuralNet, Linear, Sigmoid, ReLU

def test_1_variable_identity():
    """1 variable identity (1 input, 1 output)"""
    test_name = "1 neuron, 1 input identity"
    nn = NeuralNet()
    nn.add_layer(Linear(1, 1))
    x = np.random.randint(20, size=(1, 10))
    y = x
    nn.train(x, y, 100)
    x = [92]
    y = [92]
    y_hat = nn.predict(x)
    pct_error = abs((y[0] - y_hat[0][0]) * 1000 // y[0]) / 10
    if pct_error > 5:
        print(test_name, "with input =", x, ", expected output =", y, \
                ", actual output =", y_hat)
        print ("Weight/bias =", nn.layers[0].w, nn.layers[0].b)
        print("Test failed: error percent = ", pct_error, ">.1")
    else:
        print (test_name, "passed with", pct_error, "percent error")

    return nn.loss_per_epoch

def test_1_layer_sum():
    """1 layer sum (2 inputs, 1 output)"""
    test_name = "1 neuron, 2 input sum"
    nn = NeuralNet()
    nn.add_layer(Linear(2, 1))
    x = np.random.randint(20, size=(2, 10))
    y = x.sum(axis=0, keepdims=True)
    nn.train(x, y, 100)
    x = np.ndarray((2, 1))
    x[0][0] = 92
    x[1][0] = 17
    y = [109]
    y_hat = nn.predict(x)
    pct_error = abs((y[0] - y_hat[0][0]) * 1000 // y[0]) / 10
    if pct_error > 5:
        print(test_name, "with input = ", x, "expected output = ", y, \
                "actual output = ", y_hat)
        print("Test failed: error percent = ", pct_error, ">.1")
    else:
        print (test_name, "passed with", pct_error, "percent error")

def test_2_layer_2_variable_identity():
    """2d identity test with 2 layer"""
    test_name = "2 layer, 2 input identity"
    nn = NeuralNet()
    nn.add_layer(Linear(2, 3))
    nn.add_layer(Linear(3, 2))
    x = np.random.randint(20, size=(2, 10))
    y = x
    nn.train(x, y, 100)
    x = np.ndarray((2, 1))
    x[0][0] = 92
    x[1][0] = -17
    y = x
    y_hat = nn.predict(x)
    pct_error = np.sqrt(np.sum(np.square(y_hat - y))/np.sum(np.square(y)))*1000//1/10
    if pct_error > 5:
        print("Test failed:", test_name, "with input = ", x, "expected output = ", y, \
                "actual output = ", y_hat, "error percent = ", pct_error)
    else:
        print (test_name, "passed with", pct_error, "percent error")

def test_relu_actication():
    """ReLU activation function test"""
    test_name = "RelU activation function"
    nn = NeuralNet()
    nn.add_layer(Linear(2, 6))
    nn.add_layer(ReLU())
    nn.add_layer(Linear(6, 1))
    nn.add_layer(ReLU())

    x = np.random.randint(20, size=(2, 10))
    y = np.max(x, 0, keepdims=True) - np.min(x, 0, keepdims=True)
    nn.train(x, y, 100)

    x = np.ndarray((2, 1))
    x[0][0] = 14
    x[1][0] = 3
    y = [11]
    y_hat = nn.predict(x)
    pct_error = np.sqrt(np.sum(np.square(y_hat - y))/np.sum(np.square(y)))*1000//1/10
    if pct_error > 5:
        print("Test failed:", test_name, "with input = ", x, "expected output = ", y, \
                "actual output = ", y_hat, "error percent = ", pct_error)
    else:
        print (test_name, "passed with", pct_error, "percent error")

def test_classification():
    """Test network w 2 linear layers and 2 activation functions for classification"""
    test_name = "Classification test with 2 layers"
    nn = NeuralNet(CrossEntropy())
    nn.add_layer(Linear(2, 6))
    nn.add_layer(ReLU())
    nn.add_layer(Linear(6, 1))
    nn.add_layer(Sigmoid())

    x = np.random.randint(10, size=(2, 100))
    y = (x[0,:] == x[1,:]) * 1
    y.shape = (1, 100)

    nn.train(x, y, 100, 10)


def test():
    """Test the neural network"""
    test_1_variable_identity()
    test_1_layer_sum()
    test_2_layer_2_variable_identity()
    test_relu_actication()

test()
