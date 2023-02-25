"""test.py - test neural_net.py"""

import numpy as np
import matplotlib.pyplot as plt
from neural_net import NeuralNet, Linear, Sigmoid, ReLU, CrossEntropy

np.random.seed(12)

def show_loss(loss_per_epoch):
    """Show loss per epoch"""
    plt.plot(loss_per_epoch)
    plt.title("Loss per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

def test_1_variable_identity():
    """1 variable identity (1 input, 1 output)"""
    test_name = "1 neuron, 1 input identity"
    nn = NeuralNet()
    nn.add_layer(Linear(1, 1))
    x = np.random.randint(100, size=(1, 100))
    y = x
    nn.train(x, y, 500, 10)

    x = np.random.randint(100, size=(1, 10))
    y = x
    y_hat = nn.predict(x)
    loss = nn.loss_function.loss(y, y_hat)
    if loss > .1:
        print(f"{test_name} FAILED with loss = {loss}, input = {x}", \
              "output =", y_hat, ", weight/bias = ", nn.layers[0].w, nn.layers[0].b)
    else:
        print (f"{test_name} passed with average MSE loss {loss}")

    return nn.loss_per_epoch

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
    loss = nn.loss_function.loss(y, y_hat)
    if loss > .1:
        print(f"{test_name} FAILED with loss = {loss}, input = {x}", \
              "output =", y_hat, ", weight/bias = ", nn.layers[0].w, nn.layers[0].b)
    else:
        print (f"{test_name} passed with average MSE loss {loss}")


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
    loss = nn.loss_function.loss(y, y_hat)
    if loss > .1:
        print(f"{test_name} FAILED with loss = {loss}, input = {x}", \
              "output =", y_hat)
    else:
        print (f"{test_name} passed with average MSE loss {loss}")

def test_relu_actication():
    """ReLU activation function test"""
    test_name = "RelU activation function"
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
    loss = nn.loss_function.loss(y, y_hat)
    if loss > .1:
        print(f"{test_name} FAILED with loss = {loss}")
        print(f"{nn.loss_per_epoch[-1]} was the loss after the last epoch")
        show_loss(nn.loss_per_epoch)
    else:
        print (f"{test_name} passed with average MSE loss {loss}")

def test_classification():
    """Test network w 2 linear layers and 2 activation functions for classification"""
    test_name = "Classification test with 2 layers"
    nn = NeuralNet(CrossEntropy())
    nn.add_layer(Linear(2, 6))
    nn.add_layer(ReLU())
    nn.add_layer(Linear(6, 1))
    nn.add_layer(Sigmoid())

    x = np.random.randint(10, size=(2, 100))
    y = (x.sum(axis=0, keepdims=True) % 2).reshape(1, 100)

    nn.train(x, y, 500, 10)
    print (test_name, "test was run but not checked for correctness")
    show_loss(nn.loss_per_epoch)

def test():
    """Test the neural network"""
    test_1_variable_identity()
    test_1_layer_sum()
    test_2_layer_2_variable_identity()
    test_relu_actication()
    test_classification()

test()
