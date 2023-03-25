"""
Feedforward.py - learning to do a basic feedforward neural network using PyTorch
The code below is just a refactoring of what is in https://pytorch.org/tutorials/, which uses
a basic feedforward neural network to classify images from the FashionMNIST dataset.

Some notes: In Pytorch, each row of input is a record (vs neural_net.py which uses colums).
Each row of weights is a neuron (same as neural_net.py).
so math for linear layer l(x) = x * w.T + b (vs neural_net.py's l(x) = w * x + b).
It's weight initialization is rand(-sqrt(k), sqrt(k)) where k = 1/# inputs
"""

from datetime import datetime
from torch import nn
import vision

INITIAL_EPOCHS = 5  # Initial training
MORE_EPOCHS = 0     # Additional training

networks = {
    "simple" : {"nn": nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    ), "path": "fashion_mnist_simple_nn.pth"},
    "convolutional" : {"nn": nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 12, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(12*4*4, 10)
        ), "path": "fashion_mnist_cnn.pth"},
}

for name, network in networks.items():
    nn = vision.FashionMnist(network["nn"], network["path"])
    network_name = f"FashionMNIST {name} neural network"
    if not nn.load():
        start = datetime.now()
        print (f"Training {network_name} for {INITIAL_EPOCHS} epochs, be patient...")
        nn.train_epochs(INITIAL_EPOCHS)
        print (f"Training took {datetime.now() - start}")
        nn.save()
    # Optionally, do a bit more training
    if MORE_EPOCHS > 0:
        start = datetime.now()
        print (f"Training {network_name} for {MORE_EPOCHS} more epochs, be patient...")
        nn.train_epochs(MORE_EPOCHS)
        print (f"Training took {datetime.now() - start}")
        nn.save()
    print (f"Evaluating the {network_name} against the test data...")
    nn.eval()
