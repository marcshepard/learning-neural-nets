"""
Scatchpad for learning PyTorch

I'm working my way through https://pytorch.org/tutorials/.
Goal is to learn PyTorch and compare to my primitive neural_net.py library

Some notes: In Pytorch, each row of input is a record (vs neural_net.py which uses colums).
Each row of weights is a neuron (same as neural_net.py).
so math for linear layer l(x) = x * w.T + b (vs neural_net.py's l(x) = w * x + b).
It's weight initialization is rand(-sqrt(k), sqrt(k)) where k = 1/# inputs
"""

import torch

# Configure device to use for tensors (CPU or GPU).
# If your PC has a GPU, you'll need to perform these steps to allow PyTorch to use it:
# 1) Install CUDA: https://developer.nvidia.com/cuda-downloads
# 2) Make sure your installed PyTorch is compiled for cuda. https://pytorch.org/get-started/locally/
#    As of this writing, pytorch doesn't run on Python 3.11; so I also had to downgrade to 3.10
print(f'PyTorch version: {torch.__version__}')
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")
if device == "cuda":
    print(f'Device Name: {torch.cuda.get_device_name()}')

# From https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        # Looks like nn.Module's __str__ method enumerates all fields of type nn.Module
        self.hello = "hello"
        self.goodbye = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def train_FashionMNIST_model(model_path, model = None, epochs = 5):
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    batch_size = 64

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    if model is None:
        model = NeuralNetwork().to(device)
        print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

    torch.save(model.state_dict(), model_path)
    print("Saved PyTorch Model State", model_path)

import random
import matplotlib.pyplot as plt
def show_FashionMNIST_image(img, label):
    figure = plt.figure(figsize=(8, 8))
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
    plt.show(block = True)

def evaluate_FashionMNIST_model(path):
    model = NeuralNetwork()
    model.load_state_dict(torch.load(path))

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    model.eval()

    print(f"Model structure: {model}\n\n")
    #for name, param in model.named_parameters():
    #    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    rand_ix = random.randint(0, len(test_data))
    x, y = test_data[rand_ix][0], test_data[rand_ix][1]
    with torch.no_grad():
        pred = model(x)
        print(f'For test data at index {rand_ix}, Predicted: "{classes[pred.argmax()]}", Actual: "{classes[y]}"')
        show_FashionMNIST_image(x, classes[y])

    loss_fn = nn.CrossEntropyLoss()
    model = model.to(device)
    test(DataLoader(test_data, batch_size=len(test_data)), model, loss_fn)

import os
# Create and train the model if it doesn't exist
model_dir = "models"
model_file = os.path.join(model_dir, "fashion_mnist_model.pth")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(model_file):
    train_FashionMNIST_model(model_file)

#  Optional; more training
def more_training(epochs):
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(model_file))
    train_FashionMNIST_model(model_file, model, epochs)
#more_training(20)

evaluate_FashionMNIST_model(model_file)

