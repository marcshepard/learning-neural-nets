"""
vision.py - neural network code for image recognition built on pytorch (so I could learn)

Prereqs: matplotlib, torch, torchdata

Goals and design:
Initially based on https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html,
which constructs a basic image classifier for the FashionMNISTA dataset with poor performance,
I've since started optimizations, adding CNNs and incorporating ideas from Andrew Ng's Deep Learning
series on Coursera; it's a work in progress.

Learnings from testing so far w NVIDIA 2060 GPU, 16GB RAM, 3GHZ 8 core i7-10700F CPU:
* Adam optimizer is much faster than SGD.
* GPU was much faster on deeper networks, but similar speeds on the simple networks.
* Simple cnn accuracy caps at ~86% test, 89% test
* Cnn accuracy caps at ~92% test, 98% test w 100 epochs, Adam, weight_decay=2e-5

TODO:
Improve CNN to get test accuracy over 95% (current issue is overfitting)
Things that may help:
* Modify eval to show convolution matrix, pop pictures of biggest errors
* Break train data into train and dev; shouldn't use test data for eval
* Log various training runs
* Try it in TensorFlow to compare/contrast.

Some notes: In Pytorch, each row of input is a record (vs backprop.py which uses colums).
Each row of weights is a neuron (same as backprop.py).
so math for linear layer l(x) = x * w.T + b (vs neural_net.py's l(x) = w * x + b).
It's weight initialization is rand(-sqrt(k), sqrt(k)) where k = 1/# inputs
"""

from datetime import datetime
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Disable some pylint warnings; I like to use x, y, dx, dy, nn as variable names in this file
# pylint: disable=invalid-name

# Configure device to use for tensors (CPU or GPU).
# If your PC has a GPU, you'll need to perform these steps to allow PyTorch to use it:
# 1) Install CUDA: https://developer.nvidia.com/cuda-downloads
# 2) Make sure your installed PyTorch is compiled for cuda. https://pytorch.org/get-started/locally/
#    As of this writing, pytorch doesn't run on Python 3.11; so I also had to downgrade to 3.10

trace_callback = print
def trace (*args):
    """Trace callback"""
    trace_callback(*args)

trace(f'PyTorch version: {torch.__version__}')
device = "cuda" if torch.cuda.is_available() else "cpu"
trace(f"Using {device} device")
if device == "cuda":
    trace(f'Device Name: {torch.cuda.get_device_name()}')

# From https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
# Base class for neural network models
class NeuralNetwork(nn.Module):
    """Added some common training and testing code for reuse"""
    def __init__(self, model : torch.nn.Module, loss_fn : torch.nn.modules.loss._Loss):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(model.parameters(), weight_decay=2e-5)
        self.loss_per_epoch = []
        self.trace_debug = True
        self.to(device)

    def forward(self, x):
        """Forward pass"""
        logits = self.model(x)
        return logits

    def auto_train (self, train : DataLoader, test : DataLoader, epochs : int):
        """Train the model"""
        self.loss_per_epoch = []

        for e in range(epochs):
            loss_this_epoch = 0
            accuracy_this_epoch = 0
            self.model.train()
            for X, y in train:
                X, y = X.to(device), y.to(device)

                # Compute prediction error
                pred = self.model(X)
                loss = self.loss_fn(pred, y)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Adjust loss_this_epoch
                loss_this_epoch += loss.item()
                # Adjust accuracy_this_epoch
                accuracy_this_epoch += (pred.argmax(1) == y).type(torch.float).sum().item() # pylint: disable=no-member
        
            # Compute total loss for the epoch
            loss_this_epoch /= len(train)
            # Compute total accuracy for the epoch
            accuracy_this_epoch *= 100 / len(train.dataset)
            if self.trace_debug:
                trace (f"Epoch: {e}. Avg mini-batch data loss: {loss_this_epoch:.2f}, Accuracy: {accuracy_this_epoch:.2f}")
                self.model.eval()
                test_loss, test_accuracy = self.test(test)
                trace (f"\tTest data loss: {test_loss:.2f}, Accuracy: {test_accuracy:.2f}")
                self.model.train()
            self.loss_per_epoch.append(loss_this_epoch)

    def test (self, test : DataLoader):
        """Test the model - similar to predict, but gather some stats"""
        self.model.eval()
        loss = 0
        correct = 0
        with torch.no_grad():
            for X, y in test:
                X, y = X.to(device), y.to(device)
                pred = self.model(X)
                loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item() # pylint: disable=no-member
        return loss / len(test), correct * 100 / len(test.dataset)

    def predict (self, x : torch.Tensor):
        """predicting the output for given input"""
        self.model.eval()
        with torch.no_grad():
            return self.model(x)

class FashionMnist(NeuralNetwork):
    """Neural network for the FashionMNIST dataset"""
    CLASSES = [
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

    def __init__(self, model : torch.nn.Module, save_file : str):
        super().__init__(model, nn.CrossEntropyLoss())
        self.SAVE_DIR = "models"
        self.SAVE_FILE = os.path.join(self.SAVE_DIR, save_file)

    @staticmethod
    def get_test_dataloader ():
        """Download test data from open datasets."""
        test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
        )
        return DataLoader(test_data, batch_size=len(test_data))

    @staticmethod
    def get_train_dataloader():
        """Download training data from open datasets."""
        training_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )
        return DataLoader(training_data, batch_size=64)

    def save(self):
        """Save the model to a file"""
        if not os.path.exists(self.SAVE_DIR):
            os.makedirs(self.SAVE_DIR)
        torch.save(self.state_dict(), self.SAVE_FILE)

    def load(self) -> bool:
        """Load the model from a file. Returns True if successful, False if not"""
        try:
            self.load_state_dict(torch.load(self.SAVE_FILE))
            return True
        except (RuntimeError, FileNotFoundError):
            return False

    def train_epochs(self, epochs: int):
        """Train the model"""
        return super().auto_train(self.get_train_dataloader(), self.get_test_dataloader(), epochs)

    def show_images(self, images, labels):
        """Show a set of images and their correct vs predicted labels"""
        for x, y in zip(images, labels):
            x = x.unsqueeze(0)   # Add batch dimension to x
            with torch.no_grad():
                pred = self(x.to(device))
                label = "Predicted: " + self.CLASSES[pred.argmax()] + ", Actual: " + self.CLASSES[y]
                plt.title(label)
                plt.imshow(x.squeeze(), cmap="gray")
                plt.pause(3)
                if not plt.fignum_exists(1):
                    break

    def show_image(self, img, label):
        """Show a single image and label"""
        plt.figure(figsize=(2, 2))
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
        plt.show(block=False)

    def eval(self):
        """Evaluate the model"""
        test_data = self.get_test_dataloader()
        loss, accuracy = self.test(test_data)
        trace (f"Loss: {loss}, Accuracy = {accuracy}%")
        if len(self.loss_per_epoch) > 0:
            self.plot_loss()

        test_images, test_labels = next(iter(test_data))
        self.show_images(test_images, test_labels)

    def plot_loss(self):
        """Show loss per epoch"""
        plt.plot(self.loss_per_epoch)
        plt.title("Loss per epoch", loc = "left")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    def print_model(self):
        """Print the model structure and parameters"""
        trace(f"Model structure: {self}\n\n")
        for name, param in self.named_parameters():
            trace(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

SAVE_MODEL = False  # Save model and load previously saved model (or start from scratch)?
EPOCHS = 99         # Number of epochs to train

networks = {
    # Accuracy: 89, 86
    "simple" : nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10),
        nn.Softmax(dim=1)),
    # Accuracy: 95, 91
    "simple_cnn" : nn.Sequential(           # Input size:  1x28x28
        nn.Conv2d(1, 6, kernel_size=3),     # Output: 6x26x26; 26=28-3+1
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),        # Output: 6x13x13; 13=26/2
        nn.Conv2d(6, 12, kernel_size=3),    # Output: 12x11x11; 11=13-3+1
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),        # Output: 12x5x5; 5=11/2
        nn.Flatten(),
        nn.Linear(12*5*5, 10),
        nn.Softmax(dim=1)),
    # Accuracy: 98, 91 after 100 epochs w weight_decay 2e-5
    "cnn" : nn.Sequential(                  # Input size:  1x28x28
        nn.Conv2d(1, 32, kernel_size=3),    # Output: 32x26x26; 26=28-3+1
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),        # Output: 32x13x13; 13=26/2
        nn.Conv2d(32, 64, kernel_size=3),   # Output: 64x11x11; 11=13-3+1
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),        # Output: 64x5x5; 5=11/2
        nn.Flatten(),
        nn.BatchNorm1d(64*5*5),
        nn.Linear(64*5*5, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
        nn.Softmax(dim=1)),
    # Accuracy: ???
    "cnn_last" : nn.Sequential(             # Input size:  1x28x28
        nn.Conv2d(1, 32, kernel_size=3),    # Output: 32x26x26; 26=28-3+1
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),        # Output: 32x13x13; 13=26/2
        nn.Conv2d(32, 64, kernel_size=3),   # Output: 64x11x11; 11=13-3+1
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),        # Output: 64x5x5; 5=11/2
        nn.Flatten(),
        nn.BatchNorm1d(64*5*5),
        nn.Linear(64*5*5, 256),
        nn.ReLU(),
        nn.Dropout1d(0.05),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
        nn.Softmax(dim=1)),
}


NAME = "cnn"

nn = FashionMnist(networks[NAME], "fashion_mnist_" + NAME + ".pth")
print (f"Testing FashionMNIST {NAME} neural network")
if SAVE_MODEL and nn.load():
    print ("Loaded previously trained model:", nn.SAVE_FILE)

START = datetime.now()
print (f"Training for {EPOCHS} epochs, be patient...")
nn.train_epochs(EPOCHS)
print (f"Training took {datetime.now() - START}")

if SAVE_MODEL:
    nn.save()
    print ("Saved model:", nn.SAVE_FILE)
