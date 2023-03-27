"""
vision.py - neural network code for image recognition built on pytorch (so I could learn)

Prereqs: numpy, matplotlib, pytorch, pytorchvision

How to use: See the test code in vision_tests.py

Goals and design:
This code is largely a refactoring of what is in https://pytorch.org/tutorials/, but put into
classes for easier reuse. It includes notes on how to configure CUDA for GPU support for faster
pytorch processing if you have an NVIDIA graphics card. The main classes are:
* NeuralNetwork - nn.model subclass that provides training and testing and code reuse
* FashionMnist - NeuralNetwork subclass for manipulating the FashionMNIST dataset

vision_test.py uses FashionMnist in a couple of ways:
1) After configuring it as a simple feedforward neural network with just linear and relu layers (and
   a final signmoid layer for classification). This path is the refactored version of what is in
   https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html.
2) After configuring it as a convolutional neural network with convolutions and pooling. This was
   simple to build reusing the refactored code above and I didn't use any particular tutorial.
"""

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
        self.optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
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

            loss, accuracy = self.test(test)
            if self.trace_debug:
                trace (f"Epoch: {e}, Loss: {loss}, Accuracy: {accuracy}%")
            self.loss_per_epoch.append(loss)

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
