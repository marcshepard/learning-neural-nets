"""
Scatchpad for learning PyTorch

I'm working my way through https://pytorch.org/tutorials/.
Goal is to learn PyTorch and compare to my primitive neural_net.py library

Some notes: In Pytorch, each row of input is a record (vs neural_net.py which uses colums).
Each row of weights is a neuron (same as neural_net.py).
so math for linear layer l(x) = x * w.T + b (vs neural_net.py's l(x) = w * x + b).
It's weight initialization is rand(-sqrt(k), sqrt(k)) where k = 1/# inputs
"""

import random
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

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
# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, model : torch.nn.Module, loss_fn : torch.nn.modules.loss._Loss):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        self.loss_per_epoch = []
        self.trace_debug = True
        self.to(device)

    def forward(self, x):
        logits = self.model(x)
        return logits

    def train(self, train : DataLoader, test : DataLoader, epochs : int):
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
                print (f"Epoch: {e}, Loss: {loss}, Accuracy: {accuracy}%")
            self.loss_per_epoch.append(loss)

    def test (self, test : DataLoader):
        self.model.eval()
        loss = 0
        correct = 0
        with torch.no_grad():
            for X, y in test:
                X, y = X.to(device), y.to(device)
                pred = self.model(X)
                loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        return loss / len(test), correct * 100 / len(test.dataset)
    
    def predict (self, x : torch.Tensor):
        self.model.eval()
        with torch.no_grad():
            return self.model(x)

class FashionMNIST(NeuralNetwork):
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

    def __init__(self):
        super().__init__(nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        ), nn.CrossEntropyLoss())
        self.SAVE_DIR = "models"
        self.SAVE_FILE = os.path.join(self.SAVE_DIR, "fashion_mnist_nn.pth")

    @staticmethod
    def get_test_dataloader ():
        # Download test data from open datasets.
        test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
        )
        return DataLoader(test_data, batch_size=len(test_data))
    
    @staticmethod
    def get_train_dataloader():
        # Download training data from open datasets.
        training_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )
        return DataLoader(training_data, batch_size=64)

    def save(self):
        if not os.path.exists(self.SAVE_DIR):
            os.makedirs(self.SAVE_DIR)
        torch.save(self.state_dict(), self.SAVE_FILE)

    def load(self) -> bool:
        try:
            self.load_state_dict(torch.load(self.SAVE_FILE))
            return True
        except:
            return False

    def train(self, epochs: int):
        return super().train(self.get_train_dataloader(), self.get_test_dataloader(), epochs)

    def show_image(self, img, label):
        plt.figure(figsize=(8, 8))
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
        plt.show()

    def eval(self):
        print ("Evaluations the current FashionMNIST model against the test dataset.")
        test_data = self.get_test_dataloader()
        loss, accuracy = self.test(test_data)
        print (f"Loss: {loss}, Accuracy = {accuracy}%")
        if len(self.loss_per_epoch) > 0:
            self.plot_loss()

        test_images, test_labels = next(iter(test_data))
        while input("Type enter to show a prediction from a random index, q to quit: ") != "q":
            rand_ix = random.randint(0, len(test_images))
            x = test_images[rand_ix]
            y = test_labels[rand_ix]
            with torch.no_grad():
                pred = self(x.to(device))
                print(f'For test data at index {rand_ix}, Predicted: "{self.CLASSES[pred.argmax()]}", Actual: "{self.CLASSES[y]}"')
                self.show_image(x, self.CLASSES[y])

    def plot_loss(self):
        """Show loss per epoch"""
        plt.plot(self.loss_per_epoch)
        plt.title("Loss per epoch", loc = "left")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    def print_model(self):
        print(f"Model structure: {self}\n\n")
        for name, param in self.named_parameters():
            print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

nn = FashionMNIST()

# Try to load existing model, or create from scratch if a model wasn't saved
if not nn.load():
    epochs = 5
    print (f"Creating the FashionMNIST model for {epochs} epochs, please be patient...")
    nn.train(epochs)
    nn.save()

# Optionally, do a bit more training
if False:
    epochs = 50
    print (f"Training the FashionMNIST model for {epochs} more epochs, please be patient...")
    nn.train(epochs)
    nn.save()

# Evaluate the model
nn.eval()
