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
* Current CNN accuracy caps at ~91% test, 98% test w 100 epochs, Adam, weight_decay=2e-5
* A simpler CNN also did 91% test accuracy (but only 95% train)

TODO:
Improve CNN to get test accuracy over 95% (current issue is overfitting)
Things that may help:
* Add dev loss to the train loss plot
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
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchmetrics.classification import MulticlassConfusionMatrix

# Disable some pylint warnings; I like to use x, y, dx, dy, nn as variable names in this file
# pylint: disable=invalid-name
# Pylint can't seem to find a number of torch members, so disable that warning
# pylint: disable=no-member

# Fix random seeds for reproducibility
torch.manual_seed(12)
if torch.cuda.is_available():
    torch.cuda.manual_seed(12)
    torch.backends.cudnn.deterministic=True

# Enable tracing by default
trace_callback = print
def trace (*args):
    """Trace callback"""
    trace_callback(*args)

# Configure device to use for tensors (CPU or GPU).
# If your PC has a GPU, you'll need to perform these steps to allow PyTorch to use it:
# 1) Install CUDA: https://developer.nvidia.com/cuda-downloads
# 2) Make sure your installed PyTorch is compiled for cuda. https://pytorch.org/get-started/locally/
#    As of this writing, pytorch doesn't run on Python 3.11; so I also had to downgrade to 3.10
trace(f'PyTorch version: {torch.__version__}')
device = "cuda" if torch.cuda.is_available() else "cpu"
trace(f"Using {device} device")
if torch.cuda.is_available():
    trace(f'Device Name: {torch.cuda.get_device_name()}')

# Class to hold generic PyTorch training and eval logic for various configurations
# Currently only the network layers and weight decay are configurable;
# The optimizer (Adam) and loss function (CrossEntropy) are hard-coded
class FashionMnist(nn.Module):
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

    SAVE_DIR = "saved_models"

    def __init__(self, model_name : str, model : torch.nn.Module, weight_decay : float = 2e-5, \
                 batch_size : int = 128):
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay)
        self.loss_per_epoch = []
        self.trace_debug = True
        self.batch_size = batch_size
        self.model_name = model_name
        self.train_loader = None
        self.dev_loader = None
        self.to(device)

    @property
    def SAVE_FILE(self):
        """Return the save file name"""
        return os.path.join(self.SAVE_DIR, "FashionMnist_" + self.model_name + ".pt")

    def forward(self, x):
        """Forward pass"""
        logits = self.model(x)
        return logits

    def train_epochs(self, epochs: int):
        """Train the model"""
        dev, train = self.get_dev_train_dataloaders(self.batch_size)
        self.loss_per_epoch = []

        for e in range(epochs):
            avg_loss = 0      # Avg loss on training data per this epoch
            avg_accuracy = 0  # Avg accuracy on training data per this epoch
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
                avg_loss += loss.item()
                # Adjust accuracy_this_epoch
                avg_accuracy += (pred.argmax(1) == y).type(torch.float).sum().item() # pylint: disable=no-member

            # Compute total loss for the epoch
            avg_loss /= len(train)
            # Compute total accuracy for the epoch
            avg_accuracy *= 100 / len(train.dataset)
            if self.trace_debug:
                trace (f"Epoch {e}:")
                print (f"\tAvg train loss: {avg_loss:.2f}, Accuracy: {avg_accuracy:.2f}")
                self.model.eval()
                dev_loss, dev_accuracy = self.test(dev)
                trace (f"\tDev loss      : {dev_loss:.2f}, Accuracy: {dev_accuracy:.2f}")
                self.model.train()
            self.loss_per_epoch.append(avg_loss)

    def test (self, data : DataLoader):
        """Test the model - similar to predict, but gather some stats"""
        self.model.eval()
        loss = 0
        correct = 0
        with torch.no_grad():
            for X, y in data:
                X, y = X.to(device), y.to(device)
                pred = self.model(X)
                loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item() # pylint: disable=no-member
        return loss / len(data), correct * 100 / len(data.dataset)

    def predict (self, x : torch.Tensor):
        """predicting the output for given input"""
        self.model.eval()
        with torch.no_grad():
            return self.model(x)

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

    # Get dev and train dataloaders; dev is 10,000 records from the training set
    # And used to validate the model on unseen data and tune hyperparameters
    def get_dev_train_dataloaders(self, train_batch_size : int = None):
        """Get the dev and train dataloaders"""
        if self.train_loader is None:
            #Download training data from open datasets.
            training_data = datasets.FashionMNIST(
                root="data",
                train=True,
                download=True,
                transform=ToTensor(),
            )
            dev_set_size = 5000 # Carve out some data for validation
            train_set_size = len(training_data) - dev_set_size # Use the rest for training
            dev_data, train_data = random_split(training_data, [dev_set_size, train_set_size])
            self.dev_loader = DataLoader(dev_data, batch_size=len(dev_data))
            if train_batch_size is None:
                train_batch_size = len(train_data)
            self.train_loader = DataLoader(train_data, batch_size=train_batch_size)
    
        return self.dev_loader, self.train_loader
    

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

    def show_images(self, images, labels):
        """Show a set of images and their correct vs predicted labels"""
        for x, y in zip(images, labels):
            x = x.unsqueeze(0)   # Add batch dimension to x
            with torch.no_grad():
                pred = self(x.to(device)).argmax()
                if pred == y:
                    continue
                label = "Predicted: " + self.CLASSES[pred] + ", Actual: " + self.CLASSES[y]
                plt.title(label)
                plt.imshow(x.squeeze(), cmap="gray")
                plt.pause(3)
                if not plt.fignum_exists(1):
                    break

    def eval(self):
        """Evaluate the model"""
        dev_data, train_data = self.get_dev_train_dataloaders(self.batch_size)
        test_data = self.get_test_dataloader()
        for name, data in [("Train", train_data), ("Dev", dev_data), ("Test", test_data)]:
            loss, accuracy = self.test(data)
            trace (f"{name} loss: {loss:.2f}, Accuracy = {accuracy:.2f}%")

        cm = self.get_confusion_matrix(train_data)
        trace ("Confusion matrix for train data")
        trace (f"Classes: {self.CLASSES}")
        trace (cm)

        if len(self.loss_per_epoch) > 0:
            trace ("Training loss per epoch...")
            self.plot_loss()

        test_images, test_labels = next(iter(test_data))
        self.show_images(test_images, test_labels)

    def get_confusion_matrix(self, data : DataLoader):
        """Get the confusion matrix for the given data"""
        y_actual = torch.tensor([], dtype=torch.long)
        y_pred = torch.tensor([], dtype=torch.long)
        self.model.eval()
        with torch.no_grad():
            for X, y in data:
                X = X.to(device)
                y_hat = self.model(X).to("cpu")
                y_actual = torch.cat((y_actual, y), 0)
                y_pred = torch.cat((y_pred, y_hat.argmax(1)), 0)
        cm_metric = MulticlassConfusionMatrix (len(self.CLASSES))
        return cm_metric(y_pred, y_actual)

    def plot_loss(self):
        """Show loss per epoch"""
        plt.plot(self.loss_per_epoch)
        plt.title("Loss per epoch", loc = "left")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()
        plt.pause(3)

    def print_model(self):
        """Print the model structure and parameters"""
        trace(f"Model structure: {self}\n\n")
        for name, param in self.named_parameters():
            trace(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


# Testing code
SAVE_MODEL = False  # Save model and load previously saved model (or start from scratch)?
EPOCHS = 5         # Number of epochs to train
NAME = "cnn"
BATCH_SIZE = 128

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


nn = FashionMnist(NAME, networks[NAME], batch_size=BATCH_SIZE)
print (f"Testing FashionMNIST {NAME} neural network")
if SAVE_MODEL and nn.load():
    print ("Loaded previously trained model:", nn.SAVE_FILE)

START = datetime.now()
print (f"Training for {EPOCHS} epochs, be patient...")
nn.train_epochs(EPOCHS)
print (f"Training took {datetime.now() - START}")
nn.eval()

if SAVE_MODEL:
    nn.save()
    print ("Saved model:", nn.SAVE_FILE)
