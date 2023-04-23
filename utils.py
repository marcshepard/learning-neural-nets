"""Utility for ML projects"""

# pylint: disable=invalid-name, too-many-arguments, too-many-instance-attributes, line-too-long, import-error, too-few-public-methods

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import Module
from tensorflow.keras.callbacks import History

def load_normed_data(data_set : Module, random_seed : int = 0):
    """ Load the tf data sets, splitting off validation data from training data
    equal in size to the test data, and scaling training pixels to be between 0 and 1"""

    train, (x_test, y_test) = data_set.load_data()
    x_train, x_val, y_train, y_val = train_test_split(train[0], train[1], test_size=len(x_test),
                                                    random_state=random_seed)
    x_train = x_train / 255.0
    x_val = x_val / 255.0
    x_test = x_test / 255.0
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

# A class for viewing images in a data set
class ImageViewer:
    """Display images from a data set a grid at a time, with next and prev buttons
    Optionally provide mislabled predictions to display as well"""

    def __init__ (self, x : np.ndarray, y : np.ndarray, label_names : str, rows : int = 5, cols : int = 5, \
                  figsize=(10,10), title : str = "Images"):
        """x - the data set, y - the labels, label_names - the names of the labels, rows, cols = grid to display at a time
          figsize - the size of each image in the grid"""
        self.ind = 0
        self.x = x
        self.y = y
        self.label_names = label_names
        self.rows = rows
        self.cols = cols
        self.title = title

        plt.figure(figsize=figsize)
        plt.margins(0, 0)

        self._draw_images()

        next_btn = plt.axes([0.7, 0.1, 0.1, 0.075])
        prev_btn = plt.axes([0.2, 0.1, 0.1, 0.075])
        self.next_button = plt.Button(next_btn, 'Next')
        self.prev_button = plt.Button(prev_btn, 'Prev')
        self.next_button.on_clicked(self._next)
        self.prev_button.on_clicked(self._prev)

        plt.show()

    def _draw_images(self):
        """Draw rows*cols images from the data set, starting at ind"""
        start_ix = self.ind * self.rows * self.cols
        end_ix = start_ix + self.rows * self.cols
        plt.suptitle(f"{self.title} {start_ix} - {end_ix}")

        for i in range(self.rows * self.cols):
            plt.subplot(self.rows + 1, self.cols, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(self.x[self.ind + i], cmap=plt.cm.get_cmap("binary"))
            plt.xlabel(self.label_names[self.y[self.ind + i]])
            plt.margins(0.1, 0.1)
        plt.draw()

    def _next(self, event):  # pylint: disable=unused-argument
        """Show the next set of images"""
        if self.ind * self.rows * self.cols < len(self.x):
            self.ind += 1
            self._draw_images()

    def _prev(self, event):  # pylint: disable=unused-argument
        """Show the previous set of images"""
        if self.ind > 0:
            self.ind -= 1
            self._draw_images()


def plot_training_history(history : History, figsize=(10,10)):
    """Plot loss and accuracy during training"""
    fig, (pa, pl) = plt.subplots(2, 1, figsize=figsize)
    plt.suptitle("Training History - accuracy and loss per epoch")

    pa.autoscale(enable=True, axis='y')
    pa.plot(history.history['accuracy'], label='Training Accuracy')
    pa.plot(history.history['val_accuracy'], label='Validation Accuracy')
    pa.legend(loc='lower right')
    pa.set_ylabel('Accuracy')
    pa.set_xlabel('Epoch')

    pl.autoscale(enable=True, axis='y')
    pl.plot(history.history['loss'], label='Training Loss')
    pl.plot(history.history['val_loss'], label='Validation Loss')
    pl.legend(loc='upper right')
    pl.set_ylabel('Loss')
    pl.set_xlabel('Epoch')

    plt.show()
