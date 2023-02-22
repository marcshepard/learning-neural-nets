"""
back_prop.py - test out building backprop from basic principals

A single layer neural net that a single layer neural net with no bias can be represented by a
matrix of weights for each neuron in columns, and input as a row vector. So can be expressed as:
    f(X) = X * W (matrix multiplication)
Note: X can either a row vector (single input), or a matrix or inputs (each row is an input vector)

On scratch paper, I worked out the gradient of the MSE loss function with respect to weights as:
    dW = (f(X) - Y) * X.transpose().

So then after each training batch, we should be able to adjust W by subtracing dW * learning_rate

Below is an implementation to verify that this is correct, tested in a very simple case

Assume:
w = matrix whose rows are weights for a single layer neural net
    size = NUM_OUTPUTS x NUM_INPUTS
x = matrix of training data; each column is an input vector
    size = NUM_INPUTS x NUM_RECORDS
y = matrix of expected training outputs; each column is an expected output
    size = NUM_OUTPUTS x NUM_RECORDS
"""
import numpy as np

NUM_INPUTS = 3          # Size of input vector
NUM_OUTPUTS = 2         # Size of output vector
NUM_RECORDS = 5         # Number of (input, output) training records
LEARNING_RATE = .05     # Learning rate

# Start with random weights
w = np.random.rand(NUM_OUTPUTS, NUM_INPUTS)

# Training data; all 1's
x = np.full((NUM_INPUTS, NUM_RECORDS), 1)
y = np.full((NUM_OUTPUTS, NUM_RECORDS), 1)

# Calculate loss
def loss ():
    """Compute the MSE loss of f(X) = W*X for expected values Y"""
    diff = np.matmul(w, x) - y          # Compute f(x) - Y
    squared = np.multiply(diff, diff)   # Square each element
    dist = squared.sum(axis=0)          # Sum columns for square error of each record
    avg_err = dist.sum()/NUM_RECORDS    # Average the errors
    return avg_err

# Calculate gradient
def gradient ():
    """Compute the gradient of the MSE loss function"""
    return np.matmul((np.matmul (w, x) - y), x.T)

print ("Initial loss =", loss())

for i in range(5):
    w = w - LEARNING_RATE * gradient()
    print ("After training, run", i, "loss =", loss())
