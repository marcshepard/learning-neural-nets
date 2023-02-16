"""
back_prop.py - test out building backprop from basic principals

Assume:
W = matrix whose rows are weights for a single layer neural net
    size = NUM_OUTPUTS x NUM_INPUTS
X = matrix of training data; each column is an input vector
    size = NUM_INPUTS x NUM_RECORDS
Y = matrix of expected training outputs; each column is an expected output
    size = NUM_OUTPUTS x NUM_RECORDS
"""
import numpy as np

NUM_INPUTS = 3  # Size of input vector
NUM_OUTPUTS = 2 # Size of output vector
NUM_RECORDS = 5 # Number of (input, output) training records

# Start with random weights
W = np.random.rand(NUM_OUTPUTS, NUM_INPUTS)
learning_rate = .05

# Training data; all 1's
X = np.full((NUM_INPUTS, NUM_RECORDS), 1)
Y = np.full((NUM_OUTPUTS, NUM_RECORDS), 1)

# Calculate loss
def loss (W, X, Y):
    """Compute the MSE loss of f(X) = W*X for expected values Y"""
    diff = np.matmul(W, X) - Y          # Compute f(x) - Y
    squared = np.multiply(diff, diff)   # Square each element
    dist = squared.sum(axis=0)          # Sum columns for square error of each record
    avg_err = dist.sum()/NUM_RECORDS    # Average the errors
    return avg_err

# Calculate gradient
def gradient (W, X, Y):
    """Compute the gradient of the MSE loss function"""
    return np.matmul((np.matmul (W, X) - Y), X.transpose())

print ("Initial loss =", loss(W, X, Y))

for _ in range(5):
    W = W - learning_rate * gradient(W, X, Y)
    print ("\nAfter training, loss =", loss(W, X, Y))
    print (np.matmul (W, X) - Y)
    #print (W)
