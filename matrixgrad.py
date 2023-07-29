"""
matrixgrad.py - learning how neural network training works from basic principals

A matrix version of Andrej Karpathy's https://github.com/karpathy/micrograd, to learn how backprop
works from basic principles.

There are a few classes here:
* Matrix - wraps a numpy ndarray and supports autograd; derived from AP's engine.py
* Several neural network classes on top

See matrixgrad.ipynb for examples of how to use these classes.
"""

# pylint: disable=line-too-long, invalid-name, protected-access, too-few-public-methods, too-many-arguments, too-many-locals

from collections import OrderedDict
import numpy as np

class Matrix:
    """ A dumbed-down version of a tensor; wraps a 2d numpy array and supports autograd """
    def __init__(self, data, _children=(), _op=''):
        self.data = np.asarray(data, dtype=np.float32)        # Data being wrapped
        if len(self.data.shape) == 0:   # scalar
            self.data = self.data.reshape((1,1))
        elif len(self.data.shape) == 1: # row vector
            self.data = self.data.reshape((1, self.data.shape[0]))
        assert len(self.data.shape) == 2, "Matrix must be 2D"
        self.grad = np.zeros_like(self.data) # Gradient computed by _backward()
        self._backward = lambda: None       # Defaul to no-op, but will be set by the operation that created this value
        self._prev = set(_children)         # The values used compute this value, for backprop. Empty for inputs.
        self._op = _op                      # The op that produced this node, for graphviz / debugging / etc. Empty for inputs.

    def __add__(self, other):
        if isinstance(other, Matrix):
            out = Matrix(self.data + other.data, (self, other), '+')
            def _backward():
                self.grad += out.grad
                other.grad += out.grad
        else:
            out = Matrix(self.data + other, (self, other), '+')
            def _backward():
                self.grad += out.grad

        out._backward = _backward

        assert out.data.shape == self.data.shape, f"Incompatible dimmensions for Matrix addition {out.data.shape} and {self.data.shape}"
        assert len(out.data.shape) == 2, "Can't create a non-2D Matrix"
        return out

    def __mul__(self, other):
        if isinstance(other, Matrix):
            out = Matrix(self.data * other.data, (self, other), '*')
            def _backward():
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad
        else:
            out = Matrix(self.data * other, (self, other), '*')
            def _backward():
                self.grad += other * out.grad

        assert len(out.data.shape) == 2, "Can't create a non-2D Matrix"
        out._backward = _backward

        return out

    def __matmul__(self, other):
        if isinstance(other, Matrix):
            out = Matrix(self.data @ other.data, (self, other), '@')

            def _backward():
                self.grad += out.grad @ other.data.T
                other.grad += self.data.T @ out.grad
        else:
            out = Matrix(self.data @ other, (self, other), '@')

            def _backward():
                self.grad += other.T @ out.grad
        out._backward = _backward

        assert len(out.data.shape) == 2, "Can't create a non-2D Matrix"
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Matrix(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def transpose(self):
        """ create a transpose of this matrix """
        out = Matrix(self.data.T, (self,), 'T')

        def _backward():
            self.grad += out.grad.T
        out._backward = _backward

        return out

    def sum(self):
        """ sum the elements in this matrix """
        out = Matrix(np.sum(self.data, keepdims=True), (self,), 'sum')

        def _backward():
            self.grad += out.grad
        out._backward = _backward

        return out

    def backward(self):
        """ computes the gradients of this value with respect to all of its predecessors """

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited and isinstance(v, Matrix):
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __item__(self, x, y):
        return self.data[x, y]

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return -self + other

    def __rmul__(self, other): # other * self
        return self * other

    def __rmatmul__(self, other): # other @ self
        return Matrix(other) @ self

    def __truediv__(self, other): # self / other
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        return self * other**-1

    @property
    def shape(self):
        """ The shape as a (rows, columns) tuple """
        return self.data.shape

    def __str__(self):
        return f"Matrix({', '.join(str(dim) for dim in self.data.shape)})\n{self.data}"

# Some common activation functions

def relu(val):
    """ ReLU activation function """
    out = Matrix(np.maximum (0, val.data), (val,), 'ReLU')

    def _backward():
        val.grad += (out.data > 0) * out.grad
    out._backward = _backward

    return out

def sigmoid(val):
    """ Sigmoid activation function """
    out = Matrix(1/(1+np.exp(-val.data)), (val,), 'Sigmoid')

    def _backward():
        val.grad += (out.data * (1-out.data)) * out.grad
    out._backward = _backward

    return out

# Neural network layers

class Module:
    """ Base class for neural network modules """

    def zero_grad(self):
        """ zero out the gradients for all parameters """
        for p in self.parameters():
            p.grad.fill(0)

    def parameters(self):
        """ return a list of all Matrix parameters used in this module and its submodules """
        return []

class Linear (Module):
    """ A dense linear layer with nin inputs and nout outputs """

    def __init__(self, nin, nout):
        self.w = Matrix(np.random.uniform(-.1,.1,(nin, nout))/np.sqrt(nin))
        self.b = Matrix(np.random.uniform(-.1,.1,(1, nout)))

    def __call__(self, x):
        if isinstance(x, Matrix):
            return x @ self.w + self.b
        return self.w.__rmatmul__(x) + self.b

    def parameters(self):
        return [self.w, self.b]

    @property
    def shape(self):
        """ The shape as an (inputs, outputs) tuple """
        return (self.w.shape[0], self.w.shape[1])

    def __repr__(self):
        return f"Layer(inputs={self.shape[0]}, outputs={self.shape[1]})"

class ReLU(Module):
    """ ReLU activation layer """

    def __call__(self, x):
        return relu(x)

    def parameters(self):
        return ()

    def __repr__(self):
        return "ReLU"

class Sequential(Module):
    """ A list of other module layers that are to be run sequentially """

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"Sequential([{', '.join(str(layer) for layer in self.layers)}])"

# Now let's define some loss functions that can be applied to a Module's expected vs actual output

def mse_loss (ys, yhats):
    """ Mean Squared Error """
    assert isinstance(yhats[0], Matrix), "yhats must be a collection of Matrices"
    if not isinstance(ys[0], Matrix):
        ys = [Matrix(y) for y in ys]    # ys must be a collection of things that can be converted to matrices
    n_elements = len(ys) * yhats[0].shape[0] * yhats[0].shape[1]
    loss = ((ys[0]-yhats[0])**2).sum()
    for y,yhat in zip(ys[1:],yhats[1:]):
        loss += ((y-yhat)**2).sum()
    return loss / n_elements

def svm_max_margin_loss (ys, yhats):
    """ Max Margin Loss """
    assert isinstance(yhats[0], Matrix), "yhats must be a collection of Matrices"
    if not isinstance(ys[0], Matrix):
        ys = [Matrix(y) for y in ys]    # ys must be a collection of things that can be converted to matrices
    n_elements = len(ys) * yhats[0].shape[0] * yhats[0].shape[1]
    loss = relu(1 - ys[0]*yhats[0]).sum()
    for y,yhat in zip(ys[1:],yhats[1:]):
        loss += relu(1 - y*yhat).sum()
    return loss/n_elements

# Some common metrics for evaluating a model's performance
def accuracy (ys, yhats):
    """ Accuracy - just for binary prediction for now """
    correct = 0
    for y,yhat in zip(ys,yhats):
        correct += (y[0][0] > 0) == (yhat.data[0][0] > 0)

    return correct/len(ys)

# Optimizers
class Optimizer:
    """ Base class for optimizers """
    def step(self):
        """ Update the model's parameters """
        raise NotImplementedError

class SGD(Optimizer):
    """ Stochastic Gradient Descent """

    def __init__(self, params, lr, gradient_cliping=None, lr_decay=0):
        self.params = params
        self.lr = lr
        self.gradient_cliping = gradient_cliping
        self.lr_decay = lr_decay

    def step(self):
        for p in self.params:
            self.lr = self.lr * (1 - self.lr_decay)
            if self.gradient_cliping is not None:
                grad = p.grad.clip (-self.gradient_cliping, self.gradient_cliping)
            else:
                grad = p.grad
            p.data -= self.lr * grad

class SGDRLR (Optimizer):
    """ Stochastic Gradient Descent with dynamic learning rate.
    Auto-tune the learning rate to a ratio of the value to gradient std dev.
    Also track statistics on the optimizations
    """

    # The names of the keys in the step_info dictionary to track various statistics
    KEY_GRAD_MEAN = "grad mean"
    KEY_VALUE_MEAN = "value mean"
    KEY_GRAD_STD = "grad std"
    KEY_VALUE_STD = "value std"
    KEY_LR = "lr"
    KEY_PCT_ZERO_GRAD = "zero grad pct"
    KEY_GRAD_ABS_MAX = "grad abs max"

    def __init__(self, params):
        self.params = params
        self.step_info = []     # An array of step dictionaries with info on each step

    def step(self):
        step_info = OrderedDict()                      # Information on this step
        self.step_info.append (step_info)

        grads = [val for p in self.params for val in p.grad.flatten().tolist()]
        values = [val for p in self.params for val in p.data.flatten().tolist()]
        zeros = sum(1 for g in grads if g == 0) / len(grads)
        lr = np.std(values) / np.std(grads) / 1000

        step_info[SGDRLR.KEY_GRAD_MEAN] = np.mean(grads)
        step_info[SGDRLR.KEY_VALUE_MEAN] = np.mean(values)
        step_info[SGDRLR.KEY_PCT_ZERO_GRAD] = zeros
        step_info[SGDRLR.KEY_GRAD_ABS_MAX] = np.max(np.abs(grads))
        step_info[SGDRLR.KEY_GRAD_STD] = np.std(grads)
        step_info[SGDRLR.KEY_VALUE_STD] = np.std(values)
        step_info[SGDRLR.KEY_LR] = lr

        for p in self.params:
            p.data -= lr * p.grad

    def print_step_info (self, step_num=None):
        """ Print the step info """
        for step in range(len(self.step_info)) if step_num is None else [step_num]:
            print (f"Step {step}:")
            for key, value in self.step_info[step].items():
                print (f"\t{key}:\t{value:.4f}")

# Now let's define a function to train a model

def batch_loss(model, X, y, loss_fn, regularization_alpha, metrics):
    """ Compute the loss for a batch of data, plus optionally additional metrics """
    # forward pass
    yhat = [model(xi) for xi in X]

    # compute loss
    loss = loss_fn(y, yhat)
    if regularization_alpha is not None:
        # L2 regularization
        for p in model.parameters():
            loss += regularization_alpha * (p**2).sum()

    # compute any additional metrics
    return [loss] + [metric(y, yhat) for metric in metrics]

def evaluate(model, X, y, loss_fn=mse_loss, metrics=()):
    """ Compute the loss and metrics for a batch of data """
    return batch_loss(model, X, y, loss_fn, None, metrics)

def fit(model, X, y, optimizer, loss_fn=mse_loss, epochs=1, batch_size=None, regularization_alpha=None, metrics=(), verbose=False):
    """ Train a model the given optimizer and loss function """
    data = list(zip(X, y))

    for epoch in range(epochs):
        if batch_size is None:
            batches = [data]
        else:
            np.random.shuffle(data)
            batches = [data[i:i+batch_size] for i in range(0, len(X), batch_size)]

        epoch_loss = 0
        epoch_metric_values = [0] * len(metrics)

        for batch in batches:
            X_batch, y_batch = zip(*batch)
            # forward
            loss, *metric_values = batch_loss(model, X_batch, y_batch, loss_fn=loss_fn, regularization_alpha=regularization_alpha, metrics=metrics)

            # backward
            model.zero_grad()
            loss.backward()

            # optimize
            optimizer.step()

            # Update metrics
            epoch_loss += loss.data
            if metrics:
                epoch_metric_values = [m + v for m,v in zip(epoch_metric_values, metric_values)]

        if verbose:
            print (f"Epoch {epoch}: avg training loss={epoch_loss.data[0,0]/len(batches):.4f}", end="")
            if metrics:
                print (f", {', '.join(f'{metric.__name__}={m/len(batches)}' for metric,m in zip(metrics, epoch_metric_values))}", end="")
            print ()

def classification_test():
    """ Classification test """
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_moons

    EPOCHS = 10

    np.random.seed(12)  # 12th man, go Seahawks!

    x, y = make_moons(n_samples=100, noise=0.1) # Two interlevered half circles
    y = y.reshape ((100, 1, 1))                 # Make y an array of 1x1 arrays to match the output shape of the model

    # visualize in 2D
    plt.figure(figsize=(5,5))
    plt.scatter(x[:,0], x[:,1], c=y[:,0], s=20, cmap='jet')
    plt.show()

    model = Sequential([Linear(2, 16), ReLU(), Linear(16, 16), ReLU(), Linear(16, 1)])
    print (model)

    # train the model using SGD
    #opt = SGD(model, lr=.1, lr_decay=.02, gradient_cliping=1)
    opt = SGDRLR(model.parameters())
    loss_fn = mse_loss
    fit (model, x, y, loss_fn=loss_fn, optimizer=opt, epochs=EPOCHS, regularization_alpha=1e-4, batch_size=None, metrics=(accuracy,), verbose=True)

    opt.print_step_info()

classification_test()


