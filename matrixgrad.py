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

    def log(self):
        """ log the elements in this matrix """
        out = Matrix(np.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1/self.data) * out.grad
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

class Sigmoid(Module):
    """ Sigmoid activation layer """

    def __call__(self, x):
        return sigmoid(x)

    def parameters(self):
        return ()

    def __repr__(self):
        return "Sigmoid"

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

def binary_cross_entryopy_loss (ys, yhats):
    """ Binary cross entryopy loss for binary classification """
    assert isinstance(yhats[0], Matrix), "yhats must be a collection of Matrices"
    if not isinstance(ys[0], Matrix):
        ys = [Matrix(y) for y in ys]    # ys must be a collection of things that can be converted to matrices
    n_elements = len(ys) * yhats[0].shape[0] * yhats[0].shape[1]
    loss = -(ys[0]*yhats[0].log() + (1-ys[0])*(1-yhats[0]).log()).sum()
    for y,yhat in zip(ys[1:],yhats[1:]):
        loss += -(y*yhat.log() + (1-y)*(1-yhat).log()).sum()
    return loss / n_elements

# Some common metrics for evaluating a model's performance
def accuracy (ys, yhats):
    """ Accuracy - just for binary prediction for now """
    assert isinstance(yhats[0], Matrix), "yhats must be a collection of Matrices"
    assert yhats[0].data.size == 1, "accuracy only supported for binary predictions for now"
    correct = 0
    for y,yhat in zip(ys,yhats):
        correct += (y.flatten()[0] > .5) == (yhat.data.flatten()[0] > .5)

    return correct/len(ys)

# Optimizers
class Optimizer:
    """ Base class for optimizers """
    def step(self):
        """ Update the model's parameters """
        raise NotImplementedError

class SGD(Optimizer):
    """ Stochastic Gradient Descent """

    def __init__(self, params, lr, lr_decay=0, verbose=False):
        self.params = params
        self.lr = lr                # If None, auto-tune to std(param values) / std(param grads) / 100
        self.lr_decay = lr_decay
        self.verbose = verbose

    def step(self):
        if self.verbose or self.lr is None:
            grads = [val for p in self.params for val in p.grad.flatten().tolist()]
            values = [val for p in self.params for val in p.data.flatten().tolist()]
        if self.verbose:
            zeros = sum(1 for g in grads if g == 0) / len(grads)
            print ("Step info:")
            print (f"\tgrad mean:\t{np.mean(grads):.4f}")
            print (f"\tvalue mean:\t{np.mean(values):.4f}")
            print (f"\tgrad zero pct:\t{zeros*100:.0f}%")
            print (f"\tgrad abs max:\t{np.max(np.abs(grads)):.4f}")
            print (f"\tgrad std:\t{np.std(grads):.4f}")
            print (f"\tvalue std:\t{np.std(values):.4f}")
            print (f"\tgrad/value:\t{np.std(grads)/np.std(values):.4f}, vs lr:\t\t{self.lr:.4f}")

        for p in self.params:
            if self.lr is not None:
                self.lr = self.lr * (1 - self.lr_decay)
                lr = self.lr
            else:
                lr = np.std(values) / np.std(grads) / 100
            p.data -= lr * p.grad

class Adam(Optimizer):
    """ Adam optimizer """
    def __init__(self, params, lr, verbose=False, beta1=.9, beta2=.999):
        self.params = params
        self.lr = lr
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = 1e-8
        self.verbose = verbose

    def step(self):
        # Diagnostic info

        if self.verbose:
            grads = [val for p in self.params for val in p.grad.flatten().tolist()]
            values = [val for p in self.params for val in p.data.flatten().tolist()]
            zeros = sum(1 for g in grads if g == 0) / len(grads)
            print (f"Step {self.t} info:")
            print (f"\tgrad mean:\t{np.mean(grads):.4f}")
            print (f"\tvalue mean:\t{np.mean(values):.4f}")
            print (f"\tgrad zero pct:\t{zeros*100:.0f}%")
            print (f"\tgrad abs max:\t{np.max(np.abs(grads)):.4f}")
            print (f"\tgrad std:\t{np.std(grads):.4f}")
            print (f"\tvalue std:\t{np.std(values):.4f}")
            print (f"\tgrad/value:\t{np.std(grads)/np.std(values):.4f}, vs lr:\t\t{self.lr:.4f}")

        # Update the models parameters using Adam
        self.t += 1
        for i, (p,m,v) in enumerate(zip(self.params, self.m, self.v)):
            m = self.beta1*m + (1-self.beta1)*p.grad
            v = self.beta2*v + (1-self.beta2)*(p.grad**2)
            mhat = m / (1 - self.beta1**self.t)
            vhat = v / (1 - self.beta2**self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.epsilon)
            self.m[i] = m
            self.v[i] = v

# Now let's define a function to train a model

def fit(model, x, y, optimizer, loss_fn=mse_loss, epochs=1, batch_size=None, metrics=(), verbose=False):
    """ Train a model the given optimizer and loss function """
    data = list(zip(x, y))

    for epoch in range(epochs):
        if batch_size is None:
            batches = [data]
        else:
            np.random.shuffle(data)
            batches = [data[i:i+batch_size] for i in range(0, len(x), batch_size)]

        epoch_loss = 0
        epoch_metric_values = [0] * len(metrics)

        for batch in batches:
            x_batch, y_batch = zip(*batch)

            # forward
            yhat = [model(xi) for xi in x_batch]
            loss = loss_fn(y_batch, yhat)

            # backward
            model.zero_grad()
            loss.backward()

            # optimize
            optimizer.step()

            # Update metrics
            epoch_loss += loss.data
            if metrics:
                metric_values = [metric(y_batch, yhat) for metric in metrics]
                epoch_metric_values = [m + v for m,v in zip(epoch_metric_values, metric_values)]

        if verbose:
            print (f"Epoch {epoch}: avg training loss={epoch_loss.data[0,0]/len(batches):.4f}", end="")
            if metrics:
                print (f", {', '.join(f'{metric.__name__}={m/len(batches)}' for metric,m in zip(metrics, epoch_metric_values))}", end="")
            print ()

def evaluate(model, X, y, loss_fn=mse_loss, metrics=()):
    """ Compute the loss and metrics for a batch of data """
    yhat = [model(xi) for xi in X]
    loss = loss_fn(y, yhat)
    metric_values = [metric(y, yhat) for metric in metrics]
    return loss, metric_values
