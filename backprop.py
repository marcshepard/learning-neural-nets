# pylint: disable=line-too-long, invalid-name, protected-access, too-few-public-methods, too-many-arguments, too-many-locals
"""
backprop.py - learning how neural network training works from basic principals

I initially wrote a back-prop library from scratch to learn the algos, and I though the layering was pretty clean.
But then I saw Andrej Karpathy's https://www.youtube.com/watch?v=VMj-3S1tku0 video and accompanying
https://github.com/karpathy/micrograd, and decided to throw away my code and use that instead.
Micrograd is a bit less efficient since backprop is implemented per value (rather than per numpy matrix) so each operation is done by
the Python interpreter, rather than in bulk. But it's more elegant since it uses the moden technique (in PyTorch/Tensorflow) of
constructing a DAG of operations, where each node creates a backward() function for it's input nodes to perform
the backprop (a technique called "autograd"), while my original implementation could only handle sequential layers.

There are a few classes here:
* Value - the main class that wraps a scalar value and supports autograd; derived from AP's engine.py
* Several neural network classes - in the spirit of AP's nn.py

See backprop.ipynb for examples of how to use these classes.
"""

from math import exp, log
import random

# This class is AP's implementation of what I'll call a micro-tensor; a scalar value that supports autograd
# It wraps a scalar value and, if that value was produced by an operation on two other values
# (e.g., val1 + val2), keeps the predecessor values so gradients can be backproped
class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data                # Data being wrapped
        self.grad = 0                   # Gradient computed by _backward()
        self._backward = lambda: None   # Defaul to no-op, but will be set by the operation that created this value
        self._prev = set(_children)     # The values used compute this value, for backprop. Empty for inputs.
        self._op = _op                  # The op that produced this node, for graphviz / debugging / etc. Empty for inputs.

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        """ computes the gradients of this value with respect to all of its predecessors """

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

# Some common activation functions

def relu(val):
    """ ReLU activation function """
    out = Value(0 if val.data < 0 else val.data, (val,), 'ReLU')

    def _backward():
        val.grad += (out.data > 0) * out.grad
    out._backward = _backward

    return out

def sigmoid(val):
    """ Sigmoid activation function """
    out = Value(1/(1+exp(-val.data)), (val,), 'Sigmoid')

    def _backward():
        val.grad += (out.data * (1-out.data)) * out.grad
    out._backward = _backward

    return out

class Module:
    """ Base class for neural network modules """

    def zero_grad(self):
        """ zero out the gradients for all parameters """
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        """ return a list of all parameters used in this module and its submodules """
        return []

class Neuron(Module):
    """ A single neuron with nin inputs, with a set of weights and a bias """

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)

    def __call__(self, x):
        return sum((wi*xi for wi,xi in zip(self.w, x)), self.b)

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"Neuron({len(self.w)})"

class Layer(Module):
    """ A dense layer of nout neurons, each with nin inputs """

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer(inputs={len(self.neurons[0].w)}, outputs={len(self.neurons)})"

class ReLU(Module):
    """ ReLU activation layer """

    def __call__(self, x):
        r = [relu(xi) for xi in x]
        return r

    def parameters(self):
        return ()

    def __repr__(self):
        return "ReLU"

class Sequential(Module):
    """ A sequential container for a list of modules """

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
    loss = 0
    for y,yhat in zip(ys,yhats):
        loss += sum(list((yi-yhati)**2 for yi,yhati in zip(y,yhat)))/len(y)
    return loss / len(ys)

def cross_entropy_loss (y, yhat):
    """ Cross Entropy """
    return -sum(yi*log(yhati) + (1-yi)*log(1-yhati) for yi,yhati in zip(y,yhat))/len(y)

def svm_max_margin_loss (ys, yhats):
    """ Max Margin Loss """
    loss = 0
    for y,yhat in zip(ys,yhats):
        loss += sum(relu((1 + -yi*scorei)) for yi, scorei in zip(y, yhat))/len(y)
    return loss / len(ys)

# Some common metrics for evaluating a model's performance
def accuracy (ys, yhats):
    """ Accuracy """
    correct = 0
    for y,yhat in zip(ys,yhats):
        correct += sum(list((yi > 0) == (yhati.data > 0) for yi, yhati in zip(y, yhat)))
    return correct/len(y)

# Optimizers
class Optimizer:
    """ Base class for optimizers """

    def __init__(self, model, lr):
        self.model = model
        self.lr = lr

    def step(self):
        """ Update the model's parameters """
        raise NotImplementedError

class SGD(Optimizer):
    """ Stochastic Gradient Descent """

    def __init__(self, model, lr, gradient_cliping=None, lr_decay=0):
        super().__init__(model, lr)
        self.gradient_cliping = gradient_cliping
        self.lr_decay = lr_decay

    def step(self):
        self.lr *= (1 - self.lr_decay)
        for p in self.model.parameters():
            if self.gradient_cliping:
                grad = min(max(p.grad, -self.gradient_cliping), self.gradient_cliping)
            else:
                grad = p.grad
            p.data -= self.lr * grad

# Now let's define a function to train a model

def batch_loss(model, X, y, loss_fn, regularization_alpha, metrics):
    """ Compute the loss for a batch of data, plus optionally additional metrics """
    def make_iterable(x):
        return x if hasattr(x, '__iter__') else (x,)

    # Compute the loss for a batch of data, plus optionally additional metrics
    yhat = [make_iterable(model(xi)) for xi in X]

    # compute loss
    loss = loss_fn(y, yhat)
    if regularization_alpha is not None:
        # L2 regularization
        reg_loss = regularization_alpha * sum((p*p for p in model.parameters()))
        loss += reg_loss

    # compute any additional metrics
    return [loss] + [metric(y, yhat) for metric in metrics]

def test(model, X, y, loss_fn=mse_loss, metrics=()):
    """ Compute the loss and metrics for a batch of data """
    return batch_loss(model, X, y, loss_fn, None, metrics)

def fit(model, X, y, optimizer, loss_fn=mse_loss, epochs=1, batch_size=None, regularization_alpha=None, metrics=(), verbose=False):
    """ Train a model the given optimizer and loss function """
    data = list(zip(X, y))

    for epoch in range(epochs):
        if batch_size is None:
            batches = [data]
        else:
            random.shuffle(data)
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
            print (f"Epoch {epoch}: loss={epoch_loss/len(batches):.4f}", end="")
            if metrics:
                print (f", {', '.join(f'{metric.__name__}={m/len(batches)}' for metric,m in zip(metrics, epoch_metric_values))}", end="")
            print ()
