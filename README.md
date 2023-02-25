I created this project to learn neural nets at a deeper level. I sketched out how to do a single linear neuron on scratch paper and tested it, and for this project I drew inspiration from the following sources:
1. https://en.wikipedia.org/wiki/Backpropagation
2. https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795
3. Keras (I barely know it, but saw how they structured layers)

Here are the details of the architecture, some of this may be obvious is you already know neural nets, but I'm writing it all down to help myself understand it better. A full understanding of the below has requires calculus (at least understanding the concept of a gradient) and matrix multiplication.

First; let's talk about how real neurons work and which concepts form the basis of artificial neural nets:
1. Here's a quick summary of how a real neuron works: https://qbi.uq.edu.au/brain/brain-anatomy/what-neuron. Note that it fires a brief output signal through it's axon if the input it gets (from other neuron's axons via it's dentrites) cause an activation. Activation requires some threshold of inputs. Inputs are not equally weighted; some axon->dentrite connections are stronger than others.
2. Here's a quick summary of how artificial neural networks are typically architected as inspired by biological neural networks: https://towardsdatascience.com/the-differences-between-artificial-and-biological-neural-networks-a8b46db828b7. A few things to point out about artificial neural networks:
a. They are aranged in fully connected "layers" of neurons; an input layer of "m" neurons, each taking "n" input signals and producing "m" output signals (from their artificial axons). Fully connected means each neuron in a given layer gets all outputs from the previous layer (through their artificial dentrites)
b. The output of each neuron is modeled as a linear function of it's inputs, followed by a non-linear "activation" function. And these layers are chained together to form a full network.
c. In this implementation, I've modeled the linear function and the non-linear activations are separate layers, but some implementations (e.g, Keras) put them together.

With that in mind, I've modeled the linear layers as:
    f(x) = w*x + b (matrix multiplication and addition)
Where if:
* n is the number of neurons of the previous layer (so the dimension of a single input value)
* m is the number of neurons in this layer (and so the dimention of a single output value)
* k is the number of input records being processesd at a time.
Then:
* x is an nxk matrix of inputs; each column is an input record
* w is an mxn weight matrix; each row are the weights of a single neuron
* b is a  mx1 bias matrix, where each row is the bias for the corresponding neuron
* f(x) is an mxk matrix of outputs; each column is the output for the corresponding input column of x

Here's more intuition and math behind the design of the linear layer:
1. A neuron takes n inputs and produces a single output using a linear function. So a neuron that takes n inputs can be modeled as a weight vector w=(w1, ..., wn) and a bias, and produces, for a given input vector x=(x1, x2, ..., xn), an output of w1*x1 + ... + wn*xn + b.
2. Matrix multiplication simplies the math a lot. We can model a neuron's weight as a length-n row-vector (a 1xn matrix) and a bias, and the input as a length-n column vector (an nx1 matrix). Then the calculation for the output is f(x) = w*x + b, where "*" refers to matrix multiplication
3. An entire linear layer of m neurons can therefor be modeled as an mxn weight matrix (where each row represents the weights of the corresponding neuron), and a mx1 bias matrix. In this case, we can still model the output f(x) = w*x + b. Note that the output, f(x), is a (1xm) column matrix of outputs from each neuron.
4. An entire batch of k inputs training inputs can therefore be modeled as an nxk matrix (where n is the number of inputs in the first layer, and each column is a single training input). In this case, we can stil model the output as f(x) = w*x + b. When doing training, the expected output, y, is modeled as a (kxm) matrix, where each column is the expected output from the corresponding column of x.
5. How do we pick the initial weights and bias? It turns out they can't all have the same value or else training with backprop will always keep them the same. So we start off with initial random small values. This initialization requires more discussion in the issues section at the end.
6. Some papers/models on nerural nets use f(x) = x*w + b (so inputs and outputs are rows, weights are columns); equally valid, but I had to pick one.
7. Many math papers say biases don't need to be modeled as one can get the same effect by adding a column of "1s" to the weight matrix. While mathematically true, I found this technique to be computationally inefficient during backprop/training compared to explicitly modeling the biases.

Next, let's discuss the non-linear (activation) layers in the network
1. Without non-linear layers, there is no point in having more than 1 layer, since if f(x) and g(x) are linear functions, then so is f(g(x)). So a neural net with no non-linear layers can't model anything more complex than a linear function of it's inputs, which is extremely limited.
2. But what type of non-linear function is needed? As described in the first section, real neurons have what we can call "activation thresholds", where the output (axon) only fires if the weighted inputs are above a given threshold. So artificial neural nets, which are inspired by real neurons, follow this pattern by providing activation functions that just transform each neurons output in a manner that doesn't depend on any of the other neurons, and are modeled as non-decreasing functions. In other words, an activation function a(x) is a non-decreasing mapping of real numbers that is applied element-wise to some matrix.
3. As a test case, I've implemented two common activation layers:
a. ReLU; x -> max (x, 0). This is probably the most common activation function between internal layers. 
b. Sigmoid; x -> 1/(1 + e^-x). This is the most common activation function for the output layer of a neural net designed for binary classification (e.g, is it a picture of a boat or not?). It produces a value between 0 and 1, so the model is > .5 means "yes" else "no".

So now we can create a neural network by chaining together layers (typically alternating linear layers with non-linear activation layers). But how do we measure it's effectiveness and train it to get better? First let's talk about training. For that we need a set of training records along with the expected output of each, and then adjust the linear layer weights so make things better. That requires:
1. Defining a "loss function" that measures how close f(x) is to y, for training input records x (expressed as a matrix of column input vectors) and expected outputs y (expressed as a matrix of column output vectors). I've implemented two common loss functions:
a. MSE - mean squared error; measures the average squared distance between each column of y (expected output) and f(x) (actual output). While this is a fairly standard loss function for regression networks, I'm not a fan of it for reasons discussed in the issues section below.
b. CrossEntropy - the defacto standard for measuring the effectvieness of classification networks
2. Do a "forward pass" of the input data in batches. Note a given training session is typically divided into "epochs" (how many times all the training data is processed), and "mini-batches" (the number of records processed at a time within each epoch; typically 10-30 produces good fit and performance).
3. After each forward pass, compute the loss of expected output y vs actual output f(x) and then do a backward pass (backprop) to adjust the weights. More on this later. After each epoch, compute the loss of some validation set (for now I'm using the entire training set for that) so understand the effectiness of the training. The neural net stores the loss per epoch in a list so it can be viewed later.
4. Finally, run the neural net on some validation data it has not seen before to see how effective it is in making future predictions. Note you don't want to use loss you have seen on the the training data for this purpose as training generally results in some degree of overfitting, so training loss will generally be lower than the loss you will get in actual usage.

Here's a bit of intuition behind backprop before we get to the algo:
* First, recall the neural net is basically f(x) = ak(Lk(...a1(L1(X)...)) where each ai is an activation layer and each Li is a linear layer.
* Each linear layer, Lk, has a weight matix and bias vector that needs to have their weights adjusted (as these are the only configurable items in the layers). But how much should we adjust them by?
* Here's where calculus kicks in. We can pretend each weight matrix, W, is actually a matrix of variables (w11, w12, ...) instead of fixed values. We can then compute the gradient of the loss function with respect to each of these variables and evaluate that gradient at the given current values for the weights. Subtracting a "small multiple" of the gradient will decrease loss.
* But what does "small multiple"? Too much may move you past the local minimum of loss and result in an increase in loss. Too little may result in painfully slow training needed a ton of epochs. The standard answer is to push the problem to the person doing the training by having them define a "learning rate" that is multiplied by the gradient before subtracting it from the weights; folks typically initially set to .05 or .01. And then the user has to experiement with different values for the "learning rate" hyper parameter to figure out which works best. This is BS; the algo should do this automatically (mine does, but I suspect there are better ways).
With that in mind, here is the math for the backprop:
* First, let's consider just the very last linear layer and beyond. We can do this by considering everything before that as a single input matrix. In other words we can rewrite:
    f(x) = ak(Lk(...a1(L1(X)...))
As:
    f(h) = a(w*h + b)
Where w/b = weight and bias of the last linear layer, a = last activation function, h = all previous layers applied to x. With that in mind, we need to compute
* dw = gradient of the loss function with respect to each element in the weight matrix (pretending the weight matrix is actually a matrix of variables, and evaluating the gradient of L with respect to each of these variables at the current weight value)
* db = similar for bias
Then the thing to notice is that the chain rule gives:
   dw = a'(dl) * h.transpose()
   db = a'(dl)
where a' is the derivative of the activation function with respect to it's input w*h + b), and dl is the derivative of the loss function with respect to it's input (f(x)).

dl/dl2 at layer 2's input is dl/dl1 at layer 1's input * dl1/dl2 at layer 2's input.


dl/dw = dL/d(err) * d(err)/da * da/dw


Here's how the above summary maps to the actual source code:
* neural_net.py consists of the following classes:
    * Layer - abstract base class for a neural network layer
        * Linear - linear layer
        * Sigmoid - sigmoid activation layer
        * ReLU - ReLU activation layer
    * CostFunction - abstract base class for a cost function
        * MSE - mean squared error cost function (for regressions)
        * CrossEntropy - cross entropy cost function (for classification)
    * NeuralNetwork - configured adding a set of layers and selecting a cost function. It then has methods train (to train the network using backprop), predict (to predict output for a given input), and loss (just a wrapper for the currently configured CostFunction's loss method)
* test.py - test cases to verify the neural net works as intended for various layer configurations and training + validation data.

Some additional thoughts/TODOs:
1) More thought is needed for linear layer weight initialization. I suspect the right answer depends on the expected input and the activation functions. Different initializations algos give wildly different results (e.g., random between -.5 and .5 vs random between 0 and 1 vs normal distribution scaled to some value), as do different starting seeds.
2) I'm not a fan of MSE as a loss function, since (for example), training a linear model f(x) = 3x will give wildly different gradients/errors for the same percent error depending on if the training data is small or not. E.g., if weight is initially .5, (so needs to increase by 2.5), then training data of (.01, .03) will result in a small error/adjustment that will barely help, while training data of (100, 300) will result in a huge error/adjustment that will make things worse. That's why I added the exponential backoff. But I suspect there are better ways. I need to add an abstract base class for optimizing so I can experiment with different optimization algos (just learning rate, vs learning rate + exp backoff, vs, ...).
3) ReLU activation can sometimes make everything go to 0. So depending on the random seed used to produce the initial weights, the neural net might be great or terrible. Seems like their should be something better that always converges. This is related to (2) above; would like something more deterministic that always worked without the user having to configure learning rates.
4) Another hyper-parameter I'd like to make optional is number of epochs. Ideally the algo should be able to tell when loss is not decreasing and stop (or not decreasing enough per epoch over enough epochs - perhaps that would be a better hyper parameter).
* Finish testing sigmoid/classifications to make sure they work.
* Build a sample application on top of this to really prove things out
