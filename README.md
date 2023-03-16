I created this project to learn neural nets at a deeper level. I sketched out how to do train a single linear layer on scratch paper and tested it, and then generalized it after reading these articles:
1. https://en.wikipedia.org/wiki/Backpropagation
2. https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795

All the neural net code is in neural_net.py. The test code that actually constructs and trains neural nets is in tests.py. If you want to try it out an explore how to use it, run tests.py.

Here are the details of the architecture, some of this may be obvious is you already know neural nets, but writing it helped me understand it better. A full understanding of this write-up requires that you understand calculus (at least understanding the concept of a gradient) and matrix multiplication.

First; let's talk about the type of problems neural nets can solve. There are two types:
* Classification - for given input, predict discreet valued output. For example, given an image of a hand-written digit(an nxn grid of color intensitites), which digit (0-9) is it? Every time you deposit a check, a neural net figures this out. Computer programs could not solve this prior to the invension of neural nets.
* Regressions. These try to predict a continous numeric value from the input. For example, Zillow's zestimate tries to predict a houses value from inputs like square footage, # bedrooms, and zip code. Neural nets can solve these problems as well, but there are many alteranative methods.

Because classic computer algorithms (where someone hand-crafts rules for how to map inputs to outputs) couldn't solve certain problems that humans can solve easily (such as image classification), the idea emerged to try to create a computer program that could simulate to some degree how the brain worked. Like the brain, some training of the network is needed to teach the network how to perform each task before it can do it on it's own reliably; and the more training the better it becomes. So the combination of a generic neural net + training replaces the need for a programmer to write specific algorithms (such as how to recognize a "9" in an image). With that in mind, let's talk about how real neurons work and motivated the architecture of artificial neural nets:
1. How a real neuron works: https://qbi.uq.edu.au/brain/brain-anatomy/what-neuron. Synopsis: a neuron fires a brief output signal through it's axon if the input it gets (from other neuron's axons via it's dentrites) are above an activation threshold. Inputs are not equally weighted; some axon->dentrite connections are stronger than others, meaning an input from that neuron will more likely trigger activation. Biologically, creating, strengthening, or weakening neural connection is how we learn.
2. How this maps to artificial neural nets: https://towardsdatascience.com/the-differences-between-artificial-and-biological-neural-networks-a8b46db828b7. Synopsis:
a. The network is a set of fully connected "layers" of neurons; so if a given layer has "m" neurons and the previous layer has "n" neurons, then the layer gets "n" inputs (one from each neuron in the previous layer) and produces "m" outputs for the next layer (one from each neuron)
b. The output of each neuron is a linear function of it's inputs, followed by a non-linear "activation" function.

In this implementation, I've modeled the linear function and the non-linear activations as separate layers in the network. The linear layers are modeled as:
    f(x) = w*x + b      (matrix multiplication and addition, which greatly simplifies the math)
Where if:
* n is the number of neurons in the previous layer (so the dimension of a single input value)
* m is the number of neurons in this layer (and so the dimension of a single output value)
* k is the number of input records being processed at a time.
Then:
* x is an nxk matrix of inputs; each column is an input record
* w is an mxn weight matrix; each row are the weights of a single neuron
* b is a  mx1 bias matrix, where each row is the bias for the corresponding neuron
* f(x) is an mxk matrix of outputs; each column is the output for the corresponding input column of x

Here's more intuition and math behind the design of the linear layer:
1. A neuron takes n inputs and produces a single output using a linear function. So a neuron that takes n inputs can be modeled as a weight vector w=(w1, ..., wn) and a bias, and produces, for a given input vector x=(x1, x2, ..., xn), an output of w1*x1 + ... + wn*xn + b.
2. Matrix multiplication simplies the math. We can model a neuron's weight as a length-n row-vector (a 1xn matrix) and a bias, and the input as a length-n column vector (an nx1 matrix). Then the calculation for the output is f(x) = w*x + b, where "*" refers to matrix multiplication
3. An entire linear layer of m neurons can therefor be modeled as an mxn weight matrix (where each row represents the weights of the corresponding neuron), and a mx1 bias matrix. In this case, we can still model the output f(x) = w*x + b. Note that the output, f(x), is a (1xm) column matrix of outputs from each neuron.
4. An entire batch of k inputs training inputs can therefore be modeled as an nxk matrix (where n is the number of inputs in the first layer, and each column is a single training input). In this case, we can stil model the output as f(x) = w*x + b. When doing training, the expected output, y, is modeled as a (kxm) matrix, where each column is the expected output from the corresponding column of x.
5. How do we pick the initial weights and bias? It turns out they can't all have the same value or else training with backprop will always keep them the same. So we start off with initial random small values. This initialization requires more discussion in the issues section at the end.
6. Some papers/models on nerural nets use f(x) = x*w + b (so inputs and outputs are rows, weights are columns); equally valid, but I had to pick one.
7. Some math papers say biases don't need to be modeled as one can get the same effect by adding a column of "1s" to the weight matrix. While mathematically true, this technique can be computationally inefficient during backprop/training compared to explicitly modeling the biases.

Next, let's discuss the non-linear (activation) layers in the network
1. Without non-linear layers, there is no point in having more than 1 layer, since if f(x) and g(x) are linear functions, then so is f(g(x)). So a neural net with no non-linear layers can't model anything more complex than a linear function of it's inputs, which is extremely limited.
2. But what type of non-linear function is needed? As described in the first section, real neurons have what we can call "activation thresholds", where the output (axon) only fires if the weighted inputs are above a given threshold. So artificial neural nets, which are inspired by real neurons, follow this pattern by providing non-decreasing activation functions that just transform each neurons output in a manner that doesn't depend on any of the other neurons. In other words, an activation function a(x) is a non-decreasing mapping of real numbers that is applied element-wise to it's input matrix.
3. As a test case, I've implemented two common activation layers:
a. ReLU; x -> max (x, 0). This is a very common activation function between internal layers. 
b. Sigmoid; x -> 1/(1 + e^-x). This is a very common activation function for the output layer of a neural net designed for binary classification (e.g, is the input a picture of a boat or not?). It produces a value between 0 and 1, so the model is > .5 means "yes" else "no".

So now we can create a neural network by chaining together layers; alternating linear layers with non-linear activation layers. But how do we train it to produce the right answers? That requires:
1. Training data, which consisting of a set of inputs and their corresponding expected outputs. In this implementation, these are specified as matrices x and y, whose columns are input and expected output vectors respectively.
2. Defining a "loss function" that measures how close the actual output (f(x)) is to the expected output (y). I've implemented two common loss functions:
a. MSE - mean squared error; measures the average squared distance between each column of y and f(x). While this is a fairly standard loss function for regression networks, it has issues discussed in the issues section below.
b. CrossEntropy - a standard loss function for classification networks
2. Perform a "forward pass" of the input data in batches. A given training session is typically divided into "epochs" (how many times all the training data is processed), and "mini-batches" (the number of records processed at a time within each epoch; as a hueristic, processing 10-30 at a time typically produces good fit and performance).
3. After each forward pass, compute the loss and then do a backward pass (backprop) to adjust the weights and biases so as to make the loss smaller the next time around (more on this critical step later). After each epoch, I also compute the loss over some validation set to help understand the progress of training per epoch.
4. Finally, run the neural net on some validation data it has not seen before to see how effective it is in making future predictions. The average loss on this data will generally be higher than the loss on the training data as training results in some degree of overfiting.

Here's a bit of intuition behind backprop before we get to the algo:
* First, recall the neural net is basically f(x) = f1(f2(...(fk(x)...) where each fi is either an activation or linear layer.
* Each linear layer has a weight matix and bias vector that need to be adjusted to improve the network. But how should we adjust them? Here's where calculus kicks in. We can pretend each weight matrix, W, is actually a matrix of variables (w11, w12, ...) instead of fixed values. We can then compute the gradient of the loss function with respect to each of these variables and evaluate that gradient at the given current values for the weights. Subtracting a "small multiple" (aka learning rate) of the gradient should decrease loss by moving towards the minima (although if the multiplier is too large we can go past the minimum and make the network worse - see issues section for more info).
* But how do we compute the gradients of the loss function with respect to the weights? Here the chain rule kicks in. We iteratively compute the gradient of the loss function with respect to the inputs of each layer, from back to front. First we compute the gradient of the loss function itself with respect to it's input, f(x):
    dl/dy = gradient of loss function with respect to y=f(x)
For the standard loss functions, this is a very simple calculation (see the code - it's one line of Python each and the deriviative formula is known to anyone who has taken calculus).

Backprop then works backwards one layer at a time using the chain rule. In the code, each layer gets:
* x - the input it got during the forward pass
* dy - the gradient of the loss function with respect to the output it produced; this is computed by the next layer and passed back to it (which is why this is called "back prop", as this layer needs to do the same). The initial dy (passed to the last layer) is the gradient of the loss function with regards to it's input.
The layer then needs to produce a "dy" value for the previous layer (which is the same thing as the it's "dx" value - the gradient of it's input with respect to it's dy).
For the activation layers, this is a trivial calculation (see the code - it's one line of python and how to get the derivative will be known to anyone who has taken calculus).
For the linear layers, it's a bit more complex, but if you work through the math with a couple of examples, you will see:
    dw = dy * x.transpose       # The gradient of the weight; for weight adjustment
    db = avg_over_cols(dy)      # The gradient of the bias; for bias adjustment
    dx = w.T * dy               # The gradient to backprop - to become dy for the previous layer
The first can be worked out by hand (tricky, but that's how I originally did it). The second is obvious if you think of what would happen if the bias was included in the matrix w as a final column vector and x having an added final row vector of 1's to pick up the biases. The last also requires some thought/examples.

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

Issues/thoughts:
1) I've implemented an auto_train method eliminates a few hyper parameters and results in more efficient training:
* Dynamic learning rate. Starts with a relatively high rate (.05), but after adjusting weights for each mini-batch, if the loss function doesn't decrease for that mini-batch, cut the weight adjustment in half and try again. Idea is to make sure gradient decent didn't bypass the local minima by so much as to make loss increase. Then adjust the learning rate using exponential average of the new and old rates; the next mini-batch will start with that.
* Specify target loss rather than number of epochs. Makes training much easier, since loss rate is the goal (not number of epochs)
* Randomly shuffle the training data each epoch so we don't overfit on the mini-batches. This made a huge difference in some of the test cases (going from thousands of epochs needed to reach the target to a much smaller handful).
2) At some point should make the linear layer's weight function configurable so folks can override the default, as it seems like different initializations yeild very different results.
3) ReLU activation can sometimes make everything go to 0 depending on the weight initialization. Training with both negative and positive values helps fix this.
4) The right loss function depents on the problem space. MSE OK for regressions, but when training a linear model like f(x) = 3x it will give wildly different gradients/errors for the same percent error depending on if the training data is near zero or far away.