I created this project to learn neural nets at a lower level, as I couldn't find a satisfying summary of how backprop worked; most articles had ungodly complicated nested sigmas with no big picture.

There are two files:
* neural_net.py - a neural net implementation
* tests.py - tests

The original implementation just trained a single linear layer of neurons using
math I sketched out on a pad.

The current implementation includes the following:
1) Layer - a layer in the neural net. In addition to the linear layer, I've added ReLU and sigmoid activation functions as subclasses. These layers are stacked together and each has a "forward" method to turn it's input into input for the next layer, and a "backward" method that knows how to propogate error information to the previous layer (and the linear layers also know how to figure out the gradient of the weights for the given errors) 
2) LossFunction - an abstract class that knows how to measure "loss" (how far off a predicted output is from the expected output) and compute the initial error to use for backward propogation. Two implementations are MSE (mean squared error - a typical function for linear networks) and CrossEntropy (a typical loss function for classification networks)
3) NeuralNet - a concrete class that puts the above together

The ideas here come from:
1) Some math I sketched out on a pad
2) This article: https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795.
3) The layering/abstraction I saw in Keras (although I only have a passing knowledge of Keras)

Some thoughts/TODOs:
1) More thought is needed for linear layer weight initialization. I suspect the right answer depends on the expected input and the activation functions (since ReLU will do best when weights result in initial outputs that are small positive numbers, and sigmoid will likely do best when initial outputs are near 0)
2) I'm not a fan of MSE as a loss function, since (for example), training a linear model f(x) = 3x will give wildly different gradients/errors for the same percent error depending on if the training data is small or not. E.g., if weight is initially .5, (so needs to increase by 2.5), then training data of (.01, .03) will result in a small error/adjustment that will barely help, while training data of (100, 300) will result in a huge error/adjustment that will make things worse. That's why I added the exponential backoff. But I suspect there are better ways. I need to add an abstract base class for optimizing so I can experiment with different optimization algos (just learning rate, vs learning rate + exp backoff, vs, ...).
3) ReLU activation can sometimes make everything go to 0, seems dicy, why not
abs(x) instead if we just need something non-linear?
4) Finish testing sigmoid/classifications to make sure they work.
5) Build a sample application on top of this to really prove things out
