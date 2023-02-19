I created this project to learn neural nets at a lower level, as I couldn't find a satisfying summary of how backprop worked; most articles had ungodly complicated nested sigmas with no big picture.

After thinking it through, it occurred to me that all the sigmas I saw were just matrix multiplications, and with that in mind, it wasn't hard to work out after figuring out a few basic principles. But before we get to those, here is some required background info you need to aquire:
a. Structure of a neural network: A basic neural network consists of
   i. A fully interconnected set of neuron "layers", each getting input from all the neurons in the previous layer. Each neuron has a set of weights and a bias so that it's output is the weighted some of the inputs from the previous layer plus the bias.
   ii. A non-linear "activation function", without which stacking the layers of neurons is pointless since stacked linear functions are still linear. The most common activation functions are ReLU and sigmoid, so I'm exploring those.
   iii. For details, see https://wiki.pathmind.com/neural-network (there are many other places to learn)
b. Matric multiplication: The way to express neural networks and backprop is infinitely less complex if one uses matrix algebra. For those not familiar with matrix multiplication, have a look here: https://www.mathsisfun.com/algebra/matrix-multiplying.html.
c. Calculus: You have to have a basic knowledge of differntial calculus to derive the principals. Technically they require "multivariable calculus" (differentiating multi-variable functions with respect to each variable), but taking the "partial" deriviative of f(x, y) with respect to x is really the same thing as taking the regular derivative if you treat y like a constant.

With that in mind, here are the principles:
1) One can think of each layer in the neural network as a function f(X) = W*X, where:
   a. X is a column vector of matrix representing the inputs from the previous layer (so it's nx1, where n = size of previous layer)
   b. W is a matrix of weights, where each row contains the weights for the corresponding neuron. So W is mxn, were m is the number of neurons in this layer, and n is the number of neurons in the previous layer
   c. "*" is matrix multiplication
   d. To add bias, one can just add a "1" to the end of X (so X is now an (n+1)x1 vector), and add a final column of biases to W (so W is now an mx(n+1) matrix). This makes the math more straight-forward as you don't need to do anything special for bias.
2) The "mean squared error" loss function (which represents how accurate the neural net was for a given set of training data - zero being perfect) can be expressed as L = l(n(X) - Y) where:
   a. n() is the neural network function
   b. X is a matrix of input vectors of size (n+1)xk, each solumn represents an input vector with a "1" as the last digit (so n = size of input layer, k = number of training records)
   c. Y is a matrix of expected outputs of size mxk (where m is the number of output layers)
   d. Therefore n(X) - Y represents a matrix of differences between the actual output (n(X)) and the expected output (Y), where each column represents the difference for a particular training record.
   d. l is a function that first computes the squared size of each column (which gives the mean squared error for each training record), then sums these values, then divides by k (the number of training records)
3) After each training session of k inputs, set W = W - r*(WX-Y)*Xt/k where:
    a. Gradient: (WX-Y)*Xt/k is the matrix of partial derivatives of the weights W for training matrices X and Y (here Xt means the transpose of X). In particular, each element in this matrix represents the partial deriviative of the loss function with respect to the corresponding entry in the weight matrix. This matrix of partial derivitives is called the gradient of W with respect to the loss function L. The proof of this is left as an exercise to the reader, but can be worked out by just writing out the equations and differentiating.
    b. The gradient points to the direction of maximum increase of L. Since we want to decrease L, we subtract some multiple ("r") of the gradient; "r" is called the learning rate. Smaller learning rates learn slower, but won't blow past the local minimum. "r" is often called a "hyper parameter"
4) The derivitive of ReLU(X) is (x < 0 ? 0: 1) for each element x in matrix X, the derivative of sigmoid(X) is sigmoid(x)*(1-sigmoid(x)) for each element x in matrix X. These follow directly from the definiton and basic calculus. I'll call these functions d_ReLU(X) and d_sigmoid(X) going forward.

Let's put this all together with an example. Suppose:
    Let n(X) = sigmoid(W2*ReLU(W1*X))
In other words; a 2 layer neural net with ReLU activation for the inner layer and sigmoid for the output - a common practice for classification networks. In this example, we assume:
    n = size of input layer
    m = size of output layer

Suppose further we have a set of k training data, arranged as:
    X = (n+1)xk matrix of training input data (each column is an x+1 vector of training input with a "1" at the end)
    Y = mxk matrix of expected output

Then, let's compute the gradient of W1 and W2 so we can see how to adjust them during backprop:

First: d(n(X))/dW2
= d_sigmoid(d(W2*ReLU(W1*X))/dW2)               # Chain rule
= d_signmoid((W2*ReLU(W1*X) - Y)*ReLU(W1*X)t/k) # Per principle #2 
Note: this is easy to calculate assuming we saved away the value of "ReLU(W1*X)" and "W2*ReLU(W1*X)" during forward propogation (to calculate n(X) for our loss function).

Then: d(n(X))/dW1
= d_sigmoid(d(W2*ReLU(W1*X))/dW1)               # Chain rule
= <OK - I'm stuck here>

More generally:
<generalize so I can write code to support it>
