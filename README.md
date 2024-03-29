# Overview
I created this project to learn neural nets at a deeper level. This section is an overview of the contents. Next section is some background on neural nets. The last section is a summary of technical details I've learned so far.

## backprop.py
Basic neural net layers and test code built on numpy to learn how neural networks are trained from basic principals (calculus and matrix arithmatic).
Prereqs: numpy, matplotlib

## vision_pt.py
FashionMNISTA classifier built with PyTorch to learn PyTorch, CNNs, and practice optimizing on a real framework.
Prereqs: torch, matplotlib

## vision_tf.py
FashionMNISTA classifier built with tensorflow to compare to pt
Prereqs: tensorflow

# Why neural nets?
First; let's talk about the type of problems neural nets can solve:
* Classification - for given input, predict discreet valued output. For example, given an image of a hand-written digit(an nxn grid of color intensitites), which digit (0-9) is it? Every time you deposit a check, a neural net figures this out. Computer programs could not solve this prior to the invension of neural nets.
* Regressions. These try to predict a continous numeric value from the input. For example, Zillow's zestimate tries to predict a houses value from inputs like square footage, # bedrooms, and zip code. Neural nets can solve these problems as well, but there are many alteranative methods.
* Generative models, such as the LLM (large language model) that powers ChatGPT or stable diffusion that powers image generation tools like midjourney

Because classic computer algorithms (where someone hand-crafts rules for how to map inputs to outputs) couldn't solve certain problems that humans can solve easily (such as image classification or speach), the idea emerged to try to create a computer program that could simulate to some degree how the brain worked. Like the brain, some training of the network is needed to teach the network how to perform each task before it can do it on it's own reliably; and the more training the better it becomes. So the combination of a generic neural net + training replaces the need for a programmer to write specific algorithms (such as how to recognize a "9" in an image). With that in mind, let's talk about how real neurons work and motivated the architecture of artificial neural nets:
1. How a real neuron works: https://qbi.uq.edu.au/brain/brain-anatomy/what-neuron. Synopsis: a neuron fires a brief output signal through it's axon if the input it gets (from other neuron's axons via it's dentrites) are above an activation threshold. Inputs are not equally weighted; some axon->dentrite connections are stronger than others, meaning an input from that neuron will more likely trigger activation. Biologically, creating, strengthening, or weakening neural connection is how we learn.
2. How this maps to artificial neural nets: https://towardsdatascience.com/the-differences-between-artificial-and-biological-neural-networks-a8b46db828b7. Synopsis:
    * The network is a set of fully connected "layers" of neurons; so if a given layer has "m" neurons and the previous layer has "n" neurons, then the layer gets "n" inputs (one from each neuron in the previous layer) and produces "m" outputs for the next layer (one from each neuron)
    * The output of each neuron is a linear function of it's inputs, followed by a non-linear "activation" function. There is no point in having linear layers direction connect directly to each other since composing linear functions just creates another linear function, so one always puts non-linear activation functions between them.

# What I've learned so far.
I started with self-directed learning in building the first two files above. Then pivoted to Andrew Ng's deep learning specialization series, which I can't recommend enough:  https://www.coursera.org/specializations/deep-learning. I'm summarizing my take-aways from the first 3 classes below, so I don't forget these tips as a check-list for future projects.

## Class # 1: Neural net construction
How a basic neural net is constructed; with layers, random initial weights for the linear layers, and backprop to train the weights, is covered in his first course, https://www.coursera.org/learn/neural-networks-deep-learning (actually, weight initialization is covered in the 2nd course, https://www.coursera.org/learn/deep-neural-network, where he recommends initial weights of randn()*sqrt(2/n) to mitigate the exploding/vanishing gradient problem). I won't repeat that here as backprop.py has most of this coded up (except uses simpler weight initialization). In practice, any modern framework (like PyTorch - which is more flexible as it keeps you closer to the details, or TensorFlow/Keras - which is easier to code as it hides more of the detais) obviates the need to do any of these things by hand, but it's important to know how things work under the covers.

## Class #2: Neural net training
A quick summary of tips from his 2nd course, https://www.coursera.org/learn/deep-neural-network:
1. Pick a target metric (what you are aiming to optimize) as well as the loss function you will use as a proxy (which may be different as you want something differentiable). E.g.:
   1. For classification, your loss function might be CrossEntropy, but your target metric might be accuracy or F1 score.
   2. For regressions, your loss function might be MSE, but your target metric might be MAE
2. Find and clean your data and divide it into three data sets (each with the same distribution):
   1. Train. Training data. Most data goes here. Overfitting is common here without regularization. 70% of your data for small data sets, most of your data for large data sets.
   2. Dev. Used to measure trained networks against new data to help select final model (among different choices for hyper parameters and model layers) to pick the final model. Overfitting can also happen here if there are a lot of different models being tried. 20% of your data for small data sets, enough to be statistically significant for large data sets (e.g, 10k).
   3. Test. Used to measure your final model to get an idea how it will perform in the real world against unseen data. 20% of your data for small data sets, enough to be statistically significant for large data sets (e.g, 10k).
3. Normalize inputs. If (and only if) the input variables have order-of-magnitude different ranges, add a normalization layer to give them all the same standard deviation; this helps gradient decent during backprop move more efficiently to the min (rather than picking a more wobbly path). 
4. During training, keep track of both how things are going against the train and dev sets. Watch of for high bias = underfitting = not getting good results on training data or high variance = overfitting = not getting good results on dev set.
   1. To reduce underfitting: create a bigger network, train longer, tune hyper parameters
   2. To reduce overfitting: more training data, regularization (esp L2 and drop-out)
   3. If these don't work, consider alternative network architectures
5. Training tips
   1. Mini-batches. Size is hyper-param (1 = too small/too stochasitic, full batch = may be slow if doesn't fit in CPU/GPU memory). Rule of thumb; if small training set (<2k), use the whole batch. Else, a power of 2 between 64-512. Should randomize which records go into which mini-batches each epoch to prevent overfitting to specific mini-batches.
   2. Optimization algos. Stochastic gradient descent is baseline (what is in backprop.py) but can result in each mini-batch heading in wildly different (often wrong) directions. One technique to get faster covergence is "momentum", where gradient is exponentially weighted avg (~.9) of mini-batches (so no mini-batch goes off too far from avg direction). Alternative is RMSprop (divide each dw by the exponentially weighted avg of previous dw's). An optimizer like Adam does these and is typically more effective than pure SGD.
   3. Learning rate decay. Another hyper parameter one can tune to fine-tune algo as it gets close to the local optima (and local optima almost always = global optima in high dimentional spaces).

## Class #3: Neural net projects
A quick summary from his 3rd course on attacking ML projects, https://www.coursera.org/learn/machine-learning-projects
The basics:
1. Select a single metric to optimize and loss function proxy. It's OK to have other metrics as "requirements", but only one to optimize
2. Estimate the "bayes error" (minimum possible error for a perfect algo). Bayes error is typically > 0 due to data noise. An approximation for image classification might be that of what a human expert can do.
3. The difference between the error rate on your training data and bayes error is your "avoidable bias" (see previous section for how to fix)
4. The difference between the error rate on your training data and dev data is your "variance error" (see previous section for how to fix)
5. The difference between the error rate on your dev data and test data is another form of variance error that might happen if you have tried many, many models and used the dev set to pick one. You can increase the size of your dev set.
6. Measuring (3), (4), and (5) up front will tell you which optimizations to go after to get the best bang-for-your-buck

Data mismatches between training and dev/test data: If you need to fine tune your model for new use cases, and have relatively small amount of the new data to train on (compared to your existing training data), then mixing it in with your training data won't help. In that case:
0. It's OK to have training data have a different distribution than dev/test data. But the dev/test data must have the same distribution, since that is your "target". So distribute the new data into your dev and test sets. Do this evenly, since they must have the same distribution.
2. Carve off a subset of your training data into a new set called "dev-training"; this is what you will use for hyper-parameter tuning of your training data and to measure model variance issues. It's important to be able to measure variance issues in your model without also having to account for the new data for the new use case.
3. The dev data is now used to just measure model performance with respect to the new data.

Transfer learning: This means taking an existing neural net and reusing as the starting point for a new project.
1. This is often productive because, for example, the early layers in image classifiers often recognize basic contours, which is useful for all types of image classification (not just the one in your old project) and so for image classification, often freezing the initial layers and just replacing the output is effective. Followed by fine-tuning (unfreezing just last few initial layers and retraining with a smaller learning rate). See mobilenet_transfer_learning.py as an example.
2. LLMs use this technique as well; they call the initial training "pre-training" (which is typically unsupervised on a large amount of data), and the hand-crafted training after that fine tuning (although unlike image classification transfer, in LLMs typically all the weights are adjusted in this phase).

Multi-task learning: If you need to make multiple independent predictions from the same data (e.g., separate logits instead of softmax), it's called "multi-task learning". Cononical example is computer vision object detection (where image might contain multiple objects). It's an alternative to simply creating separate neural nets for each outcome (which can leverage transfer learning). Less common than transfer learning, but has proven useful in computer vision object detection.

Chained models: Sometimes it makes sense to have a single neural net to solve an entire problem end-to-end (e.g, audio -> transcription). Other times it makes more sense to chain together separate networks (e.g, image -> face -> employee).
* Use chained models if there are large amounts of differet training examples for each component. Or if there is less data. Or if there are useful already designed components that can be leveraged.
* Use an E2E model otherwise; it's often better 

## Class #4: CNNS
Lots of discussion on how conv and max-pool operations work not repeated here.
Vanilla architecture:
* Scale input images to 0-1 before input to network
* A few layers of (conv + max_pool + relu)
* A few layers of vanilla connected layers (might do batch norm input)

Key insights:
* Use conv padding="same" to preserve impact of edge pixels and to not cause problems with deep networks

Variants:
* Resnet: To solve the problem of vanishing/exploding gradients in deep neural nets, can add previous layer output to next layers output
* Inception: To solve problem of "which convulution is most important", create a layer that tries them all and let the network decide.
   * Inception layer output has same h,w, but channels are a "stack" of different convolution sizes + max pool
   * Need padding = "same" for all so things fit. Stride = 1 for max_pool
   * To reduce computation, add 1x1x<small number of channels> conv before each regualar conv, this "bottleneck" reduces dimension, picks most important features, reduces computations. Can add a 1x1 conv after maxpool to get the desired channels for that part.
   Inception network has multiple inception layers stacked, a softmax output from each layer to create a regularizing effect
   Invented by google. They called it googLeNet (not to LeNet). Inception named after the Sci fi movie:)
* MobileNet: To reduce the number of parameters and processing time for mobile apps (which have no GPU and slower CPUs):
   * Apply "depth-wise separable" convolutions.
      * Recall a normal convolusion with (hxwxic) input volume (ic = input channels) is an (fxfxcxoc) block of filter parameters, where f = filter size and oc=output channels. This requires (fxfxcxoc) * (# filter positions) multiplications
      * Depthwise separable convolusion instead does these three steps:
         * A depthwise convolution of ic (hxw) filters (one per channel), and applies each one just the input of the same channel. So that requires just (fxfxic) * (# filter posisions) multiplications. In other words, 1/oc less multiplications.
         * A pointwise convolution, of oc (1x1xic) convolutions. This requires (oc * ic) * (# filter posisions) multiplications
         * So we have a total of (fxfxic + oc*ic) * (# filter positions). Both have ic * # filter position factors. But normal has (fxfxoc) of these, while depth-wise separable as only (fxf + c) of these. On other words, still about 1/oc less multiplications.
      * MobileNet v1 has 13 of these layers followed by pooling/Fully connected layer/softmax. MobileNet v2 adds residual skip connections between depth-wise separable layers (like resnet). Some of the blocks ("bottleneck blocks) and also adds an "expansion" layer within each depth-wise separable layer, whcih expands the number of channels at the start of the layer (convolve with pointwise 1x1x(ic*6)) before depth-wise convolusion and projection to the output channel size.
* EfficientNet - tries several approaches to help you chose a good trade off between resolution, depth, and width.

Above was about classification. In real-world problems there may be one of more things in a given image (e.g, dashcam might include multiple cars, pedestrians, signs, a road area, etc). So solve that, one needs to detect all the items (object detection) and where they are (localization = finding it's bounding box). There are several approaches to this:
* Convolutional implementation of sliding windows. Idea is that creating sliding windows to check for objects is too slow if done window-at-a-time. But since neighboring sliding windows have overlapped convolution outputs, those can be reused. You can get this for free by replacing the fully connected layer at the end with an nxnxoc convolution that reduces the output height and width to match the number of rows/columns of windows. The each successive "fully connected" layer is another pointwise convolusion, so we keep info for each window until the end; with a softmax output per window for predicted item in the box (or 0 for none), as well as the bounding box for that item. So target output is (grid rows x grid columns x (5 + num_categories)) - the 5 corresponds to bounding box (for values from 0-1) and if there is anything there at all.
* YOLO algo; place a grid (perhaps 19x19) over the image. Training is per grid-cell; with object assigned to whichever cell contains the objects mid-point. TO determine which box gets the input, one measures "intersection over union" and performs non-max suppression for overlapped bounding boxes. Intersection over union is the ratio of the intersection to the union of the predicted bounding box and the labeled bounding box. If > 50% (or up to 70%), then it's good. Non-max suppression means finding the strongest signal for a given object, then removing neighboring predictions with overlapping bounding boxes - repeat until no overlaps. This prevents neighboring cells from predicting the same object. It also adds the concept of "anchor boxes" which are shapes within each bounding box. Objects are then mapped to the combination of (box, anchor box) - this allows multiple objects to live in the same box.
* U-net. Idea is to classify each pixel on the screen based on what type of thing it is (e.g., heart, liver, lungs; or car, street, building; or cat, background). Architecture is convolutions to reduce height/width and increase channels. Then "transpose convolusions" to expand them. Final output should have same height/width as original, but n channels (one per category) instead of 3 (RGB). There is also a res-net style skip connection joining each pair of the same height/width (from the convolution decent to the transpose convulution ascent). Idea is classification needs both the fine grained details (from skip connection) as well as the categorization (from convolutions)





