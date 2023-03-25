# Overview
I created this project to learn neural nets at a deeper level. Here are a summary of the contents

## Backprop
Overview: This was created so I could learn how neural networks are trained from basic principals (basically the calculus and matrix arithmatic behind backprop). More details (including links to where I learned some of the techniques) are in the backprop.py header.

Files:
* backprop.py           - neural net code
* backprop_tests.py     - tests

Prereqs:
* numpy (e.g., pip install numpy)
* matplotlib (pip install matplotlib)

## Vision
Overview: This was created so I could learn how to use PyTorch for basic image recognition. More details (including links to where I learned some of the techniques) are in the vision.py header.

Files:
* vision.py
* vision_tests.py

Prereqs:
* pytorch (pip install numpy)
* torchvision (pip install torchvision)
* matplotlib (pip install matplotlib)

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
