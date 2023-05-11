"""art_generator.py - generate art from a content and style image.

Given a content image and a style image, generate a new image that combines the
content of the content image with the style of the style image.

To use this file, create a directory called "art_images", place a content and style image there,
and make sure the variables CONTENT_IMAGE and STYLE_IMAGE point to the correct files.
Also configure IMG_SIZE and EPOCHS to your liking.

The is based on the art-generation-with-nerural-style-transfer" assignment in
Andrew Ng's CNN class:
https://www.coursera.org/learn/convolutional-neural-networks/programming/4AZ8P/art-generation-with-neural-style-transfer
It uses a pre-trained VGG19 model and loss function that is a weighted average of
* The content loss, which is the squared error between the content and output images
* The style loss, which measures the mean squared error of the gram matrices of the style and output images
"""

# pylint: disable=invalid-name, line-too-long, too-many-local-variables

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Configuration parameters
IMAGE_DIR = "art_images"            # Directory to store images
CONTENT_IMAGE = "LyngorNorway.jpg"  # Content image
STYLE_IMAGE = "StarryNight.jpg"     # Style image
IMG_SIZE = 400                      # Resize everyting to 400x400
EPOCHS = 200                        # More takes longer but gets more stylized

# Generated image name name derived from input file names - also placed in IMAGE_DIR
OUTPUT_IMAGE = CONTENT_IMAGE.rsplit(".", maxsplit=1)[0] + " in the style of " + STYLE_IMAGE

PATHSEP = "/"

def get_np_images (content_img_path : str, style_img_path : str, size : int, display : bool = False):
    """Transform content and style image files to numpy arrays of a given dimension
    content_img_path = path to content image
    style_img_path = path to style image
    size = resize images to size x size
    display = if True, display the images
    """
    content_img = np.array(Image.open(content_img_path).resize((size, size)))
    content_img = tf.constant(np.reshape(content_img, ((1,) + content_img.shape)))
    style_img = np.array(Image.open(style_img_path).resize((size, size)))
    style_img = tf.constant(np.reshape(style_img, ((1,) + style_img.shape)))

    if display:
        print(content_img.shape)
        plt.subplot(1, 2, 1)
        plt.imshow(content_img[0])
        plt.title("Content Image")
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(content_img_path.rsplit(PATHSEP, maxsplit=1)[-1])
        plt.subplot(1, 2, 2)
        plt.imshow(style_img[0])
        plt.title("Style Image")
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(style_img_path.rsplit(PATHSEP, maxsplit=1)[-1])
        plt.show()

    return content_img, style_img

def compute_content_cost(content_output : list, generated_output : list) -> float:
    """
    Computes the content cost function at a given layer
    
    Arguments:
    content_output - a list of images of dimension (1, height, width, channels)
    generated_output -- generated images

    Returns: 
    The squared error between the final content image and the generated image, divided by 4*height*width*channels
    """

    a_C = content_output[-1]    # Get the final content image
    a_G = generated_output[-1]  # Get the final generated image

    # Get the dimensions of a_G
    _, n_H, n_W, n_C = a_G.get_shape().as_list()

    return tf.reduce_sum(tf.square(tf.subtract(a_C, a_G)))/(4*n_H*n_W*n_C)


def compute_layer_style_cost(a_S, a_G):
    """
    Compute the style cost at a given layer

    Arguments:
    a_S -- style image of dimension (1, height, width, channels) after a given layers activation
    a_G -- generated image after a given layers activation

    Returns: 
    The normalized squared error between the style and generated images "gram" matrices; these matrices
    measure the correlation between the different channels of the images
    """
    ### START CODE HERE

    # Get the dimensions of a_G
    _, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape the tensors from (1, n_H, n_W, n_C) to (n_C, n_H * n_W) (≈2 lines)
    a_S = tf.reshape(tf.transpose(a_S, perm=[0, 3, 1, 2]), [n_C, n_H*n_W])
    a_G = tf.reshape(tf.transpose(a_G, perm=[0, 3, 1, 2]), [n_C, n_H*n_W])

    # Computing gram_matrices for both images S and G (≈2 lines)
    GS = tf.matmul(a_S, tf.transpose(a_S))
    GG = tf.matmul(a_G, tf.transpose(a_G))

    # Return the loss
    return tf.reduce_sum(tf.square(tf.subtract(GS, GG)))/(2 * n_C * n_H * n_W)**2

def compute_style_cost(style_image_output, generated_image_output, layers):
    """
    Computes the overall style cost from several chosen layers
    
    Arguments:
    style_image_output -- our tensorflow model
    generated_image_output -- python list containing the content image and the style image
    layers -- A list containing the names of the layers we would like to extract style from
            and a weight for each of them
    
    Returns: The weighted average of the layers style costs
    """

    # initialize the overall style cost
    J_style = 0

    # Set a_S to be the hidden layer activation from the layer we have selected.
    # The last element of the array contains the content layer image, which must not be used.
    a_S = style_image_output[:-1]

    # Set a_G to be the output of the choosen hidden layers.
    # The last element of the list contains the content layer image which must not be used.
    a_G = generated_image_output[:-1]
    for i, weight in zip(range(len(a_S)), layers):  
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])

        # Add weight * J_style_layer of this layer to overall style cost
        J_style += weight[1] * J_style_layer

    return J_style

@tf.function()
def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function
    
    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost
    
    Returns:
    Weighted average of the content cost and style cost
    """
    ### START CODE HERE

    #(≈1 line)
    return alpha * J_content + beta * J_style

def initialize_generated_image(content_image, display=False):
    """Creates a noisy generated image to be used as a starting point for the optimization process"""
    generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
    noise = tf.random.uniform(tf.shape(generated_image), -0.25, 0.25)
    generated_image = tf.add(generated_image, noise)
    generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)

    if display:
        plt.imshow(generated_image.numpy()[0])
        plt.show()

    return generated_image

def clip_0_1(img):
    """
    Truncate all the pixels in the tensor to be between 0 and 1
    
    Arguments:
    image -- Tensor
    J_style -- style cost coded above

    Returns:
    Tensor
    """
    return tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=1.0)

def tensor_to_image(tensor):
    """
    Converts the given tensor into a PIL image
    
    Arguments:
    tensor -- Tensor
    
    Returns:
    Image: A PIL image
    """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

def main():
    """Runs the neural style transfer algorithm"""
    content, style = get_np_images(IMAGE_DIR + PATHSEP + CONTENT_IMAGE,
                                IMAGE_DIR + PATHSEP + STYLE_IMAGE,
                                IMG_SIZE, display=False)
    output = initialize_generated_image(content, display=False)
    vgg = tf.keras.applications.VGG19(include_top=False,
                                    input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                    weights="imagenet")
    vgg.trainable = False

    STYLE_LAYERS = [
        ('block1_conv1', 0.2),
        ('block2_conv1', 0.2),
        ('block3_conv1', 0.2),
        ('block4_conv1', 0.2),
        ('block5_conv1', 0.2)]

    def get_layer_outputs(vgg, layer_names):
        """ Creates a vgg model that returns a list of intermediate output values."""
        outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]

        model = tf.keras.Model([vgg.input], outputs)
        return model

    content_layer = [('block5_conv4', 1)]

    vgg_model_outputs = get_layer_outputs(vgg, STYLE_LAYERS + content_layer)

    # Assign the content image to be the input of the VGG model.  
    preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(content, tf.float32))
    a_C = vgg_model_outputs(preprocessed_content)

    # Assign the input of the model to be the "style" image 
    preprocessed_style =  tf.Variable(tf.image.convert_image_dtype(style, tf.float32))
    a_S = vgg_model_outputs(preprocessed_style)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    output = tf.Variable(output)

    @tf.function()
    def train_step(generated_image):
        """Runs one training step and returns the generated image."""
        with tf.GradientTape() as tape:
            # In this function you must use the precomputed encoded images a_S and a_C

            ### START CODE HERE

            # Compute a_G as the vgg_model_outputs for the current generated image
            a_G = vgg_model_outputs(generated_image)

            # Compute the style cost
            J_style = compute_style_cost(a_S, a_G, STYLE_LAYERS)

            # Compute the content cost
            J_content = compute_content_cost(a_C, a_G)

            # Compute the total cost
            J = total_cost(J_content, J_style, alpha=10, beta=40)


        grad = tape.gradient(J, generated_image)

        optimizer.apply_gradients([(grad, generated_image)])
        generated_image.assign(clip_0_1(generated_image))
        # For grading purposes
        return J

    # Show the generated image at some epochs
    # Uncomment to reset the style transfer process. You will need to compile the train_step function again 
    epochs = 100
    for i in range(epochs):
        train_step(output)
        if i % 10 == 0:
            print(f"Epoch {i} ")
            #image = tensor_to_image(generated_image)
            #plt.imshow(image)
            #plt.show(xticks=[], yticks=[])

    image = tensor_to_image(output)
    image.save(IMAGE_DIR + PATHSEP + OUTPUT_IMAGE)

    # Show the 3 images in a row
    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(1, 3, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(content[0])
    ax.title.set_text('Content image')
    ax = fig.add_subplot(1, 3, 2)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(style[0])
    ax.title.set_text('Style image')
    ax = fig.add_subplot(1, 3, 3)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(output[0])
    ax.title.set_text('Generated image')
    plt.show()

main()
