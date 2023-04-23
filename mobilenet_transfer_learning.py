"""
Transfer learning from Mobilenet, from Andrew Ng's course on CNNs
The use-case here (transfer learning from MobileNetV2 to CIFAR10) gives
terrible results (in his class, the alpaca data set gave much better results,
with 80% initial accuracy and 97% after fine tuning), presumably because the
target data set is closer to the original data set.
"""

# pylint: disable=invalid-name, too-many-arguments, line-too-long, too-many-instance-attributes, import-error

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomZoom, RandomRotation
import tensorflow.keras.layers as layers
from utils import load_normed_data, ImageViewer, plot_training_history

BATCH_SIZE = 64
EPOCHS = 5

IMG_SIZE = (32, 32)
IMG_SHAPE = (32, 32, 3)
RANDOM_SEED = 12    # 12th man - go Seahawks!
tf.random.set_seed(RANDOM_SEED)

SHOW_LOADED_DATA = False

# Load the Fashion tiny image data sets, split off 5000 train records for validation
(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_normed_data(tf.keras.datasets.cifar10, RANDOM_SEED)
y_train = y_train.squeeze()
y_val = y_val.squeeze()
y_test = y_test.squeeze()

class_names = ['Plane', 'Can', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Create a data augmentation layer
data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal", input_shape=IMG_SIZE + (3,)),
    RandomRotation(0.1)
])

if SHOW_LOADED_DATA:
    print ("Cifar10 dataset loaded")
    print ("Classes: ", class_names)
    print ("Training data shape: ", x_train.shape)
    print ("Validation data shape: ", x_val.shape)
    print ("Test data shape: ", x_test.shape)
    ImageViewer(x_train, y_train, class_names)


# We'll not replace the top layer, but freeze the base model
def tiny_img_model(base, preprocessor) -> tf.keras.Model:
    """Create a model for the tiny image dataset using transfer learning from MobileNetV2"""

    # create the input layer (Same as the imageNetv2 input size)
    inputs = tf.keras.Input(shape=IMG_SHAPE)

    # apply data augmentation to the inputs
    x = data_augmentation(inputs)

    # data preprocessing using the same weights the model was trained on
    x = preprocessor(x)

    # set training to False to avoid keeping track of statistics in the batch norm layer
    x = base(x, training=False)

    # add the new Binary classification layers
    # use global avg pooling to summarize the info in each channel
    x = layers.GlobalAveragePooling2D()(x)

    # include dropout with probability of 0.2 to avoid overfitting
    x = layers.Dropout(.2)(x)

    # use a prediction layer with one neuron per class
    outputs = layers.Dense(10)(x)

    # create and return the model
    return tf.keras.Model(inputs, outputs)

# Start with MobileNetV2 as the base model
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet')
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
print ("Base model summary:")
base_model.summary()

# Create a model for the tiny image dataset using transfer learning from MobileNetV2
model = tiny_img_model(base_model, preprocess_input)
print ("Transfer learning model summary:"")
model.summary()

# Train the model with the base model frozen - should be good but sucks
base_model.trainable = False # freeze the base model by making it non trainable
base_learning_rate = 0.001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print ("Training the transfer learning model with base model frozen")
history = model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_val, y_val), batch_size=BATCH_SIZE)
plot_training_history(history)

# Fine tune the model by allowing the last 20% of the layers to be trained with a lower learning rate
base_model.trainable = True
fine_tune_at = len(base_model.layers) * 4 // 5
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate * .1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
print ("Fine tuning the model with a low learning rate and the last 20% of the base layers unfozen")
history = model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_val, y_val), batch_size=BATCH_SIZE)
plot_training_history(history)
