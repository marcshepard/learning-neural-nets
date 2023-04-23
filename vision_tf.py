"""
vision_tf.py - neural network code for image recognition built on tensorflow (so I could learn)

Prereqs: tensorflow

Goals and design: Compare this approach to pytorch
"""

# pylint: disable=invalid-name, too-many-arguments, line-too-long, too-many-instance-attributes

import tensorflow as tf
from utils import ImageViewer, load_normed_data, plot_training_history

RANDOM_SEED = 12    # 12th man - go Seahawks!
BATCH_SIZE = 256
EPOCHS = 10
SHOW_LOADED_DATA = False

tf.random.set_seed(RANDOM_SEED)

# Load the Fashion MNIST data sets, split off 5000 train records for validation
(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_normed_data(tf.keras.datasets.fashion_mnist, RANDOM_SEED)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

if SHOW_LOADED_DATA:
    print ("Fashion MNIST dataset loaded")
    print ("Classes: ", class_names)
    print ("Training data shape: ", x_train.shape)
    print ("Validation data shape: ", x_val.shape)
    print ("Test data shape: ", x_test.shape)
    ImageViewer(x_train, y_train, class_names)

# Accuracy after 10 epochs: 98.34, 92.56, 92.14 train/dev/test. Overfitting.
# Shirts are hardest to classify (vs t-shirt, pullover, coat, dress).
# Adding another Conv2D layers didn't help. Adding another dense layer didn't help.
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10),
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print ("Model summary:")
model.summary()

print ("\nModel training:")
history = model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_val, y_val), batch_size=BATCH_SIZE)

# Graph the training and validation accuracy/loss
plot_training_history(history)

print ("\nModel evaluation")
train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=0)
val_loss,   val_accuracy   = model.evaluate(x_val,   y_val, verbose=0)
test_loss,  test_accuracy  = model.evaluate(x_test,  y_test, verbose=0)
print ("Data set\tLoss\tAccuracy")
print (f"Train\t\t{train_loss:.2f}\t{train_accuracy*100:.2f}")
print (f"Validate\t{val_loss:.2f}\t{val_accuracy*100:.2f}")
print (f"Test\t\t{test_loss:.2f}\t{test_accuracy*100:.2f}")

print ("\nConfusion matrix")
y_pred = model.predict(x_train).argmax(axis=1)
cm = tf.math.confusion_matrix(labels=y_train, predictions=y_pred)
print (cm)

# Show misclassified images
bad_x = []
bad_y = []
for x, y, pred in zip(x_train, y_train, y_pred):
    if y != pred:
        bad_x.append(x)
        bad_y.append(pred)

ImageViewer(bad_x, bad_y, class_names, title="Misclassified images")
