"""
vision_tf.py - neural network code for image recognition built on tensorflow (so I could learn)

Prereqs: tensorflow

Goals and design: Compare this approach to pytorch
"""
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the Fashion MNIST training and test data sets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
# shuffle the training data, then split off 5000 records for validation
idx = tf.random.shuffle(tf.range(len(x_train)))
x_train, y_train = x_train[idx], y_train[idx]
x_val, x_train = x_train[:5000], x_train[5000:]
y_val, y_train = y_train[:5000], y_train[5000:]
# Normalize pixel values to be between 0 and 1
x_train, x_val, x_test = x_train/255.0, x_val/255.0, x_test/255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print ("Fashion MNIST dataset loaded")
print ("Classes: ", class_names)
print ("Training data shape: ", x_train.shape)
print ("Test data shape: ", x_test.shape)

if False:
  plt.figure(figsize=(10,10))
  for i in range(25):
      plt.subplot(5,5,i+1)
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)
      plt.imshow(x_train[i], cmap=plt.cm.binary)
      plt.xlabel(class_names[y_train[i]])
  plt.show()

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

from tensorflow.keras.regularizers import l2
factor = .002

regularization = tf.keras.regularizers.l2(0.001)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(factor), input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(factor)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(256, kernel_regularizer=l2(factor), activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, kernel_regularizer=l2(factor), activation='relu'),
    #tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, kernel_regularizer=l2(factor))
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print ("Model summary:")
model.summary()

print ("\nModel training:")
history = model.fit(x_train, y_train, epochs=50, validation_data=(x_val, y_val), batch_size=128)
# Graph the training and validation accuracy/loss
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')
plt.show(block = True)

# Graph the training and validation loss
plt.subplot(2, 1, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show(block = True)


print ("\nModel evaluation")
train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=0)
val_loss,   val_accuracy   = model.evaluate(x_val,   y_val, verbose=0)
test_loss,  test_accuracy  = model.evaluate(x_test,  y_test, verbose=0)
print ("Data set\tLoss\tAccuracy")
print (f"Train\t\t{train_loss:.2f}\t{train_accuracy:.2f}")
print (f"Validate\t{val_loss:.2f}\t{val_accuracy:.2f}")
print (f"Test\t\t{test_loss:.2f}\t{test_accuracy:.2f}")

print ("\nConfusion matrix")
y_pred = model.predict(x_train).argmax(axis=1)
cm = tf.math.confusion_matrix(labels=y_train, predictions=y_pred)
print (cm)
