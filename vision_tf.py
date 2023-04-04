"""
vision_tf.py - neural network code for image recognition built on tensorflow (so I could learn)

Prereqs: tensorflow

Goals and design: Compare this approach to pytorch

Learning so far: 
* 1/10th as much code as pytorch
* vastly superior training speed
* vastly superior accuracy (98%!)
I'm surely missing something in pytorch, but won't continue with it given the poor ROI.
"""
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print ("Model summary:")
model.summary()

print ("\nModel training:")
model.fit(x_train, y_train, epochs=5)

print ("\nModel evaluation")
model.evaluate(x_test,  y_test, verbose=2)

print ("\Confusion matrix")
y_pred = model.predict(x_train).argmax(axis=1)
cm = tf.math.confusion_matrix(labels=y_train, predictions=y_pred)
print (cm)
