from __future__ import absolute_import, division, print_function, unicode_literals
import time
# Install TensorFlow
import sys
import tensorflow as tf
import gzip
import _pickle as cPickle

f = gzip.open('mnist.pkl.gz', 'rb')
if sys.version_info < (3,):
    data = cPickle.load(f)
else:
    data = cPickle.load(f, encoding='bytes')
f.close()

(x_train, y_train), (x_test, y_test) = data #mnist.load_data()
    
#mnist = tf.keras.datasets.mnist
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1280, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1280, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1280, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

t1=time.time()
model.fit(x_train, y_train, epochs=1)

model.evaluate(x_test,  y_test, verbose=2)
print('time taken:', time.time()-t1)
