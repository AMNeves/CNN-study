from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here
def cnn_model_fm(features, labels, mode):
  """Model function for the CNN"""
  #Input layer - reshape to 28x28
  input_layer = tf.reshape(features["X"], [-1, 28 , 28, 1])

  #first convolution - 32 5x5 filters with relu activation function
  convo1 = tf.layers.conv2d(
    inputs=input_layer,
    filters=32,
    kernel_size=[5,5],
    padding="same",
    activation=tf.nn.relu
  )

  #pooling layer 1 - pooling with 2x2 filter and stride of 2. pooled no overlap
  pool1 = tf.layers.max_pooling2d(
    input=convo1,
    pool_size=[2, 2],
    strides=2
  )

if __name__ == "__main__":
  tf.app.run()
