from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here
def cnn_model_fn(features, labels, mode):
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
  pool1 = tf.layers.max_pooling2d(input=convo1,pool_size=[2, 2],strides=2)

  #convo2 and pool2
  convo2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu
  )
  pool2 = tf.layers.max_pooling2d(input=convo2, pool_size=[2, 2], strides=2)

  #dense layer 1
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 *64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(input=dense, rate=0.4, training = mode == tf.estimator.ModeKeys.TRAIN)

  #logits output layer
  logits = tf.layers.dense(input=dropout, units=10)

  predictions = {
    #Generate predictions
    "classes": tf.argmax(input=logits, axis=1),
    #add softmax tensor to the graph, used for predict and logging hook
    "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
  
  #loss calculation
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  #configure training op
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = otimizer.minimize(
      loss=loss,
      global_step=tf.train.get_global_step()
    )
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    



if __name__ == "__main__":
  tf.app.run()
