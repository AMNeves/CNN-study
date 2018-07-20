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
  input_layer = tf.reshape(features["x"], [-1, 28 , 28, 1])

  #first convolution - 32 5x5 filters with relu activation function
  convo1 = tf.layers.conv2d(
    inputs=input_layer,
    filters=32,
    kernel_size=[5,5],
    padding="same",
    activation=tf.nn.relu
  )
  #pooling layer 1 - pooling with 2x2 filter and stride of 2. pooled no overlap
  pool1 = tf.layers.max_pooling2d(inputs=convo1,pool_size=[2, 2],strides=2)

  #convo2 and pool2
  convo2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu
  )
  pool2 = tf.layers.max_pooling2d(inputs=convo2, pool_size=[2, 2], strides=2)

  #dense layer 1
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 *64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(inputs=dense, rate=0.4, training = mode == tf.estimator.ModeKeys.TRAIN)

  #logits output layer
  logits = tf.layers.dense(inputs=dropout, units=10)

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
    train_op = optimizer.minimize(
      loss=loss,
      global_step=tf.train.get_global_step()
    )
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics for EVAL mode
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  #estimator
  mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

  #loggin
  tensor_to_log = {"probabilities" : "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(tensors=tensor_to_log, every_n_iter=50)

  #training

  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x= {"x": train_data},
    y= train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)

  mnist_classifier.train(input_fn=train_input_fn, steps=20000, hooks=[logging_hook])
  
  #evaluation
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


    
  
if __name__ == "__main__":
  tf.app.run()
