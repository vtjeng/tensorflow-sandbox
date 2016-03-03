# implementing beginner's tutorial, found at https://www.tensorflow.org/versions/r0.7/tutorials/mnist/beginners/index.html

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

num_features = mnist.train.images.shape[1]
num_label_types = mnist.train.labels.shape[1]
# note that mnist.train.images and mnist.train.labels are just numpy arrays

x = tf.placeholder(tf.float32, [None, num_features])
# IS THIS A TYPE SYSTEM I SEE HERE
# None means that a dimension can be of any length. that's awesome.

W = tf.Variable(tf.zeros([num_features, num_label_types]))
b = tf.Variable(tf.zeros([num_label_types]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
# y is our predicted probability distribution

y_ = tf.placeholder(tf.float32, [None, num_label_types])
# y_ is the actual probability distribution

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# we're attempting to minimize the cross_entropy using a gradient descent algorithm with a learning rate of 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init) # initializing all the variables

num_iterations = 1000

for i in xrange(num_iterations):
    batch_xs, batch_ys = mnist.train.next_batch(100) # we get a batch of one hundred random data points
    sess.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict = {x: mnist.test.images, y_:mnist.test.labels}))

# note that the cross_entropy with a perfect classifier would be zero
print(sess.run(cross_entropy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels}))