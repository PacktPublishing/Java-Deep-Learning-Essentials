# -*- coding: utf-8 -*-

import input_data
import tensorflow as tf

#
# Set up
#

# import MNIST data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# define W, b
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# define placeholder for inputs
x = tf.placeholder("float", [None, 784])

# define outputs using the softmax function
y = tf.nn.softmax(tf.matmul(x, W) + b)


# define placeholder for label
y_ = tf.placeholder("float", [None, 10])

# cross entropy function
cross_entropy = - tf.reduce_sum(y_ * tf.log(y))

# apply Gradient Descent algorithm
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


#
# TensorFlow Session
#

# initialize tensorflow session
sess = tf.Session()

# initialize 
init = tf.initialize_all_variables()
sess.run(init)

for i in range(1000):
    # minibatch of MNIST
    batch_xs, batch_ys = mnist.train.next_batch(100)

    # train
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


#
# show Result
#

# predict
correct_prediction = tf.equal(
    tf.argmax(y, 1), tf.argmax(y_, 1)
)

# accuracy
accuracy = tf.reduce_mean(
    tf.cast(correct_prediction, "float")
)

print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
