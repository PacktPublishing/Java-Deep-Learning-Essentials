# -*- coding: utf-8 -*-

import input_data
import tensorflow as tf


def inference(x_placeholder):

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.nn.softmax(tf.matmul(x_placeholder, W) + b)

    return y


def loss(y, label_placeholder):

    cross_entropy = - tf.reduce_sum(label_placeholder * tf.log(y))

    return cross_entropy


def training(loss):

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    return train_step



def res(y, label_placeholder, feed_dict):

    correct_prediction = tf.equal(
        tf.argmax(y, 1), tf.argmax(label_placeholder, 1)
    )
    
    accuracy = tf.reduce_mean(
        tf.cast(correct_prediction, "float")
    )

    # 
    # eval predictions
    #

    # prediction = tf.argmax(y, 1)
    # print prediction.eval(feed_dict=feed_dict)

    # prediction = tf.argmax(label_placeholder, 1)
    # print prediction.eval(feed_dict=feed_dict)
    
    print sess.run(accuracy, feed_dict=feed_dict)
    


if __name__ == "__main__":

    # import MNIST data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


    x_placeholder = tf.placeholder("float", [None, 784])
    label_placeholder = tf.placeholder("float", [None, 10])


    with tf.Session() as sess:

        y = inference(x_placeholder)
        loss = loss(y, label_placeholder)
        train_step = training(loss)

        init = tf.initialize_all_variables()
        sess.run(init)

        for i in range(1000):

            batch_xs, batch_ys = mnist.train.next_batch(100)

            feed_dict = {x_placeholder: batch_xs, label_placeholder: batch_ys}

            sess.run(train_step, feed_dict=feed_dict)


            if i % 100 == 0:
                print 'iter=%d' %(i)
                res(y, label_placeholder, feed_dict)
                print

        # show result
        print 'Reuslt:'
        res(y, label_placeholder, feed_dict)
