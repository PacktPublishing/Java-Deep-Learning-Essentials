# -*- coding: utf-8 -*-

import input_data
import tensorflow as tf


def inference(x_placeholder):

    with tf.name_scope('inference') as scope:
        W = tf.Variable(tf.zeros([784, 10]), name="W")
        b = tf.Variable(tf.zeros([10]), name="b")

        y = tf.nn.softmax(tf.matmul(x_placeholder, W) + b)

    return y


def loss(y, label_placeholder):

    with tf.name_scope('loss') as scope:
        cross_entropy = - tf.reduce_sum(label_placeholder * tf.log(y))

        tf.scalar_summary("Cross Entropy", cross_entropy)

    return cross_entropy


def training(loss):

    with tf.name_scope('training') as scope:
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    return train_step



def res(y, label_placeholder, feed_dict):

    correct_prediction = tf.equal(
        tf.argmax(y, 1), tf.argmax(label_placeholder, 1)
    )
    
    accuracy = tf.reduce_mean(
        tf.cast(correct_prediction, "float")
    )
    
    print sess.run(accuracy, feed_dict=feed_dict)
    


if __name__ == "__main__":

    # import MNIST data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


    with tf.Graph().as_default():

        x_placeholder = tf.placeholder("float", [None, 784], name="input")
        label_placeholder = tf.placeholder("float", [None, 10], name="label")

        with tf.Session() as sess:

            y = inference(x_placeholder)
            loss = loss(y, label_placeholder)
            train_step = training(loss)


            summary_step = tf.merge_all_summaries()
            init = tf.initialize_all_variables()

            summary_writer = tf.train.SummaryWriter('data', graph_def=sess.graph_def)
            sess.run(init)

            for i in range(1000):

                batch_xs, batch_ys = mnist.train.next_batch(100)

                feed_dict = {x_placeholder: batch_xs, label_placeholder: batch_ys}

                sess.run(train_step, feed_dict=feed_dict)


                if i % 100 == 0:
                    # print 'iter=%d' %(i)
                    # res(y, label_placeholder, feed_dict)
                    # print
                    summary = sess.run(summary_step, feed_dict=feed_dict)
                    summary_writer.add_summary(summary, i)

            # show result
            print 'Reuslt:'
            res(y, label_placeholder, feed_dict)
