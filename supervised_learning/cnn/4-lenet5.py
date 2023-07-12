#!/usr/bin/env python3
"""Imports"""
import numpy as np


def lenet5(x, y):
    """
    The lenet5 function takes x and y as input, which are
    TensorFlow placeholders for input images and labels,
    respectively. It builds the LeNet-5 architecture following
    the specified layers and configurations. The function
    returns the softmax activated output tensor, the training
    operation that utilizes Adam optimization, the loss
    tensor, and the accuracy tensor for evaluating the network.
    """
    pycode = tf.contrib.layers.variance_scaling_initializer()
    soft = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)

    conv1 = tf.layers.conv2d(inputs=x,
                             filters=6,
                             kernel_size=(5, 5),
                             padding='same',
                             activation=tf.nn.relu,
                             kernel_initializer=pycode)

    pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=(2, 2),
                                    strides=(2, 2))

    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=16,
                             kernel_size=(5, 5),
                             padding='valid',
                             activation=tf.nn.relu,
                             kernel_initializer=pycode)

    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=(2, 2),
                                    strides=(2, 2))

    flatten = tf.layers.flatten(pool2)

    fc1 = tf.layers.dense(inputs=flatten,
                          units=120,
                          activation=tf.nn.relu,
                          kernel_initializer=pycode)

    fc2 = tf.layers.dense(inputs=fc1, units=84,
                          activation=tf.nn.relu,
                          kernel_initializer=pycode)

    output = tf.layers.dense(inputs=fc2,
                             units=10,
                             activation=tf.nn.softmax)

    loss = tf.reduce_mean(soft)

    correct_prediction = tf.equal(tf.argmax(output, 1),
                                  tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                      tf.float32))

    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)

    return output, train_op, loss, accuracy
