#!/usr/bin/env python3
"""Imports"""
import tensorflow as tf


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

    conv1 = tf.layers.Conv2D(filters=6,
                             kernel_size=(5, 5),
                             padding='same',
                             activation=tf.nn.relu,
                             kernel_initializer=pycode)(x)

    pool1 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                   strides=(2, 2))(conv1)

    conv2 = tf.layers.Conv2D(filters=16,
                             kernel_size=(5, 5),
                             padding='valid',
                             activation=tf.nn.relu,
                             kernel_initializer=pycode)(pool1)

    pool2 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                   strides=(2, 2))(conv2)

    flatten = tf.layers.Flatten()(pool2)

    fc1 = tf.layers.Dense(units=120,
                          activation=tf.nn.relu,
                          kernel_initializer=pycode)(flatten)

    fc2 = tf.layers.Dense(units=84,
                          activation=tf.nn.relu,
                          kernel_initializer=pycode)(fc1)

    y_pred = tf.layers.Dense(units=10,
                             kernel_initializer=pycode)(fc2)

    loss = tf.losses.softmax_cross_entropy(y, y_pred)

    train_op = tf.train.AdamOptimizer().minimize(loss)

    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, 1),
                                          tf.argmax(y, 1)), tf.float32))

    output = tf.nn.softmax(y_pred)

    return output, train_op, loss, acc
