#!/usr/bin/env python3
"""Imports"""
import tensorflow.keras as K


def lenet5(X):
    """
    The lenet5 function takes X as input, which is a Keras
    Input object representing the input images for the
    network. It builds a modified version of the LeNet-5
    architecture using Keras, adhering to the specified
    layers and configurations. The function returns a
    compiled Keras Model that utilizes Adam optimization
    with default hyperparameters and accuracy metrics
    for training and evaluation purposes.
    """

    conv1 = K.layers.Conv2D(filters=6,
                            kernel_size=(5, 5),
                            padding='same',
                            activation='relu',
                            kernel_initializer='he_normal')(X)

    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                  strides=(2, 2))(conv1)

    conv2 = K.layers.Conv2D(filters=16,
                            kernel_size=(5, 5),
                            padding='valid',
                            activation='relu',
                            kernel_initializer='he_normal')(pool1)

    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                  strides=(2, 2))(conv2)

    flatten = K.layers.Flatten()(pool2)

    fc1 = K.layers.Dense(units=120,
                         activation='relu',
                         kernel_initializer='he_normal')(flatten)

    fc2 = K.layers.Dense(units=84,
                         activation='relu',
                         kernel_initializer='he_normal')(fc1)

    output = K.layers.Dense(units=10,
                            activation='softmax')(fc2)

    model = K.Model(inputs=X, outputs=output)

    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
