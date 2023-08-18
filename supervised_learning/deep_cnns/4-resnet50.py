#!/usr/bin/env python3
"""Imports"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Building a ResNet-50 architecture
    """
    he_normal = K.initializers.he_normal()

    inputs = K.Input(shape=(224, 224, 3))

    x = K.layers.Conv2D(64,
                        (7, 7),
                        strides=(2, 2),
                        padding='same',
                        kernel_initializer=he_normal)(inputs)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)
    x = K.layers.MaxPooling2D((3, 3),
                              strides=(2, 2),
                              padding='same')(x)

    x = projection_block(x, [64, 64, 256], s=1)
    x = identity_block(x, [64, 64, 256])
    x = identity_block(x, [64, 64, 256])

    x = projection_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])

    x = projection_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])

    x = projection_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])

    x = K.layers.AveragePooling2D(pool_size=(2, 2),
                                  strides=(7, 7),
                                  padding='valid')(x)

    outputs = K.layers.Dense(1000,
                             activation='softmax',
                             kernel_initializer=he_normal)(x)

    model = K.Model(inputs, outputs)

    return model