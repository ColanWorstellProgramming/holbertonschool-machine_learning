#!/usr/bin/env python3
"""Imports"""
import numpy as np
import tensorflow.keras as K
import tensorflow as tf


def preprocess_data(X, Y):
    """Preprocess Data"""
    X_p = X.astype('float32') / 255.0
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p

def main():
    """Main Function"""
    (X_train, y_train), (X_test, y_test) = K.datasets.cifar10.load_data()

    X_train, y_train = preprocess_data(X_train, y_train)
    X_test, y_test = preprocess_data(X_test, y_test)

    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    pre_trained_model = K.applications.VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    for layer in pre_trained_model.layers:
        layer.trainable = False

    model = K.Sequential([
        pre_trained_model,
        K.layers.Flatten(),
        K.layers.Dense(256, activation='relu'),
        K.layers.Dropout(0.5),
        K.layers.Dense(10, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer=K.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    datagen.fit(X_train)

    num_samples = X_train.shape[0]
    validation_split = 0.1
    validation_samples = int(num_samples * validation_split)
    train_samples = num_samples - validation_samples

    model.fit(
        datagen.flow(X_train[:train_samples], y_train[:train_samples], batch_size=64),
        epochs=5,
        validation_data=datagen.flow(X_train[train_samples:], y_train[train_samples:], batch_size=64),
    )

    model.save("cifar10_with_transfer_learning.h5")

if __name__ == "__main__":
    main()
