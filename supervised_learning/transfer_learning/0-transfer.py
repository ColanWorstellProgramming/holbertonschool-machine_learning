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

    base_model = K.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    unfreeze_layers = 100
    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False
    for layer in base_model.layers[-unfreeze_layers:]:
        layer.trainable = True

    model = K.Sequential([
        K.layers.Lambda(lambda x: tf.image.resize(x, (128, 128))),
        base_model,
        K.layers.GlobalAveragePooling2D(),
        K.layers.Dense(256, activation='relu'),
        K.layers.Dropout(0.5),
        K.layers.Dense(10, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer=K.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    lr_scheduler = K.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    early_stopping = K.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(
        train_datagen.flow(X_train, y_train, batch_size=64),
        epochs=7,
        validation_data=validation_datagen.flow(X_test, y_test, batch_size=64),
        callbacks=[lr_scheduler, early_stopping],
    )

    model.save("cifar10.h5")

if __name__ == "__main__":
    main()
