#!/usr/bin/env python3
"""
Imports
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder
    """
    # Encoder
    input_img = keras.Input(shape=input_dims)
    x = input_img

    # Conv layers
    for filter_count in filters:
        x = keras.layers.Conv2D(filter_count,
                                (3, 3),
                                activation='relu',
                                padding='same')(x)

        x = keras.layers.MaxPooling2D((2, 2),
                                      padding='same')(x)

    encoder = keras.models.Model(input_img, x)

    # Decoder
    latent_input = keras.Input(shape=latent_dims)
    y = latent_input

    # Add Convolutional Layers In Reverse Order
    for filter_count in reversed(filters[:-1]):
        y = keras.layers.Conv2D(filter_count,
                                (3, 3),
                                activation='relu',
                                padding='same')(y)
        y = keras.layers.UpSampling2D((2, 2))(y)

    y = keras.layers.Conv2D(filters[0],
                            (3, 3),
                            activation='sigmoid',
                            padding='valid')(y)

    y = keras.layers.UpSampling2D((2, 2))(y)

    y = keras.layers.Conv2D(input_dims[-1],
                            (3, 3),
                            activation='sigmoid',
                            padding='same')(y)

    decoder = keras.models.Model(latent_input, y)

    # Autoencoder
    autoencoder_input = keras.Input(shape=input_dims)
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    autoencoder = keras.models.Model(autoencoder_input, decoded)

    # Compile
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
