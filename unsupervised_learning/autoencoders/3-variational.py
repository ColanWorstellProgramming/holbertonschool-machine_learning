#!/usr/bin/env python3
"""
Imports
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder
    """

    def sampling(args):
        """
        sampling
        """
        z_mean, z_log_var = args
        shp = keras.backend.shape(z_mean)
        epsilon = keras.backend.random_normal(shape=shp)
        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon

    input_img = keras.Input(shape=(input_dims,))
    x = input_img

    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)

    z_mean = keras.layers.Dense(latent_dims, activation=None,
                                name="z_mean")(x)
    z_log_var = keras.layers.Dense(latent_dims, activation=None,
                                   name="z_log_var")(x)

    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,),
                            name="z")([z_mean, z_log_var])

    encoder = keras.models.Model(input_img, [z, z_mean, z_log_var])

    # Decoder
    latent_input = keras.layers.Input(shape=(latent_dims,))
    y = latent_input

    for units in reversed(hidden_layers):
        y = keras.layers.Dense(units, activation='relu')(y)

    output = keras.layers.Dense(input_dims, activation='sigmoid')(y)

    decoder = keras.models.Model(latent_input, output)

    # Variational Autoencoder
    autoencoder_input = keras.layers.Input(shape=(input_dims,))
    encoded, z_mean, z_log_var = encoder(autoencoder_input)
    decoded = decoder(encoded)
    auto = keras.models.Model(autoencoder_input, decoded)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
