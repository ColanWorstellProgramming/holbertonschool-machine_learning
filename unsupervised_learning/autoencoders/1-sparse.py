#!/usr/bin/env python3
"""
Imports
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Creates a sparse autoencoder
    """
    input_img = keras.Input(shape=(input_dims,))
    output_img = input_img

    for encoding_dim in hidden_layers:
        output_img = keras.layers.Dense(encoding_dim,
                                        activation='relu')(output_img)

    lam = keras.regularizers.l1(lambtha)

    latent_layer = keras.layers.Dense(latent_dims,
                                      activation='relu',
                                      activity_regularizer=lam)(output_img)
    encoder = keras.models.Model(input_img, latent_layer)

    # Decoder
    decoder_input = keras.Input(shape=(latent_dims,))
    decoder_output = decoder_input

    for encoding_dim in reversed(hidden_layers):
        decoder_output = keras.layers.Dense(encoding_dim,
                                            activation='relu')(decoder_output)

    output_layer = keras.layers.Dense(input_dims,
                                      activation='sigmoid')(decoder_output)
    decoder = keras.models.Model(decoder_input, output_layer)

    # Autoencoder
    autoencoder_input = keras.Input(shape=(input_dims,))
    autoencoder_output = decoder(encoder(autoencoder_input))
    autoencoder = keras.models.Model(autoencoder_input,
                                     autoencoder_output)

    # Compile
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder