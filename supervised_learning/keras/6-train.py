#!/usr/bin/env python3
"""Imports"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None,
                early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """
    Training the model using keras

    This time we are checking for early stopping
    """
    if validation_data is True and early_stopping is True:
        EARLY = K.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                             patience=patience)
        return network.fit(x=data,
                           y=labels,
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=verbose,
                           shuffle=shuffle,
                           validation_data=validation_data,
                           callbacks=[EARLY])
    else:
        return network.fit(x=data,
                           y=labels,
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=verbose,
                           shuffle=shuffle,
                           validation_data=validation_data)
