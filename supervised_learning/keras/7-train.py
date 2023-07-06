#!/usr/bin/env python3
"""Imports"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None,
                early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """
    Training the model using keras

    This time we add learning rate decay
    """
    list = []

    if validation_data is True and early_stopping is True:
        list.append(K.callbacks.EarlyStopping(monitor='val_loss',
                                              mode='min',
                                              atience=patience))

    if validation_data is True and learning_rate_decay is True:
        def scheduler(epoch):
            """Scheduler"""
            return (alpha / (1 + decay_rate * epoch))
        list.append(K.callbacks.LearningRateScheduler(scheduler,
                                                      verbose=1))

    return network.fit(x=data,
                       y=labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=list)
