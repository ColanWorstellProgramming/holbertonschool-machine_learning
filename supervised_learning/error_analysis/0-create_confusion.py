#!/usr/bin/env python3
"""Imports"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    A confusion matrix is a table that represents
    the performance of a classification model by
    comparing its predicted labels against the actual
    labels of a dataset.
    """

    l1 = labels.shape[1]
    l2 = logits.shape[1]

    matrix = np.zeros((l1, l2))

    prediction = np.argmax(logits, axis=1)

    actual = np.argmax(labels, axis=1)

    for i in range(len(actual)):
        a = actual[i]
        predicted = prediction[i]
        matrix[a][predicted] += 1

    return matrix
