#!/usr/bin/env python3
"""Imports + Required From Problem"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    The F1 score is a measure of a
    model's accuracy that considers
    both precision and recall. It is
    the harmonic mean of precision and
    recall and provides a balanced evaluation
    metric. The F1 score is calculated as
    2 * (Precision * Recall) / (Precision + Recall).
    """

    Precision = precision(confusion)
    Recall = sensitivity(confusion)

    F1 = 2 * (Precision * Recall) / (Precision + Recall + np.finfo(float).eps)

    return F1
