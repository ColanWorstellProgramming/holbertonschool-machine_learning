#!/usr/bin/env python3
"""Imports"""
import numpy as np


def precision(confusion):
    """
    Precision measures the proportion of predicted
    positive cases that are truly positive. It is
    calculated as TP / (TP + FP).
    """

    TP = np.diag(confusion)
    P = np.sum(confusion, axis=0)

    FP = P - TP

    return TP / (TP + FP)
