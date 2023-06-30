#!/usr/bin/env python3
"""Imports"""
import numpy as np


def sensitivity(confusion):
    """
     Sensitivity measures the proportion of actual positive
     cases that are correctly identified by the model. It is
     calculated as TP / (TP + FN).
    """

    TP = np.diag(confusion)
    P = np.sum(confusion, axis=1)

    FN = P - TP

    return TP / (TP + FN)
