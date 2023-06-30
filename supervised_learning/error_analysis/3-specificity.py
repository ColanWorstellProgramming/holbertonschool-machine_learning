#!/usr/bin/env python3
"""Imports"""
import numpy as np


def specificity(confusion):
    """
    Specificity measures the proportion of
    actual negative cases that are correctly
    identified by the model. It is calculated
    as TN / (TN + FP).
    """

    TP = np.diag(confusion)
    P = np.sum(confusion, axis=0)
    N = np.sum(confusion, axis=1)

    FP = P - TP
    FN = N - TP
    TN = np.sum(confusion) - (TP + FP + FN)

    return TN / (TN + FP)
