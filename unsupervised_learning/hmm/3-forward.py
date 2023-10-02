#!/usr/bin/env python3
"""
Imports
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a hidden markov model
    """
    T = Observation.shape[0]
    N, _ = Emission.shape

    if Initial.shape[0] != N:
        return None, None

    if Transition.shape[0] != N or Transition.shape[1] != N:
        return None, None

    F = np.zeros((N, T))

    for i in range(N):
        F[i, 0] = Initial[i] * Emission[i, Observation[0]]

    for t in range(1, T):
        for j in range(N):
            for i in range(N):
                Py = Emission[j, Observation[t]]
                F[j, t] += F[i, t - 1] * Transition[i, j] * Py

    P = np.sum(F[:, -1])

    return P, F
