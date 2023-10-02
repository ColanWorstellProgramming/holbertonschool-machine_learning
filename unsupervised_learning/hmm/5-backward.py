#!/usr/bin/env python3
"""
Imports
"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a hidden markov model
    """
    T = len(Observation)
    N, _ = Emission.shape

    if Initial.shape[0] != N:
        return None, None

    if Transition.shape[0] != N or Transition.shape[1] != N:
        return None, None

    B = np.zeros((N, T))

    B[:, -1] = 1

    for t in range(T - 2, -1, -1):
        for i in range(N):
            for j in range(N):
                B[i, t] += Transition[i, j] * Emission[j, Observation[t + 1]] * B[j, t + 1]

    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])

    return P, B
