#!/usr/bin/env python3
"""
Imports
"""
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm for a hidden markov model
    """
    M = Transition.shape[0]
    T = len(Observations)

    for _ in range(iterations):
        P_forward, alpha = forward(Observations, Emission,
                                   Transition, Initial)
        P_backward, beta = backward(Observations, Emission,
                                    Transition, Initial)

        if P_forward is None or P_backward is None:
            return None, None

        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            denominator = np.dot(np.dot(alpha[:, t].T, Transition) *
                                 Emission[:, Observations[t + 1]].T,
                                 beta[:, t + 1])
            for i in range(M):
                Py = Emission[:, Observations[t + 1]].T * beta[:, t + 1]
                numerator = alpha[i, t] * Transition[i, :] * Py
                xi[i, :, t] = numerator / denominator

        gamma = np.sum(xi, axis=1)
        Transition = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2],
                                         axis=0).reshape((-1, 1))))

        K = Emission.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            Emission[:, l] = np.sum(gamma[:, Observations == l], axis=1)

        Emission = np.divide(Emission, denominator.reshape((-1, 1)))

    return Transition, Emission


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
                Py = Emission[j, Observation[t + 1]] * B[j, t + 1]
                B[i, t] += Transition[i, j] * Py

    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])

    return P, B
