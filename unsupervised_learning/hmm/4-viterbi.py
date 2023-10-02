#!/usr/bin/env python3
"""
Imports
"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calculates the most likely sequence of hidden
    states for a hidden markov model
    """
    T = len(Observation)
    N, _ = Emission.shape

    if Initial.shape[0] != N:
        return None, None

    if Transition.shape[0] != N or Transition.shape[1] != N:
        return None, None

    path = []
    V = np.zeros((N, T))

    for i in range(N):
        V[i, 0] = Initial[i] * Emission[i, Observation[0]]

    backpointer = np.zeros((N, T - 1), dtype=int)

    for t in range(1, T):
        for j in range(N):
            max_prob = -1
            best_state = -1

            for i in range(N):
                Py = Emission[j, Observation[t]]
                prob = V[i, t - 1] * Transition[i, j] * Py
                if prob > max_prob:
                    max_prob = prob
                    best_state = i

            V[j, t] = max_prob
            backpointer[j, t - 1] = best_state

    best_final_state = np.argmax(V[:, -1])
    path.append(best_final_state)

    for t in range(T - 2, -1, -1):
        best_final_state = backpointer[best_final_state, t]
        path.insert(0, best_final_state)

    P = np.max(V[:, -1])

    return path, P
