#!/usr/bin/env python3
"""
Imports
"""
import numpy as np


def absorbing(P):
    """
    Determines if a markov chain is absorbing
    """
    # n is the number of states in the markov chain
    n = P.shape[0]

    # make sure P is a square
    if P.shape[0] != P.shape[1]:
        return False

    # check to see if any rows are all 0's
    has_absorbing_states = np.any(np.diag(P) == 1)

    if not has_absorbing_states:
        return False

    # see if it is in canonical form  / absorbing states in the top-left corner
    absorbing_state_indices = np.where(np.diag(P) == 1)[0]
    transient_state_indices = np.setdiff1d(np.arange(n), absorbing_state_indices)

    if not np.all(transient_state_indices > absorbing_state_indices[-1]):
        return False

    return True
