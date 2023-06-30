#!/usr/bin/env python3
"""No Imports"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    A function utilizing Early Stopping Regularization, The Easiest
    One To Implement
    """
    if (opt_cost - cost) > threshold:
        count = 0
    else:
        count = count + 1

    if count >= patience:
        return (True, count)
    else:
        return(False, count)
