#!/usr/bin/env python3
"""
Imports
"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calculates the most likely sequence of hidden states for a hidden markov model
    """
    