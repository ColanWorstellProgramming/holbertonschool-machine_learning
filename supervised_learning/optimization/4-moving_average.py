#!/usr/bin/env python3
"""Imports"""
import tensorflow as tf


def moving_average(data, beta):
    """Moving Average"""
    x = 0
    x_List = []
    for y in range(len(data)):
        x = ((x * beta) + ((1 - beta) * data[y]))
        b = x / (1 - (beta ** (y + 1)))
        x_List.append(b)
    return x_List
