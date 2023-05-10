#!/usr/bin/env python3
"""Line Up Function"""


def add_arrays(arr1, arr2):
    """Add An Array Function"""
    arr3 = []

    if len(arr1) == len(arr2):
        for i in range(len(arr1)):
            arr3.append(arr1[i] + arr2[i])
        return arr3
    else:
        return None
