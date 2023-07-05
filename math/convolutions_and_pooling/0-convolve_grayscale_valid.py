#!/usr/bin/env python3
"""Imports"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    During a valid convolution, the kernel is placed on
    the input image, and the convolution operation is
    computed only where the entire kernel can be centered
    on the input without exceeding its boundaries. As a
    result, the output of a valid convolution is smaller
    in size compared to the input image.
    """

    m, h, w = images.shape
    kh, kw = kernel.shape

    H = h - kh + 1
    W = w - kw + 1

    img = np.zeros((m, H, W))

    for i in range(H):
        for j in range(W):
            p = images[:, i:i+kh, j:j+kw]
            img[:, i, j] = np.sum(p * kernel, axis=(1, 2))

    return img
