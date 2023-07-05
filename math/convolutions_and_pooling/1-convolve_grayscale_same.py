#!/usr/bin/env python3
"""Imports"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    During a same convolution, the kernel is placed on the input
    image, and the convolution operation is computed over each
    position of the input. To ensure that the output has the same
    size as the input, the input image is padded with zeros
    symmetrically around its boundaries. The padding allows
    the kernel to be centered on every pixel of the input,
    even at the edges. As a result, the output of a same
    convolution has the same size as the input image,
    preserving its spatial dimensions.
    """

    m, h, w = images.shape
    kh, kw = kernel.shape

    H_PAD = kh // 2
    W_PAD = kw // 2

    PADDED = np.pad(images, ((0, 0), (H_PAD, H_PAD), (W_PAD, W_PAD)), mode='constant')

    CONVO = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            img = PADDED[:, i:i+kh, j:j+kw]
            CONVO[:, i, j] = np.sum(img * kernel, axis=(1, 2))

    return CONVO
