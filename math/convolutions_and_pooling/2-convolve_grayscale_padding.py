#!/usr/bin/env python3
"""Imports"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    During the convolution with custom padding, the kernel
    is placed on the input image, and the convolution
    operation is computed over each position of the
    input. To accommodate custom padding requirements,
    the input image is padded with zeros symmetrically
    around its boundaries. This padding ensures that the
    kernel can be centered on every pixel of the input,
    even at the edges. The resulting output of the convolution
    has the same size as the input image, preserving its
    spatial dimensions, while incorporating the specified
    custom padding.
    """

    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    PADDED = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                    mode='constant')

    CONVO = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            img = PADDED[:, i:i+kh, j:j+kw]
            CONVO[:, i, j] = np.sum(img * kernel, axis=(1, 2))

    return CONVO
