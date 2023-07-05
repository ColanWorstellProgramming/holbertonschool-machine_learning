#!/usr/bin/env python3
"""Imports"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):

    """
    During pooling, a kernel of specified shape is applied
    to the input images. The kernel is moved over the input
    image with a specified stride, and at each position, a
    pooling operation is performed. The pooling operation
    reduces the size of the input region within the kernel
    to a single value, based on the specified pooling mode.
    The resulting output has reduced spatial dimensions
    compared to the input image. In the case of images
    with multiple channels, the pooling operation is performed
    independently on each channel, and the resulting pooled
    images maintain their original number of channels.
    """

    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    oh = (h - kh) // sh + 1
    ow = (w - kw) // sw + 1

    POOL = np.zeros((m, oh, ow, c))

    for i in range(oh):
        for j in range(ow):
            img = images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]

            if mode == 'max':
                POOL[:, i, j, :] = np.max(img, axis=(1, 2))
            elif mode == 'avg':
                POOL[:, i, j, :] = np.mean(img, axis=(1, 2))

    return POOL
