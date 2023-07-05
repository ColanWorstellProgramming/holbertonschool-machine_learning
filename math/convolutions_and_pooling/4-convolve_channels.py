#!/usr/bin/env python3
"""Imports"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    During the convolution with custom padding, the kernel is placed
    on the input image, and the convolution operation is computed over
    each position of the input. To accommodate custom padding requirements,
    the input image is padded with zeros symmetrically around its boundaries.
    This padding ensures that the kernel can be centered on every pixel of
    the input, even at the edges. The resulting output of the convolution
    has the same size as the input image, preserving its spatial dimensions,
    while incorporating the specified custom padding. In the case of images
    with multiple channels, the convolution is performed independently on
    each channel, and the resulting convolved images maintain their original
    number of channels.
    """

    m, h, w, _ = images.shape
    kh, kw, _ = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = (((h - 1) * sh) + kh - h) // 2 + 1
        pw = (((w - 1) * sw) + kw - w) // 2 + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    images = np.pad(images, ((0, 0), (ph, ph),
                             (pw, pw), ((0, 0))))

    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1

    CONVO = np.zeros((m, oh, ow))

    for i in range(oh):
        for j in range(ow):
            CONVO[:, i, j] = np.sum(images[:, i * sh:i * sh + kh,
                                           j * sw:j * sw + kw]
                                    * kernel, axis=(1, 2, 3))

    return CONVO
