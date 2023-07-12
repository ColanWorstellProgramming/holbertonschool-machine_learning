#!/usr/bin/env python3
"""Imports"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    During the convolution operation in a neural network with
    custom padding, the kernel is applied to the input image
    by computing the convolution at each position. To meet
    the requirements of custom padding, the input image is
    symmetrically padded with zeros around its boundaries.
    This padding ensures that the kernel can be centered on
    every pixel of the input, even at the edges. The output
    of the convolution operation maintains the same size as
    the input image, preserving its spatial dimensions while
    incorporating the specified custom padding. If the input
    image has multiple channels, the convolution is
    independently performed on each channel, and the resulting
    convolved images maintain their original number of channels.
    """
    m, h, w, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h = (h - kh) // sh + 1
    w = (w - kw) // sw + 1

    CONVO = np.zeros((m, h, w, c))

    for i in range(h):
        for j in range(w):
            VER_START = i * sh
            VER_END = VER_START + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            a_prev_slice = A_prev[:, VER_START:VER_END,
                                  horiz_start:horiz_end, :]

            if mode == 'max':
                CONVO[:, i, j, :] = np.max(a_prev_slice, axis=(1, 2))
            elif mode == 'avg':
                CONVO[:, i, j, :] = np.mean(a_prev_slice, axis=(1, 2))

    return CONVO
