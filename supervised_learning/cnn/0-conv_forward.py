#!/usr/bin/env python3
"""Imports"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    During the convolution with custom padding, the kernel
    is placed on the input image, and the convolution operation
    is computed over each position of the input. To accommodate
    custom padding requirements, the input image is padded with
    zeros symmetrically around its boundaries. This padding
    ensures that the kernel can be centered on every pixel of
    the input, even at the edges. The resulting output of the
    convolution has the same size as the input image, preserving
    its spatial dimensions, while incorporating the specified
    custom padding. In the case of images with multiple channels,
    the convolution is performed independently on each channel,
    and the resulting convolved images maintain their original
    number of channels.
    """
    m, h, w, _ = A_prev.shape
    kh, kw, _, c = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = (((h - 1) * sh) + kh - h) // 2
        pw = (((w - 1) * sw) + kw - w) // 2 
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    A_prev = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)))

    h_out = (h + 2 * ph - kh) // sh + 1
    w_out = (w + 2 * pw - kw) // sw + 1

    CONVO = np.zeros((m, h_out, w_out, c))

    for i in range(h_out):
        for j in range(w_out):
            for k in range(c):
                a_slice_prev = A_prev[:, i * sh:i *
                                      sh + kh, j *
                                      sw:j * sw + kw, :]
                CONVO[:, i, j, k] = np.sum(a_slice_prev *
                                           W[:, :, :, k],
                                           axis=(1, 2, 3))

    CONVO = CONVO + b

    if activation is not None:
        A = activation(CONVO)
    else:
        A = CONVO

    return A
