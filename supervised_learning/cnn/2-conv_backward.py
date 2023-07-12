#!/usr/bin/env python3
"""Imports"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    During backpropagation over a convolutional layer in a neural
    network, the partial derivatives with respect to the unactivated
    output of the convolutional layer are computed. Custom padding
    is used to ensure that the kernel can be centered on every pixel
    of the input image, even at the edges. The input image is
    symmetrically padded with zeros around its boundaries to
    accommodate this requirement. The resulting gradients are
    calculated independently for each channel of the input
    image, and the spatial dimensions of the gradients match
    the size of the input image, preserving its original
    dimensions.
    """

    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, _ = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = (((h_prev - 1) * sh) + kh - h_prev) // 2 + 1
        pw = (((w_prev - 1) * sw) + kw - w_prev) // 2 + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    A_prev = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)))
    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    for pic in range(0, m):
        for h in range(0, h_new):
            for w in range(0, w_new):
                for c in range(0, c_new):
                    dA_prev[pic, sh*h:sh*h+kh, sw*w:sw*w+kw, :] += \
                        dZ[pic, h, w, c] * W[:, :, :, c]
                    dW[:, :, :, c] += dZ[pic, h, w, c] * \
                        A_prev[pic, sh*h:sh*h+kh, sw*w:sw*w+kw, :]
                    db[:, :, :, c] += dZ[pic, h, w, c]
    dA_prev = dA_prev[:, ph:ph+h_prev, pw:pw+w_prev, :]

    return dA_prev, dW, db
