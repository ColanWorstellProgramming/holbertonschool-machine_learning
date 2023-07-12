#!/usr/bin/env python3
"""Imports"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    During backpropagation over a pooling layer in a neural network, the
    partial derivatives with respect to the output of the pooling layer are
    computed. The gradients are calculated independently for each channel of
    the input image, and the spatial dimensions of the gradients match the
    size of the input image. The pooling operation used during forward
    propagation (max or average pooling) is reversed during backpropagation
    to distribute the gradients to the appropriate positions in the previous
    layer based on the kernel shape and stride.
    """

    m, h_new, w_new, c = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for ch in range(c):
                    VER_START = h * sh
                    VER_END = VER_START + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    if mode == 'max':
                        mask = (
                            A_prev[i, VER_START:VER_END,
                                   horiz_start:horiz_end, ch]
                            == np.max(A_prev[i, VER_START:VER_END,
                                             horiz_start:horiz_end, ch])
                        )
                        dA_prev[i, VER_START:VER_END,
                                horiz_start:horiz_end, ch] += \
                            mask * dA[i, h, w, ch]
                    elif mode == 'avg':
                        dA_prev[i, VER_START:VER_END,
                                horiz_start:horiz_end, ch] += \
                            dA[i, h, w, ch] / (kh * kw)

    return dA_prev
