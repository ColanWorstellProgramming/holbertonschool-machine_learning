#!/usr/bin/env python3
"""
Imports
"""
from keras.layers import Embedding


def gensim_to_keras(model):
    """
    gensim to keras
    """
    vocab_size, embedding_dim = model.wv.vectors.shape

    embedding_matrix = model.wv.vectors

    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=True
    )

    return embedding_layer
