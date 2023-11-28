#!/usr/bin/env python3
"""
Imports
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    tf idf
    """
    vectorizer = TfidfVectorizer(vocabulary=vocab)

    embeddings = vectorizer.fit_transform(sentences).toarray()

    if vocab is None:
        features = sorted(list(set(vectorizer.get_feature_names_out())))

    else:
        features = vocab

    return embeddings, features
