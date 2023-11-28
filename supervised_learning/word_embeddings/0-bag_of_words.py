#!/usr/bin/env python3
"""
Imports
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    bag of words
    """
    vectorizer = CountVectorizer(vocabulary=vocab)

    embeddings = vectorizer.fit_transform(sentences).toarray()

    features = vectorizer.get_feature_names_out()

    return embeddings, features
