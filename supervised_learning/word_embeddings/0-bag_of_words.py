#!/usr/bin/env python3
"""
Imports
"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    bag o words
    """
    vectorizer = CountVectorizer(vocabulary=vocab)

    embeddings = vectorizer.fit_transform(sentences).toarray()

    if vocab is None:
        features = sorted(list(set(vectorizer.get_feature_names_out())))

    else:
        features = vocab

    return embeddings, features
