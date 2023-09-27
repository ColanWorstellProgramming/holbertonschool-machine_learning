#!/usr/bin/env python3
"""
Imports
"""
import numpy as np
import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means on a dataset
    """
    centroids, cluster_assignments, _ = sklearn.cluster.k_means(X, k)

    return centroids, cluster_assignments
