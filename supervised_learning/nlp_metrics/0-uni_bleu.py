#!/usr/bin/env python3
"""
Imports
"""
import numpy as np
from collections import Counter


def uni_bleu(references, sentence):
    """
    uni bleu
    """
    candidate_counts = {word: min([ref.count(word) for ref in references], default=0) for word in sentence}
    total_candidate_count = sum(candidate_counts.values())
    precision = total_candidate_count / len(sentence) if len(sentence) > 0 else 0

    closest_length = min(len(ref) for ref in references)
    brevity_penalty = np.exp(1 - (closest_length / len(sentence))) if len(sentence) < closest_length else 1

    bleu = brevity_penalty * precision

    return bleu
