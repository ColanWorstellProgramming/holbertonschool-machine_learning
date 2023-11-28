#!/usr/bin/env python3
"""
Imports
"""
import numpy as np
from collections import Counter


def cumulative_bleu(references, sentence, n):
    """
    cumulative bleu
    """
    def calculate_precision(candidate, references, n):
        """
        sub func for calculating precision
        """
        candidate_ngrams = Counter(zip(*[sentence[i:] for i in range(n)]))
        mrn = Counter()

        for ref in references:
            reference_ngrams = Counter(zip(*[ref[i:] for i in range(n)]))
            mrn += reference_ngrams

        cc = {ngram: min(candidate_ngrams[ngram],
                         mrn[ngram]) for ngram in candidate_ngrams}

        precision = sum(cc.values()) / max(1, sum(candidate_ngrams.values()))

        return precision

    bleu = 1.0

    for i in range(1, n + 1):
        precision_i = calculate_precision(sentence, references, i)
        bleu *= precision_i

    cl = min(len(ref) for ref in references)
    bp = np.exp(1 - (cl / len(sentence))) if len(sentence) < cl else 1

    bleu = bp * bleu ** (1/n)

    return bleu
