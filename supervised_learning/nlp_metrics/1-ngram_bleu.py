#!/usr/bin/env python3
"""
Imports
"""
import numpy as np
from collections import Counter


def ngram_bleu(references, sentence, n):
    """
    ngram bleu
    """
    candidate_ngrams = Counter(zip(*[sentence[i:] for i in range(n)]))
    max_reference_ngrams = Counter()

    for ref in references:
        reference_ngrams = Counter(zip(*[ref[i:] for i in range(n)]))
        max_reference_ngrams += reference_ngrams

    cc = {ngram: min(candidate_ngrams[ngram], max_reference_ngrams[ngram]) for ngram in candidate_ngrams}

    precision = sum(cc.values()) / max(1, sum(candidate_ngrams.values()))

    cl = min(len(ref) for ref in references)

    bp = np.exp(1 - (cl / len(sentence))) if len(sentence) < cl else 1

    bleu = bp * precision

    return bleu