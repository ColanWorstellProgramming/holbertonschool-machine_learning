#!/usr/bin/env python3
"""
Imports
"""
import numpy as np


def uni_bleu(references, sentence):
    """
    uni bleu
    """
    cc = {word: min([ref.count(word) for ref in references],
                                  default=0) for word in sentence}
    total_cc = sum(cc.values())
    precision = total_cc / len(sentence) if len(sentence) > 0 else 0

    cl = min(len(ref) for ref in references)

    bp = np.exp(1 - (cl / len(sentence))) if len(sentence) < cl else 1

    bleu = bp * precision

    return bleu
