#!/usr/bin/env python3
"""
Imports
"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """
    Dataset Class
    """
    def __init__(self):
        """
        Init Function
        """
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train',
                                    as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation',
                                    as_supervised=True)

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)


    def tokenize_dataset(self, data):
        """
        Tokenize Dataset Function
        """
        all_text = []
        for pt, en in data:
            all_text.append(pt.numpy())
            all_text.append(en.numpy())

        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            all_text, target_vocab_size=2**15
        )

        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            all_text, target_vocab_size=2**15
        )

        return tokenizer_pt, tokenizer_en
