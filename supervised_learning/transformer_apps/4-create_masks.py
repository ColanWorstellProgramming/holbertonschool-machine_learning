#!/usr/bin/env python3
"""
Imports
"""
import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """
    Create Masks
    """
    en_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]

    de_mask = tf.cast(tf.math.equal(target, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]

    seq_len_out = tf.shape(target)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len_out, seq_len_out)), -1, 0)
    dec_target_padding_mask = tf.cast(tf.math.equal(target, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return en_mask, combined_mask, de_mask
