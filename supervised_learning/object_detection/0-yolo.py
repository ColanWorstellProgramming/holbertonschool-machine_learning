#!/usr/bin/env python3
"""
Imports
"""
import numpy as np
import tensorflow.keras as K


class Yolo:
    """
    Yolo Class
    """
    def __init__(self, model_path, classes_path,
                 class_t, nms_t, anchors):
        """
        Class Constructor
        """
        self.model = K.models.load_model(model_path)

        with open(classes_path) as file:
            class_names = file.read()

        self.class_names = class_names.replace("\n", "|").split("|")[:-1]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
