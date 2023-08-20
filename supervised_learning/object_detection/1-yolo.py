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

    def process_outputs(self, outputs, image_size):
        boxes, box_confidences, box_class_probs = [], [], []

        img_height, img_width = image_size

        for output, anchors in zip(outputs, self.anchors):
            grid_height, grid_width, _, _ = output.shape

            t_xy = output[..., :2]
            t_wh = output[..., 2:4]
            box_confidence = output[..., 4:5]
            box_class_prob = output[..., 5:]

            sigmoid_t_xy = self.sigmoid(t_xy)
            sigmoid_t_wh = np.exp(t_wh)

            grid_x = np.arange(grid_width)
            grid_y = np.arange(grid_height)
            grid_x, grid_y = np.meshgrid(grid_x, grid_y)

            grid_x = np.expand_dims(grid_x, axis=-1)
            grid_y = np.expand_dims(grid_y, axis=-1)

            b_x = (sigmoid_t_xy[..., 0] + grid_x) / grid_width
            b_y = (sigmoid_t_xy[..., 1] + grid_y) / grid_height
            b_w = (anchors[:, 0] * sigmoid_t_wh[..., 0]) / img_width
            b_h = (anchors[:, 1] * sigmoid_t_wh[..., 1]) / img_height

            x1 = (b_x - b_w / 2) * img_width
            y1 = (b_y - b_h / 2) * img_height
            x2 = x1 + b_w * img_width
            y2 = y1 + b_h * img_height

            sigmoid_class_probs = self.sigmoid(box_class_prob)

            boxes.append(np.stack((x1, y1, x2, y2), axis=-1))
            box_confidences.append(box_confidence)
            box_class_probs.append(sigmoid_class_probs)

        return boxes, box_confidences, box_class_probs

    def sigmoid(self, x):
        """
        Sigmoid Function
        """
        return (1 / (1 + np.exp(-x)))
