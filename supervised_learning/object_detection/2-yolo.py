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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter Boxes
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for box in range(len(boxes)):
            box_classes.append(np.argmax(box_class_probs[box]
                                         * box_confidences[box],
                                         axis=-1).reshape(-1))
            box_scores.append(np.max(box_class_probs[box]
                                     * box_confidences[box],
                                     axis=-1).reshape(-1))

        box_classes_con = np.concatenate(box_classes)
        box_scores_con = np.concatenate(box_scores)
        mask = box_scores_con >= self.class_t

        filtered_boxes = np.concatenate(
            [box.reshape(-1, 4) for box in boxes], axis=0)
        filtered_boxes = filtered_boxes[mask]

        box_classes = box_classes_con[mask]
        box_scores = box_scores_con[mask]

        return filtered_boxes, box_classes, box_scores

    def process_outputs(self, outputs, image_size):
        """
        Process Outputs
        """
        boxes, box_confidences, box_class_probs = [], [], []
        image_height, image_width = image_size

        for output in range(len(outputs)):
            boxes.append(outputs[output][..., :4])
            box_confidences.append(self.sigmoid(outputs[output][..., 4:5]))
            box_class_probs.append(self.sigmoid(outputs[output][..., 5:]))

        for output in range(len(boxes)):
            grid_height = outputs[output].shape[0]
            grid_width = outputs[output].shape[1]
            anchors = outputs[output].shape[2]

            for cy in range(grid_height):
                for cx in range(grid_width):
                    for b in range(anchors):
                        tx, ty, tw, th = boxes[output][cy, cx, b]
                        pw, ph = self.anchors[output][b]
                        bx = (self.sigmoid(tx)) + cx
                        by = (self.sigmoid(ty)) + cy
                        bw = pw * np.exp(tw)
                        bh = ph * np.exp(th)
                        bx /= grid_width
                        by /= grid_height
                        bw /= self.model.input.shape[1].value
                        bh /= self.model.input.shape[2].value
                        x1 = (bx - (bw / 2)) * image_width
                        y1 = (by - (bh / 2)) * image_height
                        x2 = (bx + (bw / 2)) * image_width
                        y2 = (by + (bh / 2)) * image_height
                        boxes[output][cy, cx, b] = [x1, y1, x2, y2]

        return boxes, box_confidences, box_class_probs

    def sigmoid(self, x):
        """
        Sigmoid Function
        """
        return (1 / (1 + np.exp(-x)))
