#!/usr/bin/env python
# coding:utf-8

import random, math
import numpy as np


def rotate(x, y ,rad):
    _x = np.cos(rad)*x - np.sin(rad)*y
    _y = np.sin(rad)*x + np.cos(rad)*y
    return _x, _y

def random_rotate(img, bboxes):
    ### random rotate value
    i = np.random.randint(4)

    ### rotate image
    _, H, W = img.shape
    img = np.transpose(img, axes=(1, 2, 0))
    img = np.rot90(img, i)
    img = np.transpose(img, axes=(2, 0, 1))

    ### rotate bounding boxes
    degree = 90 * i
    new_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox[0]-(H/2.0), bbox[1]-(W/2.0), bbox[2]-(H/2.0), bbox[3]-(W/2.0)
        x1, y1 = rotate(x1, y1, math.radians(degree))
        x2, y2 = rotate(x2, y2, math.radians(degree))
        x1 += H/2.0
        y1 += W/2.0
        x2 += H/2.0
        y2 += W/2.0
        new_bbox = [min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)]
        new_bboxes.append(new_bbox)
    bboxes = np.asarray(new_bboxes)

    return img, bboxes

