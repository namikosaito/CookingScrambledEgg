#!/usr/bin/env python
# coding:utf-8

import random

def random_flip(img, bboxes, y_random=False, x_random=False):
    ### flip image
    y_flip, x_flip = False, False
    if y_random:
        y_flip = random.choice([True, False])
        if y_flip:
            img = img[:, ::-1, :]
    if x_random:
        x_flip = random.choice([True, False])
        if x_flip:
            img = img[:, :, ::-1]

    ### flip bounding boxes
    _, H, W = img.shape
    bboxes = bboxes.copy()
    if y_flip:
        y_max = H - 1 - bboxes[:, 0]
        y_min = H - 1 - bboxes[:, 2]
        bboxes[:, 0] = y_min
        bboxes[:, 2] = y_max
    if x_flip:
        x_max = W - 1 - bboxes[:, 1]
        x_min = W - 1 - bboxes[:, 3]
        bboxes[:, 1] = x_min
        bboxes[:, 3] = x_max

    return img, bboxes


def random_flip_image(img, y_random=False, x_random=False):
    ### flip image
    y_flip, x_flip = False, False
    if y_random:
        y_flip = random.choice([True, False])
        if y_flip:
            img = img[:, ::-1, :]
    if x_random:
        x_flip = random.choice([True, False])
        if x_flip:
            img = img[:, :, ::-1]

    return img

