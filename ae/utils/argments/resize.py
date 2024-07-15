#!/usr/bin/env python
# coding:utf-8

import random
import cv2
import numpy as np

inters = (
    cv2.INTER_LINEAR,
    cv2.INTER_AREA,
    cv2.INTER_NEAREST,
    cv2.INTER_CUBIC,
    cv2.INTER_LANCZOS4,
)

def random_resize(img, bboxes, resize_range=[]):
    ### random resize value
    i = random.uniform(resize_range[0], resize_range[1])

    ### resize image
    _, H, W = img.shape
    img = img.transpose((1, 2, 0))
    new_H, new_W = int(H*i), int(W*i)
    inter = random.choice(inters)
    img = cv2.resize(img, (new_W, new_H))
    img = img.astype(np.float32).transpose((2, 0, 1))

    ### resize bounding boxes
    bboxes = bboxes * i

    return img, bboxes


def resize_with_bbox(img, bboxes, size):
    #_, H, W = img.shape
    H, W, _ = img.shape
    in_size = [H, W]
    out_size = [size, size]

    ### image
    cv_img = img.transpose((1, 2, 0))
    inters = (
        cv2.INTER_LINEAR,
        cv2.INTER_AREA,
        cv2.INTER_NEAREST,
        cv2.INTER_CUBIC,
        cv2.INTER_LANCZOS4,
    )
    inter = random.choice(inters)
    H, W = size, size
    cv_img = cv2.resize(cv_img, (W, H))
    img = cv_img.astype(np.float32).transpose((2, 0, 1))

    ### bbox
    bboxes = bboxes.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bboxes[:, 0] = y_scale * bboxes[:, 0]
    bboxes[:, 2] = y_scale * bboxes[:, 2]
    bboxes[:, 1] = x_scale * bboxes[:, 1]
    bboxes[:, 3] = x_scale * bboxes[:, 3]

    return img, bboxes


def resize_image(img, size):
    _, H, W = img.shape
    in_size = [H, W]
    out_size = [size, size]

    ### image
    #cv_img = img.transpose((1, 2, 0))
    inter = random.choice(inters)
    H, W = size, size
    img = cv2.resize(img, (W, H))
    #img = cv2.resize(img, (W, H), interpolation=inter)
    #img = cv_img.astype(np.float32).transpose((2, 0, 1))
    return img

