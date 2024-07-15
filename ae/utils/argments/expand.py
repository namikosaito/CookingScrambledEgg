#!/usr/bin/env python
# coding:utf-8

import random
import numpy as np

def random_expand(img, bboxes, max_ratio=4, fill=0):
    if np.random.randint(2):
        C, H, W = img.shape
        ratio = random.uniform(1, max_ratio)
        out_H, out_W = int(H * ratio), int(W * ratio)

        y_offset = random.randint(0, out_H - H)
        x_offset = random.randint(0, out_W - W)

        out_img = np.empty((C, out_H, out_W), dtype=img.dtype)
        out_img[:] = np.array(fill).reshape((-1, 1, 1))
        out_img[:, y_offset:y_offset + H, x_offset:x_offset + W] = img

        out_bboxes = bboxes.copy()
        out_bboxes[:, :2] += (y_offset, x_offset)
        out_bboxes[:, 2:] += (y_offset, x_offset)
        return out_img, out_bboxes
    else:
        return img, bboxes
        


def random_expand_image(img, max_ratio=4, fill=0):
    if np.random.randint(2):
        C, H, W = img.shape
        ratio = random.uniform(1, max_ratio)
        out_H, out_W = int(H * ratio), int(W * ratio)

        y_offset = random.randint(0, out_H - H)
        x_offset = random.randint(0, out_W - W)

        out_img = np.empty((C, out_H, out_W), dtype=img.dtype)
        out_img[:] = np.array(fill).reshape((-1, 1, 1))
        out_img[:, y_offset:y_offset + H, x_offset:x_offset + W] = img
        return out_img
    else:
        return img

