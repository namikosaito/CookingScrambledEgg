#!/usr/bin/env python
# coding:utf-8

import random, six, sys
import numpy as np

def random_crop_image(img, size, test=False):
    #_, H, W = img.shape
    H, W, _ = img.shape

    ### crop input image with including target bbox
    min_top = 0
    min_left = 0
    max_top = H-size
    max_left = W-size

    if test:
        #dsize = img.shape[-1] - size
        dsize = img.shape[0] - size
        top  = int(dsize / 2)
        left = int(dsize / 2)
    else:
        top  = random.randint(min_top, max_top)
        left = random.randint(min_left, max_left)

    bottom = int(top + size)
    right  = int(left + size)
    #img = img[:, top:bottom, left:right]
    img = img[top:bottom, left:right, :]
    return img

