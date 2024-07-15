#!/usr/bin/env python
# coding:utf-8

import random
import cv2
import numpy as np

"""
brightness_delta = 32
contrast_range = [0.5, 1.5]
saturation_range = [0.5, 1.5]
hue_delta = 18
#"""

"""
brightness_delta = 8
contrast_range = [0.9, 1.1]
saturation_range = [0.9, 1.1]
hue_delta = 4
#"""

#"""
brightness_delta = 64
contrast_range = [0.4, 1.9]
saturation_range = [0.4, 1.9]
hue_delta = 36
#"""


def convert(img, alpha=1, beta=0):
    img = img.astype(float) * alpha + beta
    img[img < 0] = 0
    img[img > 255] = 255
    return img.astype(np.uint8)

def brightness(img):
    if random.randrange(2):
        return convert(img, beta=random.uniform(-brightness_delta, brightness_delta))
    else:
        return img

def contrast(img):
    if random.randrange(2):
        return convert(img, alpha=random.uniform(contrast_range[0], contrast_range[1]))
    else:
        return img

def saturation(img):
    if random.randrange(2):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:, :, 1] = convert(
            img[:, :, 1], alpha=random.uniform(saturation_range[0], saturation_range[1]))
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    else:
        return img

def hue(img):
    if random.randrange(2):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:, :, 0] = (
            img[:, :, 0].astype(int) + random.randint(-hue_delta, hue_delta)) % 180
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    else:
        return img

def random_distort(img, test=False):
    #img = img[::-1].transpose((1, 2, 0)).astype(np.uint8) # change PIL format to cv2 format
    img = img[::-1].astype(np.uint8) # change PIL format to cv2 format

    ### color argmentation
    img = brightness(img)
    if random.randrange(2):
        img = contrast(img)
        img = saturation(img)
        img = hue(img)
    else:
        img = saturation(img)
        img = hue(img)
        img = contrast(img)

    #"""
    if random.randrange(2):
        img[:,:,0] = img[:,:,0]*random.uniform(0.5, 1.5)
    if random.randrange(2):
        img[:,:,0] = img[:,:,1]*random.uniform(0.5, 1.5)
    if random.randrange(2):
        img[:,:,0] = img[:,:,2]*random.uniform(0.5, 1.5)
    #"""


    #img = img.astype(np.float32).transpose((2, 0, 1))[::-1]
    img = img[::-1]
    return img

