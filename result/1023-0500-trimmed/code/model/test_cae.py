#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

#mid_size=26 #size=224
mid_size=14 #size=128

class CAE(nn.Module):
    def __init__(self, ch=3, seed=1, mid=30):
        super(CAE, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(ch,32,6,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32,64,6,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64,128,6,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(128*mid_size*mid_size, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(True),
            nn.Linear(1000,mid),
            nn.BatchNorm1d(mid),
            nn.Sigmoid()
        )

        self.decoder1 = nn.Sequential(
            nn.Linear(mid,1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(True),
            nn.Linear(1000, 128*mid_size*mid_size),
            nn.BatchNorm1d(128*mid_size*mid_size),
            nn.ReLU(True)
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128,64,6,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64,32,6,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32,ch,6,stride=2,padding=0),
            nn.ReLU(True)
        )

    def encode(self, x):
        #print(self.encoder1(x))
        #print("x=",x)
        #print(x.shape)
        hid = self.encoder1(x)
        #print hid.shape
        hid = hid.view(x.shape[0], 128*mid_size*mid_size)
        hid = self.encoder2(hid)
        return hid

    def decode(self, x):
        hid = self.decoder1(x)
        hid = hid.view(x.shape[0], 128, mid_size, mid_size)
        hid = self.decoder2(hid)
        return hid

    #def __call__(self, x):
    #    return self.decode(self.encode(x))

    def forward(self, x):
        return self.decode(self.encode(x))
