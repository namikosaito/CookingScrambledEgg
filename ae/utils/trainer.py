#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re, sys, math, time
import random
import numpy as np
from PIL import Image
import cv2
from .argments.resize  import resize_image
from .argments.crop    import random_crop_image
from .argments.flip    import random_flip_image
from .argments.distort import random_distort

def save_img(img, path, denormalize=True, transpose=True):
    if denormalize:
        img *= 255.5
    if transpose:
        img = img.transpose(1,2,0)
    img = np.uint8(img)
    cv2.imwrite(path, img)
    #img = Image.fromarray(img)
    #img.save(path)

class dataClass(object):
    def __init__(self, path_list, size, dsize, batchsize, distort=False, test=False):
        self.size  = size
        self.dsize = dsize
        self.batch = batchsize
        self.test  = test
        self.distort = distort
        self.load(path_list)
        self.max_iter  = int(math.ceil(len(self)/float(self.batch)))
        self.iter_all =0

    def load(self, path_list):
        pathes = []
        seq_name = []
        seq_id = []
        time_id = []
        for i,path in enumerate(path_list):
            seq_name.append(path.split("/")[-2])
            root = path.replace(path.split("/")[-1],"")
            with open(path, "r") as fr:
                lines = fr.readlines()
            for ts,line in enumerate(lines):
                path = line.replace("\n","").split(" ")[0]
                pathes.append(os.path.join(root, path))
                seq_id.append(i)
                time_id.append(ts)

        """
        N = len(pathes)
        W = H = self.size+self.dsize
        img = np.zeros( (N, W, H, 3) )
        print('load img')
        for i, p in enumerate(pathes):
            print("{}/{}:{}".format(i,N,p))
            img[i] = cv2.resize( cv2.imread(p), (W,H) )
        """
        
        #import ipdb; ipdb.set_trace()
        self.dataset = np.asarray(pathes)
        self.seq_ids = np.asarray(seq_id)
        self.time_ids = np.asarray(time_id)
        self.seq_names = np.asarray(seq_name)

    def minibatch_next(self):
        current_idx = self.iter*self.batch
        self.batch_idx = self.perm[current_idx:current_idx+self.batch]
        self.iter += 1
        """
        if not self.test:
            if self.iter%self.test_iter==0:
                if self.iter/self.test_iter<=self.c_num:
                    self.check = True
            else:
                self.check = False
        """
        if self.iter==self.max_iter:
            self.loop = False
        self.iter_all += 1

    def minibatch_reset(self, rand=True):
        if rand:
            self.perm = np.random.permutation(len(self))
        else:
            self.perm = np.arange(len(self))
        self.iter = 0
        self.loop = True

    def transform(self, imgpath):
        img = cv2.imread(imgpath)
        #img = img[:450,110:580,:]
        #print img.shape
        #print img.shape
        #save_img(img, os.path.join("/home/assimilation","test_"+imgpath.split("/")[-1]), denormalize=False, transpose=False)
        img = resize_image(img, self.size+self.dsize)
        #print img.shape
        img = random_crop_image(img, self.size, test=self.test)
        #print img.shape
        #save_img(img, os.path.join("/home/assimilation","test3_"+imgpath.split("/")[-1]), denormalize=False, transpose=False)
        #if not self.test:
        #    img = random_flip_image(img, x_random=True)
        if self.distort:
            #img = random_distort(img)
            img_distort = np.copy(img)
            if random.randrange(2):
                img_distort = random_distort(img_distort)
            img_distort = img_distort.astype(np.float32).transpose(2,0,1)
            img_distort /= 255.0
        else:
            img_distort = np.copy(img)
            img_distort = img_distort.astype(np.float32).transpose(2,0,1)
            img_distort /= 255.0
        img = img.astype(np.float32).transpose(2,0,1)
        img /= 255.0 # <- very important !!! (VAE)
        #if self.test:
        #    save_img(img, os.path.join("/home/assimilation","test_"+imgpath.split("/")[-1]))
        #else:
        #    save_img(img, os.path.join("/home/assimilation",imgpath.split("/")[-1]))
        #save_img(img, os.path.join("/home/assimilation","test2_"+imgpath.split("/")[-1]))
        #sys.exit()
        return img_distort.astype(np.float32), img.astype(np.float32)

    def get_idx(self):
        return self.seq_ids[self.batch_idx], self.time_ids[self.batch_idx]

    def get_path(self):
        return self.dataset[self.batch_idx]

    def __call__(self):
        N = len(self.batch_idx)
        x_in  = np.zeros((N, 3, self.size, self.size), np.float32)
        x_out = np.zeros((N, 3, self.size, self.size), np.float32)
        #x_in, x_out = [], []
        for i, imgpath in enumerate(self.dataset[self.batch_idx]):
            img_in, img_out = self.transform(imgpath)
            x_in[i] = img_in
            x_out[i] = img_out
            #x_in.append(img_in)
            #x_out.append(img_out)
        return x_in, x_out

    def __len__(self):
        return len(self.dataset)



