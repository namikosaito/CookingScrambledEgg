#!/usr/bin/env python3
# coding:utf-8

import os
import numpy as np
import cv2
from utils.logger  import Logger
from utils.trainer import dataClass
import torch
from torch.autograd import Variable


def save_img(img, path):
    img *= 255.5
    img = img.transpose(1,2,0)
    cv2.imwrite(path, np.uint8(img))


def reconstract(img_paths, params, ae):
    outdir = params["resume"].replace(".tar", "_rec")
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    ### reconstract
  
    test = dataClass(img_paths, size=params["size"], dsize=params["dsize"], 
                     batchsize=params["batch"], distort=False, test=True)
    model = ae()
    checkpoint = torch.load(params["resume"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    test.minibatch_reset(rand=False)
    while test.loop:
        test.minibatch_next()
        x_in, x_out = test()
        x_in = torch.autograd.Variable(torch.tensor(np.asarray(x_in)))
        y = model(x_in)
        y = y.cpu().detach().numpy().copy()

        ### save images
        for img, path in zip(y, test.get_path()):
            dirpath = os.path.join(outdir, path.split("/")[-3])
            if not os.path.isdir(dirpath):
                os.makedirs(dirpath)
            save_img(img, os.path.join(dirpath, path.split("/")[-1]))


def extract(img_paths, params, ae, mot_paths=[]):
    outdir = params["resume"].replace(".tar", "_mid")
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    ### motion data to concat with image features
    seqs = []
    for mot_path in mot_paths:
        with open(mot_path, "r") as fr:
            lines = fr.readlines()
        mots = []
        for line in lines:
            mots.append(line.rstrip("\n").split(" "))
        seqs.append(mots)

    ### extract
    print(params["size"])
    test = dataClass(img_paths, size=params["size"], dsize=params["dsize"], 
                     batchsize=params["batch"], distort=False, test=True)
    model = ae()
    checkpoint = torch.load(params["resume"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    test.minibatch_reset(rand=False)
    while test.loop:
        test.minibatch_next()
        x_in, x_out = test()
        x_in = torch.autograd.Variable(torch.tensor(np.asarray(x_in)))
        y = model.encode(x_in)
        y = y.cpu().detach().numpy().copy()

        ### save features with motion data
        seq_ids, time_ids = test.get_idx()
        for (h, seq_id, time_id) in zip(y, seq_ids, time_ids):
            f_name = "features_{}.dat".format(test.seq_names[seq_id])
            if mot_path!=None:
                f_name = "mot_features_{}.dat".format(test.seq_names[seq_id])
            with open(os.path.join(outdir, f_name), "a") as f:
                print(f_name)
                print("time_id =", time_id, len(seqs[seq_id]))
                if len(seqs[seq_id]) > time_id:
                    for i,v in enumerate(seqs[seq_id][time_id]):
                        f.write("{} ".format(v))
                    for i,v in enumerate(h):
                        if i == len(h)-1:
                            f.write("{}\n".format(v))
                        else:
                            f.write("{} ".format(v))

