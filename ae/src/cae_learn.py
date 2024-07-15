#!/usr/bin/env python3
# coding:utf-8

import os, sys, six, shutil
sys.path.append('../')
import pickle
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
from utils.logger  import Logger
from torch.utils.data import DataLoader, Dataset
from utils.trainer2 import ImageFolder
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F


def train(train_path, test_path, params, ae):

    ###----------------- setup -----------------###
    ### logger
    logger = Logger(params["outdir"], name=["loss"], loc=[1], log=True)
    if os.path.isdir(os.path.join(params["outdir"], "code")):
        shutil.rmtree(os.path.join(params["outdir"], "code"))
    shutil.copytree("../ae", os.path.join(params["outdir"], "code"))
    with open(os.path.join(params["outdir"], "code", "nn_params.pickle"), mode='wb') as f:
        pickle.dump(params, f)
    ### dataset
    train_im  = ImageFolder(train_path, size=params["size"], dsize=params["dsize"], distort=True, test=False, gpu=params["gpu"])
    traindata = DataLoader(train_im, batch_size=params["batch"], shuffle=True, num_workers=params["n_workers"], drop_last=True, pin_memory=True)
    test_im   = ImageFolder(test_path, size=params["size"], dsize=params["dsize"], distort=True, test=True, gpu=params["gpu"])
    testdata  = DataLoader(test_im, batch_size=params["batch"], shuffle=True, num_workers=params["n_workers"], drop_last=True, pin_memory=True)
    
    ### model, loss, optimizer
    model = ae().cuda(params["gpu"])
    print("load model")
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(list(model.parameters()))
    ### load
    if params["resume"]:
        checkpoint = torch.load(params["resume"])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    ###----------------- learn -----------------###

    pbar = tqdm(six.moves.range(1,params["epoch"]+1),total=params["epoch"],desc="epoch",ascii=True)
    for epoch in pbar:
        ### train
        model.train()
        sum_loss = 0.0
        for x_in, x_out in traindata:
            x_in  = x_in.cuda(params["gpu"])  + (torch.rand(params["batch"], 3, params["size"], params["size"]) / 50).cuda(params["gpu"]) ### noise
            x_out = x_out.cuda(params["gpu"])
            optimizer.zero_grad()
            y = model(x_in)
            loss = criterion(y, x_out)
            sum_loss += float(loss.data) * len(x_in.data)
            loss.backward()
            optimizer.step()

        if (epoch%params["print_iter"])==0 or epoch==params["epoch"]:
            info_train = "train: {}/{} loss: {:.2e}".format(epoch, params["epoch"], sum_loss/len(traindata))
            logger(info_train)

            ### test
            model.eval()
            sum_loss_test = 0.0

            for x_in, x_out in testdata:
                x_in  = x_in.cuda(params["gpu"])
                x_out = x_out.cuda(params["gpu"])             
                y = model(x_in)
                loss = criterion(y, x_out)
                sum_loss_test += float(loss.data) * len(x_in.data)

            info_test = "test:  {}/{} loss: {:.2e}".format(epoch, params["epoch"], sum_loss_test/len(testdata))
            logger(info_test)

        if (epoch%params["snap_iter"])==0 or epoch==params["epoch"]:
            logger.save_model(epoch, model, optimizer)
