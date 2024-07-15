#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, logging, requests, json
from datetime import datetime
from .vis_trainlog import Visualizer
import numpy as np
import torch

#----------------------------------------------------------
def printArgs(parser):
    #print "#----------------- Arguments -----------------#"
    for x in vars(parser):
        print( "{0:10s}".format(x), " = ", getattr(parser, x) )
    #print "#----------------------------------------------# \n"

def saveArgs(parser, filename):
    with open(filename, "w") as f:
        for x in vars(parser):
            strline =  "{0:10s}".format(x) + " = " + str(getattr(parser, x)) + "\n"
            f.write(strline)

#----------------------------------------------------------
class Logger(object):
    def __init__(self, outdir, name=["loss"], loc=[1], log=False):
        self.outdir = outdir

        #---------- make Directories
        self.resultdir = os.path.join(self.outdir,"snap")
        if not os.path.isdir(self.resultdir):
            os.makedirs(self.resultdir)
        logfile = os.path.join(self.outdir, "log.dat")

        #---------- logging
        self.file_logger = logging.FileHandler(filename=logfile, mode="w")
        self.file_logger.setLevel(logging.INFO)
        self.file_logger.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logging.getLogger().addHandler(self.file_logger)
        logging.getLogger().setLevel(logging.INFO)

        #---------- import visualize funtion
        self.visflag = True
        try:
            self.visualizer = Visualizer(logfile, name, loc, self.outdir, log=log)
        except:
            print("[INFO] can not import visualizer")
            self.visflag = False

    def save_argments(self, args, prints=True):
        if prints:
            printArgs(args)
        saveArgs(args, os.path.join(self.outdir,"args.txt"))

    def save_model(self, epoch, model, optimizer):
        path = os.path.join(self.resultdir, "{0:05d}.tar".format(epoch))
        torch.save({
                  'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()
                  }, path)

    def __call__(self, data):
        logging.info(data)
        if self.visflag:
            self.visualizer(False)
