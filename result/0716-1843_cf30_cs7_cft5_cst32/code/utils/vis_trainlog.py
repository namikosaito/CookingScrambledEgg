#!/usr/bin/env python
# coding:utf-8

import os, argparse
from parse import search
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#----------------------------------------------------------
class Visualizer(object):
    #print("print")
    def __init__(self, file, keys, locs, outdir, title="training history", log=False):
        print("e")
        self.file = file
        self.keys = keys
        self.locs = locs
        self.mode = ["train","test"]
        self.title  = title
        self.outdir = outdir
        self.log  = log
        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)

    def plotGraph(self,data):
        #print("print")
        for key,loc in zip(self.keys,self.locs):
            plt.plot(data["train-epoch"], data["train-"+key], "b-", label="train")
            if data["test-epoch"]:
                plt.plot(data["test-epoch"], data["test-"+key], "r-", label="test")
            plt.title(self.title)
            plt.xlabel("epoch")
            plt.ylabel(key)
            if self.log:
                plt.yscale("log")
            plt.grid()
            plt.legend(loc=loc)
            plt.savefig(os.path.join(self.outdir,"fig_"+key+".png"))
            plt.clf()
            print("a")

    def plotGraph_iter(self,data):
        for key,loc in zip(self.keys,self.locs):
            plt.plot(data["iteration"], data["iter-"+key], "b-", label="train")
            plt.title(self.title)
            plt.xlabel("iteration")
            plt.ylabel(key)
            if self.log:
                plt.yscale("log")
            plt.grid()
            plt.legend(loc=loc)
            plt.savefig(os.path.join(self.outdir,"fig_iter_"+key+".png"))
            plt.clf()
            print("b")

    def initialize(self):
        print("c")
        data = {}
        for m in self.mode:
            data[m+"-epoch"] = []
            for key in self.keys:
                data[m+"-"+key] = []
        data["iteration"] = []
        for key in self.keys:
            data["iter-"+key] = []
        return data

    def __call__(self, iter_flag=False):
        print("d")
        #print("print")
        data = self.initialize()
        f = open(self.file)
        lines = f.readlines()
        f.close()
        if iter_flag:
            for line in lines:
                if search("iter",line):
                    iteration = search("{:d}//", line)[0]
                    data["iteration"].append(iteration)
                    for key in self.keys:
                        val = search(key+": {:.2e}", line)[0]
                        data["iter-"+key].append(float(val))
                else:
                    pass
            #print("print2")
            self.plotGraph_iter(data)
        else:
            for line in lines:
                if search("train", line):
                    epoch = search("{:d}/", line)[0]
                    data["train-epoch"].append(epoch)
                    for key in self.keys:
                        val = search(key+": {:.2e}", line)[0]
                        data["train-"+key].append(float(val))
                elif search("test",line):
                    epoch = search("{:d}/", line)[0]
                    data["test-epoch"].append(epoch)
                    for key in self.keys:
                        val = search(key+": {:.2e}", line)[0]
                        data["test-"+key].append(float(val))
                else:
                    pass
            #print("print3")
            self.plotGraph(data)
