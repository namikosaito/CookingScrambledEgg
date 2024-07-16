#!/usr/bin/env python3
# coding:utf-8

import os, sys
import math
import pickle
import datetime
from model.mtrnn_attention import RNN as rnn
from src.rnn_learn import train
from src.rnn_test  import test

DATASET_PATH = "../pickle_dataset/"
TRAIN_PATH = "rnn_train.pickle"
TEST_PATH = "rnn_test.pickle"
RESULT_LOG = "../result/"

TAR_PATH = "../result/0716-1843_cf30_cs7_cft5_cst32/snap/20000.tar"
# (You can also use your own trained model)

now = datetime.datetime.now()
current_time = now.strftime("%m%d-%H%M")
LOG_PATH = RESULT_LOG + current_time

# ---------------------------- #
args = sys.argv
mode = args[1]
# ---------------------------- #

if mode=="train":
    angle_dim = 7
    image_dim = 61 # img+torque+touch
    input_param = {"mot":0.1, "img":0.1} # open rate
    input_param_test = {"mot":0.1, "img":0.1}

    c_size = {"cf":30, "cs":7}   ## neuron
    tau = {"io":2.0, "cf":5.0, "cs":32.0}  ## time constant

    name_nodes  = ["mot", "img"]
    split_nodes = {"mot":angle_dim, "img":image_dim}
    train_path = os.path.join(DATASET_PATH, TRAIN_PATH)
    test_path = os.path.join(DATASET_PATH, TEST_PATH)
    train_log = LOG_PATH + "_cf" + str(c_size["cf"]) + "_cs" + str(c_size["cs"]) + "_cft" + str(math.floor(tau["cf"])) + "_cst" + str(math.floor(tau["cs"])) + "/"
    nn_params = {"tau":tau, 
                 "name_node":name_nodes, 
                 "input_param":input_param, 
                 "input_param_test":input_param_test, 
                 "gpu":-1, 
                 "epoch":20000,
                 "print_iter":100, 
                 "snap_iter":5000, 
                 "c_size":c_size,
                 "outdir":train_log,
                 "split_node":split_nodes,
                 "train":train_path,
                 "test":test_path}
    train(nn_params, rnn)


elif mode=="test":
    resume = TAR_PATH
    with open(os.path.join(resume.rstrip(resume.split("/")[-1]), "../code/nn_params.pickle"), "rb") as f:
        nn_params = pickle.load(f)
    nn_params["resume"] = resume
    nn_params["input_param_test"] = {"mot":0.2, "img":0.4}
    print(nn_params)
    test(nn_params, rnn)