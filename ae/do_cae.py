#!/usr/bin/env python
# coding:utf-8

import os
from re import sub
import sys
import glob
import shutil
import pickle
from src.cae_learn import train
from src.cae_eval import reconstract, extract
import datetime

# check python version
if sys.version_info[0] == 2:
    print("please run with python3.8 not python2")
    exit()

DATASET_PATH = "../dataset/"
RESULT_LOG = "../result/"  
SENSOR_TXT = "sensor_combine_2_5Hz.txt"

#### comment out either of them ####
Learning_Target = "whole"  #"whole"

### when we use trimmed image
if Learning_Target == "trimmed":
    IMAGELIST_NAME = "imglist_part_2_5Hz.dat"
    from model.cae_trimmed import CAE as ae
    RESUME_PATH = "../result/1023-0500-trimmed/snap/01500.tar" 

### when we use whole image
elif Learning_Target == "whole":
    IMAGELIST_NAME = "imglist_entire_2_5Hz.dat"
    from model.cae_whole import CAE as ae
    RESUME_PATH = "../result/1025-2335-whole/snap/01500.tar" 

####################################

### if you select own trained model
# RESUME_PATH = "../result/XXXX/snap/XXXXX.tar" 
# RESUME_PATH = "../result/0716-1841-trimmed/snap/00010.tar"

args = sys.argv
mode = args[1]

now = datetime.datetime.now()
current_time = now.strftime("%m%d-%H%M-")
log_name = RESULT_LOG + current_time + Learning_Target


# dataset
train_path = []
test_path = []
for dir in glob.glob(os.path.join(DATASET_PATH, "*")):
    if "train" in dir:
        train_path.append(os.path.join(dir, IMAGELIST_NAME))
        print (os.path.join(dir, IMAGELIST_NAME))
    elif "test" in dir:
        test_path.append(os.path.join(dir, IMAGELIST_NAME))
        print (os.path.join(dir, IMAGELIST_NAME))

if mode == "train":
    resume = ""
    if resume:
        log = resume.rstrip(".tar")
    nn_params = {"gpu": 0,
                 "batch": 100, 
                 "size": 128,   # image size
                 "dsize": 5,    # image augumentation shift picxel size
                 "epoch": 1500,
                 "n_workers": 6,
                 "print_iter": 5,
                 "snap_iter": 50,
                 "outdir": log_name,
                 "train": train_path,
                 "test": test_path,
                 "resume": resume}
    train(train_path, test_path, nn_params, ae)


elif mode == "test":
    resume = RESUME_PATH
    if os.path.isfile("./model/test_cae.py"):
        os.remove("./model/test_cae.py")
    if Learning_Target == "trimmed":
        shutil.copyfile(os.path.join(resume.rstrip(resume.split("/")[-1]), "../code/model/cae_trimmed.py"), "./model/test_cae.py")
    elif Learning_Target == "whole":
        shutil.copyfile(os.path.join(resume.rstrip(resume.split("/")[-1]), "../code/model/cae_whole.py"), "./model/test_cae.py")
    from model.test_cae import CAE as ae

    nn_params = {"gpu": 0,
                 "batch": 1, 
                 "size": 128,   # image size
                 "dsize": 5,    # image augumentation shift picxel size
                 "resume": resume}

    files = glob.glob(os.path.join(resume.replace(".tar", "_mid"), "*.dat"))  

    mot_path = []
    for path in test_path:
        print(path)
        mot_path.append(path.replace(path.split("/")[-1], SENSOR_TXT))
    reconstract(test_path, nn_params, ae)
    extract(test_path, nn_params, ae, mot_paths=mot_path)


    mot_path = []
    for path in train_path:
        print(path)
        mot_path.append(path.replace(path.split("/")[-1], SENSOR_TXT))
    extract(train_path, nn_params, ae, mot_paths=mot_path)
    reconstract(train_path, nn_params, ae)