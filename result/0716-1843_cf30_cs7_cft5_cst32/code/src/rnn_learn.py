#!/usr/bin/env python
# coding:utf-8

import os, six, shutil
import pickle
from tqdm import tqdm
import numpy as np
from utils.logger import Logger
import torch
import torch.nn.functional as F


ignore_value = -999


def train_partical(model, criterion, in_data, out_data, input_param, sp, n_name,
                   optimizer=None, tbtt=True, choice_list=[500, 1000]):
    mot_input_param = input_param["mot"]
    im_input_param = input_param["img"]
    sp_dim = sp["mot"]

    loss = [0]*(len(n_name))
    x_data = torch.tensor(in_data, requires_grad=False)
    t_data = torch.tensor(out_data, requires_grad=False)
    for ts in range(in_data.shape[1]):
        mask = np.where(out_data[:, ts, 0] != ignore_value)[0]
        x_ndarray = x_data[:, ts].clone().detach()
        t_ndarray = t_data[:, ts].clone().detach()

        if ts != 0:
            prev_out = y.data
            x1 = mot_input_param * x_ndarray[:, :sp_dim] + (1.0-mot_input_param) * prev_out[:, :sp_dim]
            x2 = im_input_param * x_ndarray[:, sp_dim:] + (1.0-im_input_param) * prev_out[:, sp_dim:]
            x_ndarray = torch.cat((x1, x2), axis=-1)
        x = x_ndarray + ((torch.rand(x_ndarray.shape)-0.5)/50) #noise
        t = t_ndarray
        y, cf, cs, cf_inter, cs_inter = model.forward(x, ts)

        if len(mask) != 0:
            loss[0] += criterion(y[mask, :sp_dim], t[mask, :sp_dim])*1#mot 
            loss[1] += criterion(y[mask, sp_dim:sp_dim+14], t[mask, sp_dim:sp_dim+14])*1 #tor
            loss[1] += criterion(y[mask, sp_dim+14:sp_dim+18], t[mask, sp_dim+14:sp_dim+18])*1#tac
            loss[1] += criterion(y[mask, sp_dim+18:], t[mask, sp_dim+18:])*1 #img

    if optimizer != None:
        optimizer.zero_grad()
        sum(loss).backward()
        optimizer.step()
    return model, optimizer, loss


def train(params, rnn, lqr=False, lqrstep=5):
    logger = Logger(params["outdir"], name=["loss"]+params["name_node"],
                    loc=[1]*(len(params["name_node"])+1))
    if os.path.isdir(os.path.join(params["outdir"], "code")):
        shutil.rmtree(os.path.join(params["outdir"], "code"))
    shutil.copytree("../rnn", os.path.join(params["outdir"], "code"))
    with open(os.path.join(params["outdir"], "code", "nn_params.pickle"), mode='wb') as f:
        pickle.dump(params, f)

    with open(params["train"], "rb") as f:
        dataset_train = pickle.load(f, encoding="latin1")
    with open(params["test"], "rb") as f:
        dataset_test = pickle.load(f, encoding="latin1")
    teach_in = dataset_train[:, :-1, :] 
    teach_out = dataset_train[:, 1:, :]
    test_in = dataset_test[:, :-1, :]
    test_out = dataset_test[:, 1:, :]

    _, steps, insize = teach_in.shape
    model = rnn(insize, insize, params)
    model.initialize_c_state(rand=False, cf=False, cs=False)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    pbar = tqdm(six.moves.range(1, params["epoch"]+1), total=params["epoch"],
                desc="epoch", ascii=True)
    for epoch in pbar:
        model.train()
        model, optimizer, loss = train_partical(model, criterion, teach_in, teach_out,
                                                params["input_param"], params["split_node"], params["name_node"],
                                                optimizer=optimizer, tbtt=True)

        if (epoch % params["print_iter"]) == 0 or epoch == params["epoch"]:
            info_train = "train: {}/{} loss: {:.2e}".format(epoch,
                                                            params["epoch"], sum(loss).data/steps)
            info_train += "\t{}: {:.2e}".format(params["name_node"][0], loss[0].data/steps)
            info_train += "\t{}: {:.2e}".format(params["name_node"][1], loss[1].data/steps)
            logger(info_train)

            model.eval()
            _, _, loss = train_partical(model, criterion, test_in, test_out,
                                        params["input_param_test"], params["split_node"], params["name_node"])
            info_test = "test:\t{}/{}\tloss: {:.2e}".format(epoch, params["epoch"], sum(loss).data/steps)
            info_test += "\t{}: {:.2e}".format(params["name_node"][0], loss[0].data/steps)
            info_test += "\t{}: {:.2e}".format(params["name_node"][1], loss[1].data/steps)
            logger(info_test)

        if (epoch % params["snap_iter"]) == 0 or epoch == params["epoch"]:
            logger.save_model(epoch, model, optimizer)
