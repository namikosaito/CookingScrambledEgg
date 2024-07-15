#!/usr/bin/env python
# coding:utf-8

import os, csv
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from utils.logger import Logger

# FILE_NAMES = ['/home/mayu/kubo/scrambledegg_ws/py2_log/0818-1701_entire/snap/01500_mid/mot_features_blue_1_train1.dat', '/home/mayu/kubo/scrambledegg_ws/py2_log/0818-1701_entire/snap/01500_mid/mot_features_blue_1_train2.dat', '/home/mayu/kubo/scrambledegg_ws/py2_log/0818-1701_entire/snap/01500_mid/mot_features_blue_1_train3.dat', '/home/mayu/kubo/scrambledegg_ws/py2_log/0818-1701_entire/snap/01500_mid/mot_features_blue_1_train4.dat', '/home/mayu/kubo/scrambledegg_ws/py2_log/0818-1701_entire/snap/01500_mid/mot_features_blue_2_train1.dat', '/home/mayu/kubo/scrambledegg_ws/py2_log/0818-1701_entire/snap/01500_mid/mot_features_blue_2_train2.dat', '/home/mayu/kubo/scrambledegg_ws/py2_log/0818-1701_entire/snap/01500_mid/mot_features_blue_2_train3.dat', '/home/mayu/kubo/scrambledegg_ws/py2_log/0818-1701_entire/snap/01500_mid/mot_features_blue_2_train4.dat', '/home/mayu/kubo/scrambledegg_ws/py2_log/0818-1701_entire/snap/01500_mid/mot_features_blue_3_train1.dat', '/home/mayu/kubo/scrambledegg_ws/py2_log/0818-1701_entire/snap/01500_mid/mot_features_blue_3_train2.dat', '/home/mayu/kubo/scrambledegg_ws/py2_log/0818-1701_entire/snap/01500_mid/mot_features_blue_3_train3.dat', '/home/mayu/kubo/scrambledegg_ws/py2_log/0818-1701_entire/snap/01500_mid/mot_features_blue_3_train4.dat', '/home/mayu/kubo/scrambledegg_ws/py2_log/0818-1701_entire/snap/01500_mid/mot_features_normal_1_train1.dat', '/home/mayu/kubo/scrambledegg_ws/py2_log/0818-1701_entire/snap/01500_mid/mot_features_normal_1_train2.dat', '/home/mayu/kubo/scrambledegg_ws/py2_log/0818-1701_entire/snap/01500_mid/mot_features_normal_1_train3.dat', '/home/mayu/kubo/scrambledegg_ws/py2_log/0818-1701_entire/snap/01500_mid/mot_features_normal_1_train4.dat', '/home/mayu/kubo/scrambledegg_ws/py2_log/0818-1701_entire/snap/01500_mid/mot_features_normal_2_test2.dat', '/home/mayu/kubo/scrambledegg_ws/py2_log/0818-1701_entire/snap/01500_mid/mot_features_normal_2_train1.dat', '/home/mayu/kubo/scrambledegg_ws/py2_log/0818-1701_entire/snap/01500_mid/mot_features_normal_2_train2.dat', '/home/mayu/kubo/scrambledegg_ws/py2_log/0818-1701_entire/snap/01500_mid/mot_features_normal_2_train3.dat', '/home/mayu/kubo/scrambledegg_ws/py2_log/0818-1701_entire/snap/01500_mid/mot_features_normal_2_train4.dat', '/home/mayu/kubo/scrambledegg_ws/py2_log/0818-1701_entire/snap/01500_mid/mot_features_normal_3_train1.dat', '/home/mayu/kubo/scrambledegg_ws/py2_log/0818-1701_entire/snap/01500_mid/mot_features_normal_3_train2.dat', '/home/mayu/kubo/scrambledegg_ws/py2_log/0818-1701_entire/snap/01500_mid/mot_features_normal_3_train3.dat', '/home/mayu/kubo/scrambledegg_ws/py2_log/0818-1701_entire/snap/01500_mid/mot_features_normal_3_train4.dat', '/home/mayu/kubo/scrambledegg_ws/py2_log/0818-1701_entire/snap/01500_mid/mot_features_soy_1_train1.dat', '/home/mayu/kubo/scrambledegg_ws/py2_log/0818-1701_entire/snap/01500_mid/mot_features_soy_1_train2.dat', '/home/mayu/kubo/scrambledegg_ws/py2_log/0818-1701_entire/snap/01500_mid/mot_features_soy_1_train3.dat', '/home/mayu/kubo/scrambledegg_ws/py2_log/0818-1701_entire/snap/01500_mid/mot_features_soy_1_train4.dat', '/home/mayu/kubo/scrambledegg_ws/py2_log/0818-1701_entire/snap/01500_mid/mot_features_soy_2_train1.dat', '/home/mayu/kubo/scrambledegg_ws/py2_log/0818-1701_entire/snap/01500_mid/mot_features_soy_2_train2.dat', '/home/mayu/kubo/scrambledegg_ws/py2_log/0818-1701_entire/snap/01500_mid/mot_features_soy_2_train3.dat', '/home/mayu/kubo/scrambledegg_ws/py2_log/0818-1701_entire/snap/01500_mid/mot_features_soy_2_train4.dat', '/home/mayu/kubo/scrambledegg_ws/py2_log/0818-1701_entire/snap/01500_mid/mot_features_soy_3_train1.dat', '/home/mayu/kubo/scrambledegg_ws/py2_log/0818-1701_entire/snap/01500_mid/mot_features_soy_3_train2.dat', '/home/mayu/kubo/scrambledegg_ws/py2_log/0818-1701_entire/snap/01500_mid/mot_features_soy_3_train3.dat', '/home/mayu/kubo/scrambledegg_ws/py2_log/0818-1701_entire/snap/01500_mid/mot_features_soy_3_train4.dat']
FILE_NAMES = ['/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_test_blue_4_1.dat', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_test_blue_5_1.dat', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_test_blue_5_2.dat', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_test_blue_5_3.dat', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_test_blue_5_4', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_test_corn_4_1', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_test_corn_4_2', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_test_corn_5_1', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_test_normal_4_1', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_test_sausage_4_1', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_test_sausage_4_2', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_test_sausage_5_1', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_train_blue_4_1.dat', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_train_blue_4_2.dat', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_train_blue_4_3.dat', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_train_blue_4_4.dat', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_train_blue_5_1.dat', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_train_blue_5_2.dat', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_train_blue_5_3.dat', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_train_blue_5_4.dat', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_train_corn_4_1.dat', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_train_corn_4_2.dat', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_train_corn_4_3.dat', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_train_corn_4_4.dat', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_train_corn_5_1.dat', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_train_corn_5_2.dat', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_train_corn_5_3.dat', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_train_corn_5_4.dat', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_train_normal_4_1.dat', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_train_normal_4_2.dat', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_train_normal_4_3.dat', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_train_normal_4_4.dat', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_train_normal_5_1.dat', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_train_normal_5_2.dat', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_train_normal_5_3.dat', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_train_normal_5_4.dat', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_train_sausage_4_1.dat', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_train_sausage_4_2.dat', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_train_sausage_4_3.dat', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_train_sausage_4_4.dat', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_train_sausage_5_1.dat', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_train_sausage_5_2.dat', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_train_sausage_5_3.dat', '/home/ogata/hirako/scrambledegg_ws/results_ae_ito/1021-1757/snap/01500_mid/mot_features_train_sausage_5_4.dat']
ignore_value = -999.


def train_partical(model, in_data, out_data, input_param, sp, n_name, outdir_seq):

    t_seq, x_seq, y_seq, cf_seq, cs_seq = [], [], [], [], []
    for ts in range(in_data.shape[1]):
        x_ndarray = np.array(in_data[:,ts])
        t_ndarray = np.array(out_data[:,ts])
        x_ndarray = torch.tensor(x_ndarray)
        t_ndarray = torch.tensor(t_ndarray)
        if ts != 0:
            prev_out = y.data
            num = 0
            sp_ = sp[n_name[num]]
            for i in range(x_ndarray.shape[1]):
                if i < sp_:
                    if ts < 30:
                        ip = 0.5
                    else:
                        ip = input_param[n_name[num]]
                    x_ndarray[:,i] = ip * x_ndarray[:,i] + (1.0-ip) * prev_out[:,i]
                else:
                    num += 1
                    if ts < 30:
                        ip = 0.5
                    else:
                        ip = input_param[n_name[num]]
                    x_ndarray[:,i] = ip * x_ndarray[:,i] + (1.0-ip) * prev_out[:,i]
                    sp_  += sp[n_name[num]]

        x = Variable(x_ndarray)
        t = Variable(t_ndarray)
        y, cf, cs, cf_inter, cs_inter = model.forward(x, ts, dir_for_output = outdir_seq)

        if ts==0:
            t_seq   = t.data[:,None,:]
            x_seq   = x.data[:,None,:]
            y_seq   = y.data[:,None,:]
            cf_seq  = cf.data[:,None,:]
            cs_seq  = cs.data[:,None,:]
        else:
            t_seq  = np.concatenate([t_seq, t.data[:,None,:]], axis=1)
            x_seq  = np.concatenate([x_seq, x.data[:,None,:]], axis=1)
            y_seq  = np.concatenate([y_seq, y.data[:,None,:]], axis=1)
            cf_seq = np.concatenate([cf_seq, cf.data[:,None,:]], axis=1)
            cs_seq = np.concatenate([cs_seq, cs.data[:,None,:]], axis=1)
    return t_seq, x_seq, y_seq, cf_seq, cs_seq



def test(params, rnn):
    dim = 18

    outdir_seq = params["resume"].replace(".tar", "_seq")
    if not os.path.isdir(outdir_seq):
        os.makedirs(outdir_seq)

    with open(params["train"], "rb") as f:
        dataset_train = pickle.load(f,  encoding='latin1')
    with open(params["test"], "rb") as f:
        dataset_test = pickle.load(f,  encoding='latin1')
    teach_in = dataset_train[:,:-1,:]
    teach_out = dataset_train[:,1:,:]
    test_in = dataset_test[:,:-1,:]
    test_out = dataset_test[:,1:,:]

    N, steps, insize = teach_in.shape
    print(N, steps, insize)
    tN, _, _ = test_in.shape
    model = rnn(insize, insize, params)
    model.initialize_c_state(rand=False, cf=False, cs=False)
    checkpoint = torch.load(params["resume"])
    model.load_state_dict(checkpoint['model_state_dict'])
    print(model.init_cf)
    print(model.init_cs)

    colors = ["g", "b", "r", "k", "y", "c", "m", "g", "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "gray", "violet", "cyan", "lime", "salmon"]#, "navy"]
    def func(data, linestyles, title="", xlabel="", ylabel="", path="fig.jpg", lim=[-1.2, 1.2]):
        fig = plt.figure()
        figs = []
        for i in range(len(data[0][0])):
            figs.append(fig.add_subplot(7,1,i+1))
        for seq, linestyle in zip(data, linestyles):
            for j in range(seq.shape[-1]):
                val = seq[:,j]
                figs[j].plot(val, linestyle=linestyle, color=colors[j])
                figs[j].grid(True)
                figs[j].set_ylim(lim)
        plt.savefig(path)
        plt.clf()

    def csv_func(data, path):
        for seq in zip(data):
            with open(path, 'a') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(seq[0])

    def plot_seq(_y_seq, _t_seq, name=""):

        tmp_i = 0
        for i, (seq1, seq2) in enumerate(zip(_y_seq, _t_seq)):
            mask = np.where(seq2[:, 0] != ignore_value)[0]

            if name == 'test':
                name_tmp = ("_".join((FILE_NAMES[tmp_i].split("/")[-1]).split("_")[2:])).split(".")[0]
                while "train" in name_tmp:
                    tmp_i += 1
                    name_tmp = ("_".join((FILE_NAMES[tmp_i].split("/")[-1]).split("_")[2:])).split(".")[0]
                name1 = name_tmp
                if "test" in name_tmp:
                    tmp_i += 1

            elif name == 'train':
                name_tmp = ("_".join((FILE_NAMES[tmp_i].split("/")[-1]).split("_")[2:])).split(".")[0]
                while "test" in name_tmp:
                    tmp_i += 1
                    name_tmp = ("_".join((FILE_NAMES[tmp_i].split("/")[-1]).split("_")[2:])).split(".")[0]
                name1 = name_tmp
                if "train" in name_tmp:
                    tmp_i += 1

            csv_func(seq1[mask, :7], path=os.path.join(outdir_seq, ('{}_angle_{}.csv').format(name, name1)))
            func([seq1[mask, :7], seq2[mask, :7]], ['solid', 'dashed'], path=os.path.join(outdir_seq, ('{}_angle_{}.png').format(name, name1)))
            func([seq1[mask, 7:14], seq2[mask, 7:14]], ['solid', 'dashed'], path=os.path.join(outdir_seq, ('{}_force_{}.png').format(name, name1)))
            func([seq1[mask, 14:18], seq2[mask, 14:18]], ['solid', 'dashed'], path=os.path.join(outdir_seq, ('{}_tactile_{}.png').format(name, name1)))
            
    model.eval()
    outdir_analysis = outdir_seq+"/train_analysis"
    if not os.path.isdir(outdir_analysis):
        os.makedirs(outdir_analysis)
    t_seq, x_seq, y_seq, cf_seq, cs_seq = train_partical(model, teach_in, teach_out, 
                           params["input_param_test"], params["split_node"], params["name_node"], outdir_seq = outdir_analysis)
    plot_seq(y_seq, t_seq, name="train")

    outdir_analysis = outdir_seq+"/test_analysis"
    if not os.path.isdir(outdir_analysis):
        os.makedirs(outdir_analysis)
    tt_seq, tx_seq, ty_seq, tcf_seq, tcs_seq = train_partical(model, test_in, test_out, 
                           params["input_param_test"], params["split_node"], params["name_node"], outdir_seq = outdir_analysis)
    plot_seq(ty_seq, tt_seq, name="test")
