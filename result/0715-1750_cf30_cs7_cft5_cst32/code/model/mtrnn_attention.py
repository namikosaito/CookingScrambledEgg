#!/usr/bin/env python
# coding:utf-8
import torch
import torch.nn as nn
import numpy as np
import os, csv

cs_filename = "online_cs.csv"
cf_filename = "online_cf.csv"
io_filename = "online_io.csv"

def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))

class RNN(nn.Module):
    def __init__(self, in_size=1, out_size=1, params=None, csv_record=False, mg=False):

        super(RNN, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.c_size = params["c_size"]
        self.tau = params["tau"]
        self.atten = params["split_node"]["mot"] + params["split_node"]["img"]
        self.csv_record = csv_record
        self.mg = mg
        print()
        print("c_size : {}\ntau : {}\natten : {}\nmg : {}".format(self.c_size, self.tau, self.atten, self.mg))

        self.i2atten = Attention(self.in_size, self.c_size['cf'], self.atten, mg=mg)
        self.i2cf  = nn.Linear(self.in_size, self.c_size['cf'])
        self.cf2cs = nn.Linear(self.c_size['cf'], self.c_size['cs'])
        self.cf2cf = nn.Linear(self.c_size['cf'], self.c_size['cf'])
        self.cs2cf = nn.Linear(self.c_size['cs'], self.c_size['cf'])
        self.cs2cs = nn.Linear(self.c_size['cs'], self.c_size['cs'])
        self.cf2o  = nn.Linear(self.c_size['cf'], self.out_size)


    def initialize_c_state(self, rand=False, cf=False, cs=False):
        self.init_cf = torch.zeros(self.c_size["cf"])
        self.init_cs = torch.zeros(self.c_size["cs"])
        if rand:
            print("initialization: random cs, cf")
            self.init_cf = torch.randn(self.c_size["cf"])
            self.init_cs = torch.randn(self.c_size["cs"])
        if cf:
            print("initialization: train init cf")
            self.init_cf = nn.Parameter(self.init_cf)
        if cs:
            print("initialization: train init cs")
            self.init_cs = nn.Parameter(self.init_cs)


    def forward(self, x, ts=999, cf_bef=0, cs_bef=0, dir_for_output=""):
        ### fast context
        if ts==0:
            self.cf_state = torch.tanh(self.init_cf)
            self.cs_state = torch.tanh(self.init_cs)
            self.cf_inter = (1.0-1.0/self.tau['cf']) * self.init_cf + (1.0/self.tau['cf']) \
                            * (self.i2cf(x) + self.cf2cf(self.cf_state) +self.cs2cf(self.cs_state))

        else:
            self.attention_map = self.i2atten(x, self.cf_state)
            self.cf_inter = (1.0-1.0/self.tau['cf']) * self.cf_inter + (1.0/self.tau['cf']) \
                            * (self.i2cf(self.attention_map* x) + self.cf2cf(self.cf_state) +self.cs2cf(self.cs_state))
            
            if self.csv_record == True:
                attention_inter_output = torch.softmax(self.attention_map, dim=0).to('cpu').detach().numpy().copy()
                for i in range(self.attention_map.shape[0]):
                    self.attention_filename = os.path.join(dir_for_output, "online_attention_"+ str(i) +".csv")
                    with open(self.attention_filename, 'a') as f_handle:
                        writer = csv.writer(f_handle, delimiter='\t')
                        writer.writerow(attention_inter_output[i])

        if self.csv_record == True:
            cf_inter_output = self.cf_inter.to('cpu').detach().numpy().copy()
            for i in range(self.cf_inter.shape[0]):
                self.cf_filename = os.path.join(dir_for_output, "online_cf_iter_"+ str(i) +".csv")
                with open(self.cf_filename, 'a') as f_handle:
                    writer = csv.writer(f_handle, delimiter='\t')
                    writer.writerow(cf_inter_output[i])

        self.cf_state = torch.tanh(self.cf_inter) 

        ### slow context
        if ts==0:
            self.cf_state = torch.tanh(self.cf_inter)  
            self.cs_inter = (1.0-1.0/self.tau['cs']) * self.init_cs + (1.0/self.tau['cs']) \
                            * (self.cf2cs(self.cf_state) + self.cs2cs(self.cs_state))

        else:
            self.cs_inter = (1.0-1.0/self.tau['cs']) * self.cs_inter + (1.0/self.tau['cs']) \
                            * (self.cf2cs(self.cf_state) + self.cs2cs(self.cs_state))

        if self.csv_record == True:
            cs_inter_output = self.cs_inter.to('cpu').detach().numpy().copy()
                
            for i in range(self.cs_inter.shape[0]):
                self.cs_filename = os.path.join(dir_for_output, "online_cs_iter_"+ str(i) +".csv")
                with open(self.cs_filename, 'a') as f_handle:
                    writer = csv.writer(f_handle, delimiter='\t')
                    writer.writerow(cs_inter_output[i])

        self.cs_state = torch.tanh(self.cs_inter)

        ### output
        y = torch.tanh(self.cf2o(self.cf_state))

        return y, self.cf_state, self.cs_state, self.cf_inter, self.cs_inter


class Attention(nn.Module):
    
    def __init__(self, dec_hid_dim, enc_hid_dim, attn_dim, mg=False):
        
        super(Attention, self).__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.attn_dim=attn_dim

        self.attn_in = (enc_hid_dim ) + dec_hid_dim
        self.attn = nn.Linear(self.attn_in, attn_dim)
        self.mg = mg

        
    def forward(self, decoder_hidden, encoder_outputs):
        
        src_len = encoder_outputs.shape[0]
        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(
            1, 
            src_len, 
            1
        )
        if self.mg == False:
            energy = torch.tanh(self.attn(
                torch.cat((
                    decoder_hidden,
                    encoder_outputs
                ),dim = 1
            )))
        else:
            energy = torch.tanh(self.attn(
                torch.cat((
                    decoder_hidden,
                    encoder_outputs
                ),dim = 0
            )))

        if self.mg == False:
            output1 = torch.softmax(energy, dim=1)
        else:
            output1 = torch.softmax(energy, dim=0) #! MotionGeneration
        output = torch.sigmoid(output1)

        return output
