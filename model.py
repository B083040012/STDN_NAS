from turtle import shape
from matplotlib.cbook import flatten
import numpy as np
import torch.nn as nn
import torch.nn.functional as nnf
import torch
import random

from choice_block import CNN_Flow,CNN_Vol

nbhd_channel = [2,64,64,64]
flow_channel = [4,64,64,64]


last_channel = 1

class STDN_NAS(nn.Module):
    def __init__(self, lstm_seq_len):
        super(STDN_NAS, self).__init__()

        self.level=3
        self.kernel_list=[1,2,3]
        self.stride=1
        self.lstm_seq_len=lstm_seq_len
        self.device='cuda'

        self.relu_layer=nn.ReLU()

        self.choice_block=nn.ModuleList([])

        for level_num in range(self.level):
            nbhd_inp, nbhd_oup=nbhd_channel[level_num], nbhd_channel[level_num+1]
            flow_inp, flow_oup=4, flow_channel[level_num+1]
            flow_block=nn.ModuleList([])
            vol_block=nn.ModuleList([])
            for kernel_size in self.kernel_list:
                vol_block.append(CNN_Vol(nbhd_inp,nbhd_oup,kernel_size, self.stride))
                flow_block.append(CNN_Flow(flow_inp, flow_oup, kernel_size, self.stride))
            self.choice_block.append(nn.ModuleList([vol_block, flow_block]))

    def __call__(self, nbhd_input, flow_input, choice):
        y=self.forward(nbhd_input, flow_input, choice)
        return y

    def forward(self, nbhd_input, flow_input, choice):
            nbhd_input=[nbhd_input[i].to(self.device) for i in range(self.lstm_seq_len)]
            flow_input=[flow_input[i].to(self.device) for i in range(self.lstm_seq_len)]

            nbhd_input=[nbhd_input[i].float() for i in range(self.lstm_seq_len)]
            flow_input=[flow_input[i].float() for i in range(self.lstm_seq_len)]

        # level 1 with lstm_seq_len model
            nbhd_convs=[self.choice_block[0][0][choice[0]](nbhd_input[i]) for i in range(self.lstm_seq_len)]
            nbhd_convs=[self.relu_layer(nbhd_convs[i]) for i in range(self.lstm_seq_len)]

            expected_shape=(nbhd_convs[0].shape[-2],nbhd_convs[0].shape[-1])

            flow_convs=[self.choice_block[0][1][choice[0]](flow_input[i]) for i in range(self.lstm_seq_len)]
            flow_convs=[self.relu_layer(flow_convs[i]) for i in range(self.lstm_seq_len)]

            flow_gates=[torch.sigmoid(flow_convs[i]) for i in range(self.lstm_seq_len)]
            # linear_layer=nn.Linear(in_features=flow_gates[0].shape[-1], out_features=nbhd_convs[0].shape[-1])
            # flow_gates=[linear_layer(flow_gates[i]) for i in range(self.lstm_seq_len)]
            flow_gates=[nnf.interpolate(flow_gates[i], size=expected_shape) for i in range(self.lstm_seq_len)]

            nbhd_convs=[torch.mul(nbhd_convs[i], flow_gates[i]) for i in range(self.lstm_seq_len)]

        # level 2 with lstm_seq_len_model
            nbhd_convs=[self.choice_block[1][0][choice[1]](nbhd_convs[i]) for i in range(self.lstm_seq_len)]
            nbhd_convs=[self.relu_layer(nbhd_convs[i]) for i in range(self.lstm_seq_len)]

            expected_shape=(nbhd_convs[0].shape[-2],nbhd_convs[0].shape[-1])

            flow_convs=[self.choice_block[1][1][choice[1]](flow_input[i]) for i in range(self.lstm_seq_len)]
            flow_convs=[self.relu_layer(flow_convs[i]) for i in range(self.lstm_seq_len)]

            flow_gates=[torch.sigmoid(flow_convs[i]) for i in range(self.lstm_seq_len)]
            flow_gates=[nnf.interpolate(flow_gates[i], size=expected_shape) for i in range(self.lstm_seq_len)]

            nbhd_convs=[torch.mul(nbhd_convs[i], flow_gates[i]) for i in range(self.lstm_seq_len)]

        # level 3 with lstm_seq_len_model
            nbhd_convs=[self.choice_block[2][0][choice[2]](nbhd_convs[i]) for i in range(self.lstm_seq_len)]
            nbhd_convs=[self.relu_layer(nbhd_convs[i]) for i in range(self.lstm_seq_len)]

            expected_shape=(nbhd_convs[0].shape[-2],nbhd_convs[0].shape[-1])

            flow_convs=[self.choice_block[2][1][choice[2]](flow_input[i]) for i in range(self.lstm_seq_len)]
            flow_convs=[self.relu_layer(flow_convs[i]) for i in range(self.lstm_seq_len)]

            flow_gates=[torch.sigmoid(flow_convs[i]) for i in range(self.lstm_seq_len)]
            flow_gates=[nnf.interpolate(flow_gates[i], size=expected_shape) for i in range(self.lstm_seq_len)]

            nbhd_convs=[torch.mul(nbhd_convs[i], flow_gates[i]) for i in range(self.lstm_seq_len)]

        # dense part
            flatten_layer=nn.Flatten().to(self.device)
            nbhd_convs=[flatten_layer(nbhd_convs[i]) for i in range(self.lstm_seq_len)]

            linear_layer=nn.Linear(in_features=nbhd_convs[0].shape[-1], out_features=128).to(self.device)
            nbhd_convs=[linear_layer(nbhd_convs[i]) for i in range(self.lstm_seq_len)]

            nbhd_vecs=[self.relu_layer(nbhd_convs[i]) for i in range(self.lstm_seq_len)]

        # concetanate part
            nbhd_vec=torch.cat(nbhd_vecs, dim=1)
            # print("nbhd_vec shape: ", nbhd_vec.shape)
            # nbhd_vec=torch.reshape(nbhd_vec, shape=(self.lstm_seq_len, 128))

            linear_layer=nn.Linear(in_features=nbhd_vec.shape[-1], out_features=2).to(self.device)
            output=linear_layer(nbhd_vec)
            # print("shape of output: ", output.shape)
            return output
        # lstm part
            # lstm_output=torch.lstm(batch_first=True)

            print("shape of nbhd_vecs:", nbhd_vecs[0].shape)

            return nbhd_vecs