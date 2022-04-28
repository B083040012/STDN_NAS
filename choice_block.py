import torch
import torch.nn as nn

"""
Block for Vol (nbhd) and Flow process
"""

class CNN_Vol(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride):
        super(CNN_Vol, self).__init__()

        self.in_channel=in_channel
        self.out_channel=out_channel
        self.kernel=kernel
        self.stride=stride

        self.cb=nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, kernel_size=self.kernel, stride=self.stride, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        y=self.cb(x)
        return y

class CNN_Flow(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride):
        super(CNN_Flow, self).__init__()

        self.in_channel=in_channel
        self.out_channel=out_channel
        self.kernel=kernel
        self.stride=stride

        self.cb=nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, kernel_size=self.kernel, stride=self.stride, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        y=self.cb(x)
        return y