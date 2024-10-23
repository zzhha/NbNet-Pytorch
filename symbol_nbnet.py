"""References:
Guangcan Mai, Kai Cao, Pong C. Yuen and Anil K. Jain.
"On the Reconstruction of Face Images from Deep Face Templates."
IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) (2018)

Gao Huang, Zhuang Liu, Laurens van der Maaten and Kilian Weinberger.
"Densely Connected Convolutional Networks." CVPR2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AddLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dconv=False, pad=1, kernel_size=3, dropout=0., l2_reg=1e-4):
        super(AddLayer, self).__init__()
        layers = []
        # BatchNorm
        layers.append(nn.BatchNorm2d(in_channels, eps=l2_reg))
        # Activation
        layers.append(nn.ReLU(inplace=True))
        # Convolution or Deconvolution
        if not dconv:
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=pad,
                bias=False
            )
        else:
            conv = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=pad,
                bias=False
            )
        layers.append(conv)
        # Dropout if any
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class NbBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate, net_switch='nbnetb', dropout=0., l2_reg=1e-4):
        super(NbBlock, self).__init__()
        self.net_switch = net_switch
        self.num_layers = int(num_layers)
        self.growth_rate = growth_rate
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.layers = nn.ModuleList()
        self.in_channels = in_channels

        if net_switch == 'nbneta':
            channels = [in_channels]
            for i in range(self.num_layers):
                layer = AddLayer(
                    in_channels=channels[i],
                    out_channels=growth_rate,
                    dropout=dropout,
                    l2_reg=l2_reg
                )
                self.layers.append(layer)
                channels.append(growth_rate)
            self.total_channels = sum(channels)
        elif net_switch == 'nbnetb':
            channels = in_channels
            for i in range(self.num_layers):
                layer = AddLayer(
                    in_channels=channels,
                    out_channels=growth_rate,
                    dropout=dropout,
                    l2_reg=l2_reg
                )
                self.layers.append(layer)
                channels += growth_rate
            self.total_channels = channels

    def forward(self, x):
        if self.net_switch == 'nbneta':
            out = [x]
            for i, layer in enumerate(self.layers):
                y = layer(out[i])
                out.append(y)
            x = torch.cat(out, dim=1)
        elif self.net_switch == 'nbnetb':
            for layer in self.layers:
                y = layer(x)
                x = torch.cat([x, y], dim=1)
        return x

class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0., l2_reg=1e-4):
        super(TransitionBlock, self).__init__()
        self.layer = AddLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            dconv=True,
            pad=1,
            kernel_size=4,
            dropout=dropout,
            l2_reg=l2_reg
        )

    def forward(self, x):
        x = self.layer(x)
        return x

class DenseNet(nn.Module):
    def __init__(self,
                 input_channels=3,
                 num_block=4,
                 num_layer=32,
                 growth_rate=8,
                 dropout=0.,
                 l2_reg=1e-4,
                 net_switch='nbnetb',
                 init_channels=256):
        super(DenseNet, self).__init__()
        self.input_channels = input_channels
        self.num_block = num_block
        self.num_layer = num_layer
        self.growth_rate = growth_rate
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.net_switch = net_switch
        self.init_channels = init_channels

        self.n_channels = init_channels

        # Initial deconvolution layer
        self.conv0 = nn.ConvTranspose2d(
            in_channels=self.input_channels,
            out_channels=self.n_channels,
            kernel_size=5,
            stride=1,
            padding=0,
            bias=False
        )

        self.blocks = nn.ModuleList()
        n_channels = self.n_channels
        num_layer = self.num_layer

        for i in range(num_block - 1):
            # nb_block
            block = NbBlock(
                in_channels=n_channels,
                num_layers=num_layer,
                growth_rate=self.growth_rate,
                net_switch=self.net_switch,
                dropout=self.dropout,
                l2_reg=self.l2_reg
            )
            self.blocks.append(block)
            n_channels = n_channels + num_layer * self.growth_rate  # after nb_block

            # Reduce n_channels for transition block
            n_channels_new = int(n_channels / 2)
            num_layer = int(num_layer / 2)

            # transition block
            trans_block = TransitionBlock(
                in_channels=n_channels,
                out_channels=n_channels_new,
                dropout=self.dropout,
                l2_reg=self.l2_reg
            )
            self.blocks.append(trans_block)

            n_channels = n_channels_new  # update n_channels for next iteration

        # Last nb_block
        self.last_block = NbBlock(
            in_channels=n_channels,
            num_layers=num_layer,
            growth_rate=self.growth_rate,
            net_switch=self.net_switch,
            dropout=self.dropout,
            l2_reg=self.l2_reg
        )
        n_channels = n_channels + num_layer * self.growth_rate

        # Final layers
        self.bn_last = nn.BatchNorm2d(int(n_channels), eps=self.l2_reg)
        self.relu_last = nn.ReLU(inplace=True)
        self.conv_last = nn.Conv2d(
            in_channels=int(n_channels),
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.tanh_last = nn.Tanh()

    def forward(self, x):
        # x is expected to be of shape (batch_size, input_channels)
        x = x.view(x.size(0), x.size(1), 1, 1)  # reshape to (batch_size, input_channels, 1, 1)

        x = self.conv0(x)
        # Now go through the blocks
        for block in self.blocks:
            x = block(x)
        x = self.last_block(x)
        x = self.bn_last(x)
        x = self.relu_last(x)
        x = self.conv_last(x)
        x = self.tanh_last(x)
        return x
