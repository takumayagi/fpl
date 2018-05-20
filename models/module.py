#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

from __future__ import print_function
from __future__ import division
from six.moves import range

import chainer
import chainer.functions as F
import chainer.links as L


class Linear_BN(chainer.Chain):
    def __init__(self, nb_in, nb_out, no_bn=False):
        super(Linear_BN, self).__init__()
        self.no_bn = no_bn
        with self.init_scope():
            self.fc = L.Linear(nb_in, nb_out)
            if not no_bn:
                self.bn = L.BatchNormalization(nb_out)

    def __call__(self, x):
        if self.no_bn:
            return self.fc(x)
        else:
            return F.relu(self.bn(self.fc(x)))


class Conv_BN(chainer.Chain):
    def __init__(self, nb_in, nb_out, ksize=1, pad=0, no_bn=False):
        super(Conv_BN, self).__init__()
        self.no_bn = no_bn
        with self.init_scope():
            self.conv = L.ConvolutionND(1, nb_in, nb_out, ksize=ksize, pad=pad)
            if not no_bn:
                self.bn = L.BatchNormalization(nb_out)

    def __call__(self, x):
        if self.no_bn:
            return self.conv(x)
        else:
            return F.relu(self.bn(self.conv(x)))


class DConv_BN(chainer.Chain):
    def __init__(self, nb_in, nb_out, ksize=3, dilate=1, no_bn=False):
        super(DConv_BN, self).__init__()
        self.no_bn = no_bn
        with self.init_scope():
            self.conv = L.DilatedConvolution2D(nb_in, nb_out, ksize=(ksize, 1), pad=(dilate, 0), dilate=(dilate, 1))
            if not no_bn:
                self.bn = L.BatchNormalization(nb_out)

    def __call__(self, x):
        # x (B, C, N) -> (B, C, N, 1)
        if self.no_bn:
            return self.conv(x)
        else:
            return F.relu(self.bn(self.conv(x)))


class FC_Module(chainer.Chain):
    def __init__(self, nb_in, nb_out, inter_list=[], no_act_last=False):
        super(FC_Module, self).__init__()
        self.nb_layers = len(inter_list) + 1
        with self.init_scope():
            if len(inter_list) == 0:
                setattr(self, "fc1", Linear_BN(nb_in, nb_out, no_act_last))
            else:
                setattr(self, "fc1", Linear_BN(nb_in, inter_list[0]))
                for lidx, (nin, nout) in enumerate(zip(inter_list[:-1], inter_list[1:])):
                    setattr(self, "fc{}".format(lidx+2), Linear_BN(nin, nout))
                setattr(self, "fc{}".format(self.nb_layers), Linear_BN(inter_list[-1], nb_out, no_act_last))

    def __call__(self, h, no_act_last=False):
        for idx in range(1, self.nb_layers + 1, 1):
            h = getattr(self, "fc{}".format(idx))(h)
        return h


class Conv_Module(chainer.Chain):
    def __init__(self, nb_in, nb_out, inter_list=[], no_act_last=False):
        super(Conv_Module, self).__init__()
        self.nb_layers = len(inter_list) + 1
        with self.init_scope():
            if len(inter_list) == 0:
                setattr(self, "layer1", Conv_BN(nb_in, nb_out, no_bn=no_act_last))
            else:
                setattr(self, "layer1", Conv_BN(nb_in, inter_list[0]))
                for lidx, (nin, nout) in enumerate(zip(inter_list[:-1], inter_list[1:])):
                    setattr(self, "layer{}".format(lidx+2), Conv_BN(nin, nout))
                setattr(self, "layer{}".format(self.nb_layers), Conv_BN(inter_list[-1], nb_out, no_bn=no_act_last))

    def __call__(self, h):
        for idx in range(1, self.nb_layers + 1, 1):
            h = getattr(self, "layer{}".format(idx))(h)
        return h


class DEncoder(chainer.Chain):
    def __init__(self, nb_inputs, channel_list, dilate_list, pad=0):
        super(DEncoder, self).__init__()
        self.nb_layers = len(channel_list)
        channel_list = [nb_inputs] + channel_list
        for idx, (nb_in, nb_out, dilate) in enumerate(zip(channel_list[:-1], channel_list[1:], dilate_list)):
            self.add_link("conv{}".format(idx), DConv_BN(nb_in, nb_out, ksize, ksize))

    def __call__(self, x):
        h = F.swapaxes(x, 1, 2)  # (B, D, L)
        h = F.expand_dims(h, 2)
        for idx in range(self.nb_layers):
            h = getattr(self, "conv{}".format(idx))(h)
        return h[...,0]


class Encoder(chainer.Chain):
    def __init__(self, nb_inputs, channel_list, ksize_list, pad_list=[]):
        super(Encoder, self).__init__()
        self.nb_layers = len(channel_list)
        channel_list = [nb_inputs] + channel_list
        if len(pad_list) == 0:
            pad_list = [0 for _ in range(len(ksize_list))]
        for idx, (nb_in, nb_out, ksize, pad) in enumerate(zip(channel_list[:-1], channel_list[1:], ksize_list, pad_list)):
            self.add_link("conv{}".format(idx), Conv_BN(nb_in, nb_out, ksize, pad))

    def __call__(self, x):
        h = F.swapaxes(x, 1, 2)  # (B, D, L)
        for idx in range(self.nb_layers):
            h = getattr(self, "conv{}".format(idx))(h)
        return h


class Decoder(chainer.Chain):
    def __init__(self, nb_inputs, channel_list, ksize_list, no_act_last=False):
        super(Decoder, self).__init__()
        self.nb_layers = len(channel_list)
        self.no_act_last = no_act_last
        channel_list = channel_list + [nb_inputs]
        for idx, (nb_in, nb_out, ksize) in enumerate(zip(channel_list[:-1], channel_list[1:], ksize_list[::-1])):
            self.add_link("deconv{}".format(idx), L.DeconvolutionND(1, nb_in, nb_out, ksize))
            if no_act_last and idx == self.nb_layers - 1:
                continue
            self.add_link("bn{}".format(idx), L.BatchNormalization(nb_out))

    def __call__(self, h):
        for idx in range(self.nb_layers):
            if self.no_act_last and idx == self.nb_layers - 1:
                h = getattr(self, "deconv{}".format(idx))(h)
            else:
                h = F.relu(getattr(self, "bn{}".format(idx))(getattr(self, "deconv{}".format(idx))(h)))
        return h
