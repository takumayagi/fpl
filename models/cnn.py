#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import numpy as np

import chainer
import chainer.functions as F
import cupy
from chainer import Variable, cuda

from logging import getLogger
logger = getLogger("main")

from models.module import Conv_Module, Encoder, Decoder


class CNNBase(chainer.Chain):
    def __init__(self, mean, std, gpu):
        super(CNNBase, self).__init__()
        self._mean = mean
        self._std = std
        self.nb_inputs = len(mean)
        self.target_idx = -1
        self.mean = Variable(cuda.to_gpu(mean.astype(np.float32), gpu))
        self.std = Variable(cuda.to_gpu(std.astype(np.float32), gpu))

    def _prepare_input(self, inputs):
        pos_x, pos_y, poses, egomotions = inputs[:4]
        if pos_y.data.ndim == 2:
            pos_x = F.expand_dims(pos_x, 0)
            pos_y = F.expand_dims(pos_y, 0)
            if egomotions is not None:
                egomotions = F.expand_dims(egomotions, 0)
            poses = F.expand_dims(poses, 0)

        # Pos
        x = (pos_x - F.broadcast_to(self.mean, pos_x.shape)) / F.broadcast_to(self.std, pos_x.shape)
        y = (pos_y - F.broadcast_to(self.mean, pos_y.shape)) / F.broadcast_to(self.std, pos_y.shape)
        y = y - F.broadcast_to(x[:, -1:, :], pos_y.shape)

        # Ego
        past_len = pos_x.shape[1]
        if egomotions is not None:
            ego_x = egomotions[:, :past_len, :]
            ego_y = egomotions[:, past_len:, :]

        # Pose
        poses = F.reshape(poses, (poses.shape[0], poses.shape[1], -1))
        pose_x = poses[:, :past_len, :]
        pose_y = poses[:, past_len:, :]

        if egomotions is not None:
            return x, y, x[:, -1, :], ego_x, ego_y, pose_x, pose_y
        else:
            return x, y, x[:, -1, :], None, None, pose_x, pose_y

    def predict(self, inputs):
        return self.__call__(inputs)


class CNN(CNNBase):
    def __init__(self, mean, std, gpu, channel_list, dc_channel_list, ksize_list,
                 dc_ksize_list, inter_list, last_list, pad_list):
        super(CNN, self).__init__(mean, std, gpu)
        if len(ksize_list) > 0 and len(dc_ksize_list) == 0:
            dc_ksize_list = ksize_list
        with self.init_scope():
            self.pos_encoder = Encoder(self.nb_inputs, channel_list, ksize_list, pad_list)
            self.pos_decoder = Decoder(dc_channel_list[-1], dc_channel_list, dc_ksize_list[::-1])
            self.inter = Conv_Module(channel_list[-1], dc_channel_list[0], inter_list)
            self.last = Conv_Module(dc_channel_list[-1], self.nb_inputs, last_list, True)

    def __call__(self, inputs):
        pos_x, pos_y, offset_x, ego_x, ego_y, pose_x, pose_y = self._prepare_input(inputs)
        batch_size, past_len, _ = pos_x.shape

        h = self.pos_encoder(pos_x)
        h = self.inter(h)
        h = self.pos_decoder(h)
        pred_y = self.last(h)
        pred_y = F.swapaxes(pred_y, 1, 2)
        pred_y = pred_y[:, :pos_y.shape[1], :]
        loss = F.mean_squared_error(pred_y, pos_y)

        pred_y = pred_y + F.broadcast_to(F.expand_dims(offset_x, 1), pred_y.shape)
        pred_y = cuda.to_cpu(pred_y.data) * self._std + self._mean
        return loss, pred_y, None


class CNN_Ego(CNNBase):
    def __init__(self, mean, std, gpu, channel_list, dc_channel_list, ksize_list,
                 dc_ksize_list, inter_list, last_list, pad_list, ego_type):
        super(CNN_Ego, self).__init__(mean, std, gpu)
        ego_dim = 6 if ego_type == "sfm" else 96 if ego_type == "grid" else 24
        if len(ksize_list) > 0 and len(dc_ksize_list) == 0:
            dc_ksize_list = ksize_list
        with self.init_scope():
            self.pos_encoder = Encoder(self.nb_inputs, channel_list, ksize_list, pad_list)
            self.ego_encoder = Encoder(ego_dim, channel_list, ksize_list, pad_list)
            self.pos_decoder = Decoder(dc_channel_list[-1], dc_channel_list, dc_ksize_list[::-1])
            self.inter = Conv_Module(channel_list[-1]*2, dc_channel_list[0], inter_list)
            self.last = Conv_Module(dc_channel_list[-1], self.nb_inputs, last_list, True)

    def __call__(self, inputs):
        pos_x, pos_y, offset_x, ego_x, ego_y, pose_x, pose_y = self._prepare_input(inputs)
        batch_size, past_len, _ = pos_x.shape

        h_pos = self.pos_encoder(pos_x)
        h_ego = self.ego_encoder(ego_x)
        h = F.concat((h_pos, h_ego), axis=1)  # (B, C, 2)
        h = self.inter(h)
        h_pos = self.pos_decoder(h)
        pred_y = self.last(h_pos)  # (B, 10, C+6+28)
        pred_y = F.swapaxes(pred_y, 1, 2)
        pred_y = pred_y[:, :pos_y.shape[1], :]
        loss = F.mean_squared_error(pred_y, pos_y)

        pred_y = pred_y + F.broadcast_to(F.expand_dims(offset_x, 1), pred_y.shape)
        pred_y = cuda.to_cpu(pred_y.data) * self._std + self._mean
        return loss, pred_y, None


class CNN_Pose(CNNBase):
    def __init__(self, mean, std, gpu, channel_list, dc_channel_list, ksize_list,
                 dc_ksize_list, inter_list, last_list, pad_list):
        super(CNN_Pose, self).__init__(mean, std, gpu)
        if len(ksize_list) > 0 and len(dc_ksize_list) == 0:
            dc_ksize_list = ksize_list
        with self.init_scope():
            self.pos_encoder = Encoder(self.nb_inputs, channel_list, ksize_list, pad_list)
            self.pose_encoder = Encoder(36, channel_list, ksize_list, pad_list)
            self.pos_decoder = Decoder(dc_channel_list[-1], dc_channel_list, dc_ksize_list[::-1])
            self.inter = Conv_Module(channel_list[-1]*2, dc_channel_list[0], inter_list)
            self.last = Conv_Module(dc_channel_list[-1], self.nb_inputs, last_list, True)

    def __call__(self, inputs):
        pos_x, pos_y, offset_x, ego_x, ego_y, pose_x, pose_y = self._prepare_input(inputs)
        batch_size, past_len, _ = pos_x.shape

        h_pos = self.pos_encoder(pos_x)
        h_pose = self.pose_encoder(pose_x)
        h = F.concat((h_pos, h_pose), axis=1)  # (B, C, 2)
        h = self.inter(h)
        h_pos = self.pos_decoder(h)
        pred_y = self.last(h_pos)  # (B, 10, C+6+28)
        pred_y = F.swapaxes(pred_y, 1, 2)
        pred_y = pred_y[:, :pos_y.shape[1], :]
        loss = F.mean_squared_error(pred_y, pos_y)

        pred_y = pred_y + F.broadcast_to(F.expand_dims(offset_x, 1), pred_y.shape)
        pred_y = cuda.to_cpu(pred_y.data) * self._std + self._mean
        return loss, pred_y, None


class CNN_Ego_Pose(CNNBase):
    def __init__(self, mean, std, gpu, channel_list, dc_channel_list, ksize_list,
                 dc_ksize_list, inter_list, last_list, pad_list, ego_type):
        super(CNN_Ego_Pose, self).__init__(mean, std, gpu)
        ego_dim = 6 if ego_type == "sfm" else 96 if ego_type == "grid" else 24
        if len(ksize_list) > 0 and len(dc_ksize_list) == 0:
            dc_ksize_list = ksize_list
        with self.init_scope():
            self.pos_encoder = Encoder(self.nb_inputs, channel_list, ksize_list, pad_list)
            self.ego_encoder = Encoder(ego_dim, channel_list, ksize_list, pad_list)
            self.pose_encoder = Encoder(36, channel_list, ksize_list, pad_list)
            self.pos_decoder = Decoder(dc_channel_list[-1], dc_channel_list, dc_ksize_list[::-1])
            self.inter = Conv_Module(channel_list[-1]*3, dc_channel_list[0], inter_list)
            self.last = Conv_Module(dc_channel_list[-1], self.nb_inputs, last_list, True)

    def __call__(self, inputs):
        pos_x, pos_y, offset_x, ego_x, ego_y, pose_x, pose_y = self._prepare_input(inputs)
        batch_size, past_len, _ = pos_x.shape

        h_pos = self.pos_encoder(pos_x)
        h_pose = self.pose_encoder(pose_x)
        h_ego = self.ego_encoder(ego_x)
        h = F.concat((h_pos, h_pose, h_ego), axis=1)  # (B, C, 2)
        h = self.inter(h)
        h_pos = self.pos_decoder(h)
        pred_y = self.last(h_pos)  # (B, 10, C+6+28)
        pred_y = F.swapaxes(pred_y, 1, 2)
        pred_y = pred_y[:, :pos_y.shape[1], :]
        loss = F.mean_squared_error(pred_y, pos_y)

        pred_y = pred_y + F.broadcast_to(F.expand_dims(offset_x, 1), pred_y.shape)
        pred_y = cuda.to_cpu(pred_y.data) * self._std + self._mean
        return loss, pred_y, None
