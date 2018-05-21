#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import argparse

import numpy as np
from chainer import serializers

from models.cnn import CNN, CNN_Pose, CNN_Ego, CNN_Ego_Pose
from logging import getLogger

logger = getLogger('main')


def get_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--in_data', type=str)
    parser.add_argument('--nb_train', type=int, default=-1)
    parser.add_argument('--nb_jobs', type=int, default=8)
    parser.add_argument('--nb_splits', type=int, default=5)
    parser.add_argument('--eval_split', type=int, default=0)

    # Model
    parser.add_argument('--model', type=str, default="cnn")
    parser.add_argument('--input_len', type=int, default=10)
    parser.add_argument('--offset_len', type=int, default=10)
    parser.add_argument('--pred_len', type=int, default=10)
    parser.add_argument('--inter_list', type=int, nargs="*", default=[])
    parser.add_argument('--last_list', type=int, nargs="*", default=[])
    parser.add_argument('--channel_list', type=int, nargs="*", default=[])
    parser.add_argument('--deconv_list', type=int, nargs="*", default=[])
    parser.add_argument('--ksize_list', type=int, nargs="*", default=[])
    parser.add_argument('--dc_ksize_list', type=int, nargs="*", default=[])
    parser.add_argument('--pad_list', type=int, nargs="*", default=[])

    # Training
    parser.add_argument('--nb_iters', type=int, default=10000)
    parser.add_argument('--iter_snapshot', type=int, default=1000)
    parser.add_argument('--iter_display', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr_step_list', type=float, nargs="*", default=[])
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--resume', type=str, default="")

    # Others
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--root_dir', type=str, default="outputs")
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=960)
    parser.add_argument('--nb_grids', type=int, default=6)
    parser.add_argument('--seed', type=int, default=1701)
    parser.add_argument('--ego_type', type=str, default="sfm")

    return parser.parse_args()


def get_model(args):
    mean = np.array([640., 476.23620605, 88.2875590389])
    std = np.array([227.59802246, 65.00177002, 52.7303319245])
    if "scale" not in args.model:
        mean, std = mean[:2], std[:2]

    logger.info("Mean: {}, std: {}".format(mean, std))
    if args.model == "cnn" or args.model == "cnn_scale":
        model = CNN(mean, std, args.gpu, args.channel_list, args.deconv_list, args.ksize_list,
                    args.dc_ksize_list, args.inter_list, args.last_list, args.pad_list)
    elif args.model == "cnn_pose" or args.model == "cnn_pose_scale":
        model = CNN_Pose(mean, std, args.gpu, args.channel_list, args.deconv_list, args.ksize_list,
                         args.dc_ksize_list, args.inter_list, args.last_list, args.pad_list)
    elif args.model == "cnn_ego" or args.model == "cnn_ego_scale":
        model = CNN_Ego(mean, std, args.gpu, args.channel_list, args.deconv_list, args.ksize_list,
                        args.dc_ksize_list, args.inter_list, args.last_list, args.pad_list, args.ego_type)
    elif args.model == "cnn_ego_pose" or args.model == "cnn_ego_pose_scale":
        model = CNN_Ego_Pose(mean, std, args.gpu, args.channel_list, args.deconv_list, args.ksize_list,
                             args.dc_ksize_list, args.inter_list, args.last_list, args.pad_list, args.ego_type)
    else:
        logger.info("Invalid argument: model={}".format(args.model))
        exit(1)

    if args.resume != "":
        serializers.load_npz(args.resume, model)

    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    return model


def write_prediction(pred_dict, batch, pred_y):
    for idx, (sample, py) in enumerate(zip(batch, pred_y)):
        past, ground_truth, pose, vid, frame, pid, flipped, egomotion, scale, mag, size = sample
        frame, pid = str(frame), str(pid)

        err = np.linalg.norm(py - ground_truth, axis=1)[-1]
        front_cnt = sum([1 if ps[11][0] - ps[8][0] > 0 else 0 for ps in pose])
        hip_dist = np.mean([np.abs(ps[11, 0] - ps[8, 0]) for ps in pose])
        front_ratio = front_cnt / len(pose)

        # 0: front 1: back 2: cross 3:other
        if hip_dist < 0.25:
            traj_type = 2
        elif front_ratio > 0.75:
            traj_type = 0
        elif front_ratio < 0.25:
            traj_type = 1
        else:
            traj_type = 3

        if vid not in pred_dict:
            pred_dict[vid] = {}
        if frame not in pred_dict[vid]:
            pred_dict[vid][frame] = {}

        result = [vid, frame, pid, flipped, py, None, None, err, traj_type]
        result = list(map(lambda x: x.tolist() if type(x).__module__ == "numpy" else x, result))
        pred_dict[vid][frame][pid] = result
