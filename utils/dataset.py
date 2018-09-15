#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import numpy as np
import quaternion
from functools import reduce
from operator import add

from chainer.dataset import dataset_mixin


def parse_data_CV(data, split_list, input_len, offset_len, pred_len, nb_train):
    if type(split_list) != list:
        split_list = [split_list]
    trajectories = data["trajectories"].astype(np.float32)
    splits = data["splits"]
    traj_len = offset_len + pred_len
    idxs_past = np.arange(offset_len - input_len, offset_len)
    idxs_pred = np.arange(offset_len, traj_len)
    idxs_both = np.arange(offset_len - input_len, traj_len)
    idxs_split = reduce(add, [splits == s for s in split_list])
    if nb_train != -1:
        idxs = np.random.choice(np.arange(sum(idxs_split)), nb_train)
    data = list(map(lambda x: None if x is None else x[idxs] if nb_train != -1 else x, [
        trajectories[idxs_split][:, idxs_past, :],
        trajectories[idxs_split][:, idxs_pred, :],
        np.array(data["video_ids"][idxs_split]),
        np.array(data["frames"][idxs_split]),
        np.array(data["person_ids"][idxs_split]),
        np.array(data["poses"][idxs_split][:, idxs_both]),
        np.array(data["turn_mags"][idxs_split]),
        np.array(data["trans_mags"][idxs_split]),
        np.array(data["masks"][idxs_split][:, idxs_pred], dtype=np.float32) if "masks" in data else None
    ]))
    return data + [offset_len - input_len]


def accumulate_egomotion(rots, vels):
    # Accumulate translation and rotation
    egos = []
    qa = np.quaternion(1, 0, 0, 0)
    va = np.array([0., 0., 0.])
    for rot, vel in zip(rots, vels):
        vel_rot = quaternion.rotate_vectors(qa, vel)
        va += vel_rot
        qa = qa * quaternion.from_rotation_vector(rot)
        egos.append(np.concatenate(
            (quaternion.as_rotation_vector(qa), va), axis=0))
    return egos


class SceneDatasetCV(dataset_mixin.DatasetMixin):
    def __init__(self, data, input_len, offset_len, pred_len, width, height,
                 data_dir, split_list, nb_train=-1, flip=False, use_scale=False, ego_type="sfm"):
        self.X, self.Y, self.video_ids, self.frames, self.person_ids, \
            raw_poses, self.turn_mags, self.trans_mags, self.masks, self.offset = \
            parse_data_CV(data, split_list, input_len, offset_len, pred_len, nb_train)

        # (N, T, D, 3)
        past_len = input_len
        poses = raw_poses[:, :, :, :2]
        spine = (poses[:, :, 8:9, :2] + poses[:, :, 11:12, :2]) / 2
        neck = poses[:, :, 1:2, :2]
        scales_all = np.linalg.norm(neck - spine, axis=3)  # (N, T, 1)
        scales_all[scales_all < 1e-8] = 1e-8  # Avoid ZerodivisionError
        poses = (poses - spine) / scales_all[:, :, :, np.newaxis]  # Normalization

        self.poses = poses.astype(np.float32)
        self.scales_all = scales_all[:, :, 0].astype(np.float32)
        self.scales = self.scales_all[:, -pred_len-1]

        # (x, y) -> (x, y, s)
        if use_scale:
            self.X = np.concatenate((self.X, self.scales_all[:, :past_len, np.newaxis]), axis=2)
            self.Y = np.concatenate((self.Y, self.scales_all[:, past_len:past_len+pred_len, np.newaxis]), axis=2)

        self.width = width
        self.height = height
        self.data_dir = data_dir
        self.flip = flip
        self.nb_inputs = self.X.shape[2]
        self.ego_type = ego_type

        self.egomotions = []
        for vid, frame in zip(self.video_ids, self.frames):
            ego_dict = data["egomotion_dict"][vid]
            if ego_type == "sfm":  # SfMLearner
                rots, vels = [], []
                for frame in range(frame + self.offset, frame + self.offset + past_len + pred_len):
                    key = "rgb_{:05d}.jpg".format(frame)
                    key_m1 = "rgb_{:05d}.jpg".format(frame-1)
                    rot_vel = ego_dict[key] if key in ego_dict \
                        else ego_dict[key_m1] if key_m1 in ego_dict \
                        else np.array([0., 0., 0., 0., 0., 0.])
                    rots.append(rot_vel[:3])
                    vels.append(rot_vel[3:6])

                egos = accumulate_egomotion(rots[:past_len], vels[:past_len]) + \
                    accumulate_egomotion(rots[past_len:past_len+pred_len], vels[past_len:past_len+pred_len])
            else:  # Grid optical flow
                raw_egos = [ego_dict["rgb_{:05d}.jpg"].format(f) for f in
                            range(frame + self.offset, frame + self.offset + past_len + pred_len)]
                egos = [np.sum(raw_egos[:idx+1], axis=0) for idx in range(past_len)] + \
                    [np.sum(raw_egos[past_len:past_len+idx+1], axis=0) for idx in range(pred_len)]
            self.egomotions.append(egos)
        self.egomotions = np.array(self.egomotions).astype(np.float32)

    def __len__(self):
        return len(self.frames)

    def get_example(self, i):
        X = self.X[i].copy()
        Y = self.Y[i].copy()
        poses = self.poses[i].copy()
        egomotions = self.egomotions[i].copy()

        horizontal_flip = np.random.random() < 0.5 if self.flip else False
        if horizontal_flip:
            X[:, 0] = self.width - X[:, 0]
            Y[:, 0] = self.width - Y[:, 0]
            poses[:, :, 0] = -poses[:, :, 0]
            if self.ego_type == "sfm":
                egomotions[:, [1, 2, 3]] = -egomotions[:, [1, 2, 3]]
            else:
                nb_dims = egomotions.shape[1]
                egomotions[:, range(0, nb_dims, 2)] = -egomotions[:, range(0, nb_dims, 2)]

        return X, Y, poses, self.video_ids[i], self.frames[i], self.person_ids[i], \
            horizontal_flip, egomotions, self.scales[i], self.turn_mags[i], self.scales_all[i]


class SceneDatasetForAnalysis(dataset_mixin.DatasetMixin):
    """
    Dataset class only for plot
    """
    def __init__(self, data, input_len, offset_len, pred_len, width, height):

        self.X, self.Y, self.video_ids, self.frames, self.person_ids, \
            raw_poses, self.turn_mags, self.trans_mags, self.masks, self.offset = \
            parse_data_CV(data, list(range(5, 10, 1)), input_len, offset_len, pred_len, -1)

        # (N, T, D, 3)
        past_len = input_len
        poses = raw_poses[:, :, :, :2]
        spine = (poses[:, :, 8:9, :2] + poses[:, :, 11:12, :2]) / 2
        neck = poses[:, :, 1:2, :2]
        sizes = np.linalg.norm(neck - spine, axis=3)  # (N, T, 1)
        poses = (poses - spine) / sizes[:, :, :, np.newaxis]  # Normalization

        self.poses = poses.astype(np.float32)
        self.sizes = sizes[:, :, 0].astype(np.float32)
        self.scales = self.sizes[:, -pred_len-1]

        self.X = np.concatenate((self.X, self.sizes[:, :past_len, np.newaxis]), axis=2)
        self.Y = np.concatenate((self.Y, self.sizes[:, past_len:past_len+pred_len, np.newaxis]), axis=2)

        self.width = width
        self.height = height
        self.nb_inputs = self.X.shape[2]

    def __len__(self):
        return len(self.frames)

    def get_example(self, i):

        X = self.X[i].copy()
        Y = self.Y[i].copy()
        poses = self.poses[i].copy()

        return X, Y, poses, self.video_ids[i], self.frames[i], self.person_ids[i], \
            self.scales[i], self.turn_mags[i], self.sizes[i]
