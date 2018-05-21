#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

from __future__ import print_function
from __future__ import division
from six.moves import range

import os
import argparse
import json
import time
import joblib
import datetime

import numpy as np


def read_sfmlearn(ego_path, flip):
    """
    Right-hand coordinate (following SfMLearn paper).
    X: Pitch, Y: Yaw, Z: Roll
    """
    ego_dict = {}
    with open(ego_path, "r") as f:
        for line in f:
            strings = line.strip("\r\n").split(",")
            key = strings[0]
            vx, vy, vz, rx, ry, rz = list(map(lambda x:float(x), strings[1:]))
            if flip:
                ego_dict[key] = np.array([rx, -ry, -rz, -vx, vy, vz])
            else:
                ego_dict[key] = np.array([rx, ry, rz, vx, vy, vz])
    return ego_dict


def read_gridflow(ego_path, flip):
    ego_dict = {}
    with open(ego_path) as f:
        for line in f:
            strings = line.strip("\r\n").split(",")
            key = strings[0]
            grid_flow = np.array(list(map(float, strings[1:])))
            if flip:
                grid_flow[range(0, len(grid_flow), 2)] *= -1
            ego_dict[key] = grid_flow
    return ego_dict


def read_vid_list(indir_list):
    vid_list, nb_image_list = [], []
    blacklist = {}
    with open(indir_list) as f:
        for line in f:
            strs = line.strip("\n").split(",")
            if strs[0].startswith("#"):
                continue
            vid_list.append(strs[0])
            nb_image_list.append(int(strs[1]))
            if len(strs) > 2:
                blacklist[strs[0]] = strs[2:]

    return vid_list, nb_image_list, blacklist


if __name__ == "__main__":
    """
    Create dataset for training
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('indir_list', type=str)
    parser.add_argument('--traj_length', type=int, default=30)
    parser.add_argument('--traj_skip', type=int, default=2)
    parser.add_argument('--traj_skip_test', type=int, default=5)
    parser.add_argument('--nb_splits', type=int, default=3)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=960)
    parser.add_argument('--skip', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1701)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--no_flip', action="store_true")
    parser.add_argument('--ego_type', type=str, default="sfm")
    args = parser.parse_args()
    start = time.time()

    data_dir = "data/"

    date = datetime.datetime.now()
    out_fn = "dataset/{}_{}_{}_sp{}_{}_{}.joblib".format(
        os.path.splitext(os.path.basename(args.indir_list))[0], date.strftime('%y%m%d_%H%M%S'),
        args.traj_length, args.nb_splits,
        args.traj_skip, args.traj_skip_test)
    h, w = args.height, args.width

    vid_list, nb_image_list, blacklist = read_vid_list(args.indir_list)
    print("Number of videos: {}".format(len(vid_list)))

    shuffled_ids = np.copy(vid_list)
    np.random.seed(args.seed)
    np.random.shuffle(shuffled_ids)
    ratio = 1 / args.nb_splits

    nb_videos = len(vid_list)
    split_dict = {}
    for sp, st in enumerate(np.arange(0, 1, ratio)):
        print(shuffled_ids[int(nb_videos*st):int(nb_videos*(st+ratio))])
        for vid in shuffled_ids[int(nb_videos*st):int(nb_videos*(st+ratio))]:
            split_dict[vid] = sp

    video_ids, frames, person_ids, trajectories, poses, splits, \
        turn_mags, trans_mags, pids = [], [], [], [], [], [], [], [], []
    start_dict, traj_dict, pose_dict = {}, {}, {}

    total_frames = 0
    egomotion_dict = {}
    nb_trajs = 0
    nb_traj_list = [0 for _ in range(args.nb_splits)]
    for video_id, nb_images in zip(vid_list, nb_image_list):
        trajectory_path = os.path.join(
            data_dir, "trajectories/{}_trajectories_dynamic.json".format(video_id))
        with open(trajectory_path, "r") as f:
            trajectory_dict = json.load(f)

        total_frames += nb_images

        if args.ego_type == "sfm":
            egomotion_path = os.path.join(data_dir, "egomotions/{}_egomotion.csv".format(video_id))
            egomotion_dict[video_id] = read_sfmlearn(egomotion_path, False)
        else:
            egomotion_path = os.path.join(data_dir, "egomotions/{}_gridflow_24.csv".format(video_id))
            egomotion_dict[video_id] = read_gridflow(egomotion_path, False)

        lr_mag_list = [abs(v[1]) for k, v in sorted(egomotion_dict[video_id].items())]

        start_dict[video_id] = {}
        traj_dict[video_id] = {}
        pose_dict[video_id] = {}

        # pid search
        pids = []
        for pid, info in trajectory_dict.items():
            if video_id in blacklist and pid in blacklist[video_id]:
                print("Blacklist: {} {}".format(video_id, pid))
                continue
            if "traj_sm" not in info:
                continue
            traj = info["traj_sm"]
            pose = info["pose_sm"]
            if len(traj) < args.traj_length:
                continue
            front_cnt = sum([1 if ps[11][0] - ps[8][0] > 0 else 0 for ps in pose])
            pids.append(pid)

        pid_cnt = len(pids)
        nb_trajs += pid_cnt

        traj_cnt = 0
        for pid in pids:
            info = trajectory_dict[pid]
            t_s = info["start"]
            traj = info["traj_sm"]
            pose = info["pose_sm"]
            pid = int(pid)

            if t_s <= 2 or t_s + len(traj) >= nb_images - 1:
                continue

            start_dict[video_id][pid] = info["start"]
            traj_dict[video_id][pid] = info["traj_sm"]
            pose_dict[video_id][pid] = info["pose_sm"]

            def add_sample(split):
                x_max = np.max([x[0] for x in traj[tidx+args.traj_length//2:tidx+args.traj_length]])
                x_min = np.min([x[0] for x in traj[tidx+args.traj_length//2:tidx+args.traj_length]])
                y_max = np.max([x[1] for x in traj[tidx+args.traj_length//2:tidx+args.traj_length]])
                y_min = np.min([x[1] for x in traj[tidx+args.traj_length//2:tidx+args.traj_length]])
                trans_mag = np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
                turn_mag = np.max(lr_mag_list[tidx + t_s - 1 + args.traj_length // 2:tidx + t_s - 1 + args.traj_length])

                frames.append(tidx + t_s)
                person_ids.append(pid)
                trajectories.append(traj[tidx:tidx+args.traj_length])
                poses.append(pose[tidx:tidx+args.traj_length])
                splits.append(split)
                video_ids.append(video_id)

                turn_mags.append(turn_mag)
                trans_mags.append(trans_mag)

            # Training set (split 0-4)
            for tidx in range(0, len(traj) - args.traj_length + 1, args.traj_skip):
                add_sample(split_dict[video_id])
                traj_cnt += 1

            # Evaluation set (split 5-9)
            for tidx in range(0, len(traj) - args.traj_length + 1, args.traj_skip_test):
                add_sample(split_dict[video_id] + args.nb_splits)

        print(video_id, nb_images, pid_cnt, traj_cnt)
        nb_traj_list[split_dict[video_id]] += traj_cnt

    if not os.path.exists(os.path.dirname(out_fn)):
        os.makedirs(os.path.dirname(out_fn))

    splits = np.array(splits)

    print("Total number of frames: {}".format(total_frames))
    result_str = ""
    for sp in range(args.nb_splits):
        result_str += "Split {}: {}".format(sp + 1, sum(splits == sp))
        if sp < args.nb_splits - 1:
            result_str += ", "

    print(result_str)
    print("Number of tracklets:", nb_trajs)
    print("Number of samples:", nb_traj_list)
    if not args.debug:
        joblib.dump({
            "video_ids": np.array(video_ids),
            "frames": np.array(frames),
            "person_ids": np.array(person_ids),
            "trajectories": np.array(trajectories),
            "poses": np.array(poses),
            "splits": splits,
            "egomotion_dict": egomotion_dict,
            "turn_mags": np.array(turn_mags),
            "trans_mags": np.array(trans_mags),
            "start_dict": start_dict,
            "traj_dict": traj_dict,
            "pose_dict": pose_dict
        }, out_fn)

    print("Written to {}".format(out_fn))
    print("Completed. Elapsed time: {} (s)".format(time.time()-start))
