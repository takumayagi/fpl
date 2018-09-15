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

import os

import time
import json
import argparse
import joblib
from box import Box

import numpy as np
import cv2

from utils.dataset import SceneDatasetForAnalysis
from utils.plot import draw_line, draw_dotted_line, draw_x

from mllogger import MLLogger
logger = MLLogger(init=False)


def get_traj_type(hip_dist):
    # 0: front 1: back 2: cross 3:other
    if hip_dist < 0.25:
        traj_type = 2
    elif front_ratio > 0.75:
        traj_type = 0
    elif front_ratio < 0.25:
        traj_type = 1
    else:
        traj_type = 3

    return traj_type


if __name__ == "__main__":
    """
    Plot trajectory from prediction (prediction.json)
    By default, blue is input, red is gt, green is prediction

    Input:
    * fold_path: experiment id (e.g. 5fold_yymmss_HHMMSS/yymmss_HHMMSS)
    * traj_type: trajectory type to output (0: front, 1: back, 2: across, 3:other)
    * video_ids: only plot specified videos (e.g. GOPR0235U20)
    * img_ratio: output image ratio (default: 0.5, 640x480)
    * no_pred: do not plot prediction
    * debug: if True, do not predict any images
    * output_dir: directory to plot images

    Output:
        plot images to <output_dir>/yymmss_HHMMSS/pred_<vid>_<frame>_<pid>_[1-5].jpg
    """
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('fold_path', type=str)
    parser.add_argument('--traj_type', type=int, default=-1)
    parser.add_argument('--video_ids', type=str, nargs="*", default=[])
    parser.add_argument('--img_ratio', type=float, default=0.5)
    parser.add_argument('--no_pred', action='store_true')

    # Others
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--output_dir', type=str, default="plots")
    args = parser.parse_args()
    start = time.time()

    data_dir = "data"
    logger.initialize(args.output_dir, debug=args.debug)
    logger.info(vars(args))
    save_dir = logger.get_savedir()
    logger.info("Written to {}".format(save_dir))

    # Load json
    prediction_path = os.path.join(args.fold_path, "prediction.json")
    with open(prediction_path) as f:
        data = json.load(f)
        predictions = data["predictions"]
        sargs = Box(data["arguments"])

    print("Loading data...")
    dataset = joblib.load(sargs.in_data)
    valid_split = sargs.eval_split + sargs.nb_splits * 2
    valid_dataset = SceneDatasetForAnalysis(
        dataset, sargs.input_len, sargs.offset_len if "offset" in sargs else 10, sargs.pred_len,
        sargs.width, sargs.height)

    print(len(valid_dataset))

    # Green, Purple, Yellow, Gray, Cyan
    colors = [(0, 255, 0), (255, 42, 107), (0, 255, 255), (163, 163, 163), (192, 192, 0)]

    # Color
    color_ps = (255, 64, 64)
    color_gt = (0, 0, 255)
    color_pr = (0, 255, 0)

    # Line thickness
    thick_circle = 20
    thick_x = 12
    thick = 10

    cnt = 0
    total_error = 0
    for idx in range(len(valid_dataset)):
        past, gt, pose, vid, frame, pid, flipped = valid_dataset.get_example(idx)[:7]
        if len(args.video_ids) > 0 and vid not in args.video_ids:
            continue
        if vid not in predictions or str(frame) not in predictions[vid] \
                or str(pid) not in predictions[vid][str(frame)]:
            continue
        past = past[...,:2]
        gt = gt[...,:2]

        front_cnt = sum([1 if ps[11][0] - ps[8][0] > 0 else 0 for ps in pose])
        hip_dist = np.mean([np.abs(ps[11, 0] - ps[8, 0]) for ps in pose])
        front_ratio = front_cnt / len(pose)

        traj_type = get_traj_type(hip_dist)
        if args.traj_type != -1 and traj_type != args.traj_type:
            continue

        print("Video: {}, pid: {}, frame: {}".format(vid, pid, frame))
        img_dir = os.path.join(data_dir, "pseudo_viz", vid, "images")
        img = cv2.imread(os.path.join(
            img_dir, "rgb_{:05d}.jpg".format(frame + sargs.input_len - 1)))
        img2 = cv2.imread(os.path.join(
            img_dir, "rgb_{:05d}.jpg".format(frame + sargs.input_len + sargs.pred_len - 1)))

        for idx, im in enumerate([img1, img2]):
            # Past
            cv2.circle(im, (int(past[0][0]*1.0), int(past[0][1]*1.0)), thick_circle, color_ps, -1)
            im = draw_line(im, past, color_ps, 1.0, thick)
            im = draw_x(im, past[-1], color_ps, 1.0, thick)

            # Ground truth
            cv2.circle(im, (int(gt[0][0]*1.0), int(gt[0][1]*1.0)), thick_circle, color_gt, -1)
            im = draw_dotted_line(im, gt, color_gt, 1.0, thick)
            im = draw_x(im, gt[-1], color_gt, 1.0, thick_x)

            vid, frame, pid, flipped, py, pe, pp, err, traj_type = predictions[vid][str(frame)][str(pid)]
            predicted = np.array(py)[...,:2]
            total_error += err

            if not args.no_pred:
                # Prediction
                if predicted.ndim == 1:
                    im = draw_x(im, predicted, color_pr, 1.0, thick_x)
                else:
                    im = draw_dotted_line(im, predicted, color_pr, 1.0, thick)
                    im = draw_x(im, predicted[-1], color_pr, 1.0, thick_x)

            if not args.debug:
                if args.img_ratio < 1.0:
                    im = cv2.resize(im, None, fx=args.img_ratio, fy=args.img_ratio)
                cv2.imwrite(os.path.join(save_dir, "pred_{}_{}_{}_{}.jpg".format(vid, frame, pid, idx+1)), im)

        cnt += 1

    print(total_error / cnt)
    logger.info("Elapsed time: {} (s), Saved at {}".format(time.time()-start, save_dir))
