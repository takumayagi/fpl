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


if __name__ == "__main__":
    """
    Write prediction as video from prediction.json
    By default, blue is input, red is gt, green is prediction

    This method requires vid, frame, pid but if the latter two is unspecified it'll give you candidates

    Input:
    * fold_path: fold id (e.g. 5fold_yymmss_HHMMSS/yymmss_HHMMSS)
    * traj_type: trajectory type to output (0: front, 1: back, 2: across, 3:other)
    * img_ratio: output image ratio (default: 0.5, 640x480)
    * no_pred: do not plot prediction
    * debug: if True, do not predict any images
    * output_dir: directory to plot images and videos
    * vid: video id (e.g. GOPR0235U20)
    * frame: video frame number
    * pid: person id number

    Output:
        plot images and videos to <output_dir>/yymmss_HHMMSS/
    """
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('fold_path', type=str)
    parser.add_argument('--traj_type', type=int, default=-1)
    parser.add_argument('--vid', type=str)
    parser.add_argument('--frame', type=int, default=-1)
    parser.add_argument('--pid', type=int, default=-1)
    parser.add_argument('--nb_pause', type=int, default=5)
    parser.add_argument('--img_ratio', type=float, default=0.5)
    parser.add_argument('--ratio', type=float, default=0.7)
    parser.add_argument('--thick', type=int, default=10)
    parser.add_argument('--size_x', type=int, default=12)
    parser.add_argument('--size_circle', type=int, default=20)
    parser.add_argument('--fps', type=float, default=3)
    parser.add_argument('--no_video', action='store_true')
    parser.add_argument('--no_past', action='store_true')
    parser.add_argument('--no_pred', action='store_true')
    parser.add_argument('--no_gt', action='store_true')
    parser.add_argument('--simultaneous', action='store_true')

    # Others
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--output_dir', type=str, default="video_plots")
    args = parser.parse_args()
    start = time.time()

    data_dir = "data"
    logger.initialize(args.output_dir, debug=args.debug)
    logger.info(vars(args))
    save_dir = logger.get_savedir()
    logger.info("Written to {}".format(save_dir))
    nb_pause = args.nb_pause
    output_image_list = []

    prediction_path = os.path.join(args.fold_path, "prediction.json")
    with open(prediction_path) as f:
        data = json.load(f)
        predictions = data["predictions"]
        sargs = Box(data["arguments"])

    print(predictions.keys())
    if args.vid not in predictions:
        print("No prediction found: aborted {} {} {}".format(args.vid, args.frame, args.pid))
        exit(1)

    if str(args.frame) not in predictions[args.vid] or str(args.pid) not in predictions[args.vid][str(args.frame)]:
        print("Specify all vid, frame and pid: aborted {} {} {}".format(args.vid, args.frame, args.pid))
        if args.vid in predictions and str(args.frame) in predictions[args.vid]:
            print("Valid pids: {}".format(sorted(predictions[args.vid][str(args.frame)].keys())))
        elif args.vid in predictions:
            print("Valid frames {}".format(sorted(predictions[args.vid].keys())))
        exit(1)

    print("Loading data...")
    dataset = joblib.load(sargs.in_data)
    valid_split = sargs.eval_split + sargs.nb_splits * 2
    valid_dataset = SceneDatasetForAnalysis(
        dataset, sargs.input_len, sargs.offset_len if "offset" in sargs else 10, sargs.pred_len,
        sargs.width, sargs.height)

    color = (0, 255, 0)
    color_ps = (255, 64, 64)
    color_gt = (0, 0, 255)
    color_pr = (0, 255, 0)
    color_bb = (204, 102, 255)

    # pid search
    if args.frame != -1:
        pid_list = []
        for idx in range(len(valid_dataset)):
            _, _, _, vid, frame, pid = valid_dataset.get_example(idx)[:6]
            if vid == args.vid and frame == args.frame:
                pid_list.append(pid)

        print("Found pid: {}".format(np.unique(pid_list)))
        if len(np.unique(pid_list)) == 1:
            args.pid = pid_list[0]
            print("Continue")
        else:
            exit(1)

    idx_list = []
    for idx in range(len(valid_dataset)):
        _, _, _, vid, frame, pid = valid_dataset.get_example(idx)[:6]
        if vid == args.vid and pid == args.pid:
            idx_list.append((idx, frame))

    if len(idx_list) == 0:
        print("Not found: {} {}".format(args.vid, args.pid))
        exit(1)
    if args.frame == -1:
        print("Candidate: {}".format(idx_list))
        exit(1)
    if args.frame not in map(lambda x: x[1], idx_list):
        print("Wrong frame")
        exit(1)

    selected_idx = idx_list[list(map(lambda x: x[1], idx_list)).index(args.frame)][0]

    past, ground_truth, pose, vid, frame, pid, flipped = valid_dataset.get_example(selected_idx)[:7]
    img_dir = os.path.join(data_dir, "pseudo_viz", vid, "images")

    def draw_past(img_dir, b, offset):
        past, ground_truth, pose, vid, frame, pid, flipped = b[:7]
        im = cv2.imread(os.path.join(img_dir, "rgb_{:05d}.jpg".format(int(frame) + offset)))
        im = cv2.addWeighted(im, args.ratio, im, 0.0, 0)

        # Past
        if offset > 0:
            cv2.circle(im, (int(past[0][0]), int(past[0][1])), args.size_circle, color_ps, -1)
            im = draw_line(im, past[:offset+1], color_ps, 1.0, args.thick)
            im = draw_x(im, past[offset], color_ps, 1.0, args.size_x)

        return im

    def draw_prediction(img_dir, b, offset_g, offset_p, predictions, img_offset, no_pred=False, no_gt=False):
        past, ground_truth, pose, vid, frame, pid, flipped = b[:7]
        vid, frame, pid, flipped, py, pe, pp, err, traj_type = predictions[vid][str(frame)][str(pid)]
        predicted = np.array(py)[...,:2]
        im = cv2.imread(os.path.join(img_dir, "rgb_{:05d}.jpg".format(int(frame) + img_offset)))
        im = cv2.addWeighted(im, args.ratio, im, 0.0, 0)
        cv2.circle(im, (int(past[0][0]), int(past[0][1])), args.size_circle, color_ps, -1)
        im = draw_line(im, past, color_ps, 1.0, args.thick)
        im = draw_x(im, past[-1], color_ps, 1.0, args.size_x)

        if not args.no_gt:
            if offset_g > 0:
                im = draw_dotted_line(im, ground_truth[:offset_g+1], color_gt, 1.0, args.thick)
            if offset_g > -1:
                cv2.circle(im, (int(ground_truth[0][0]), int(ground_truth[0][1])), args.size_circle, color_gt, -1)
                im = draw_x(im, ground_truth[offset_g], color_gt, 1.0, args.size_x)

        if not no_pred:
            if offset_p > 0:
                im = draw_dotted_line(im, predicted[:offset_p+1], color_pr, 1.0, args.thick, 0.7)
            if offset_p > -1:
                im = draw_x(im, predicted[offset_p], color_pr, 1.0, args.size_x)
        return im

    def write_img(oframe, out_img):
        cv2.imwrite("{}/{}_{}_{:02d}.jpg".format(save_dir, args.vid, args.pid, oframe), out_img)

    start_b = valid_dataset.get_example(selected_idx)
    input_len = sargs["input_len"]
    pred_len = sargs["pred_len"]

    # Input
    if not args.no_past:
        for x in range(input_len):
            out = draw_past(img_dir, start_b, x)
            write_img(x, out)
            output_image_list.append(out)

    if not args.no_pred and not args.simultaneous:
        if not args.no_past:
            # Pause
            for x in range(nb_pause):
                out = draw_past(img_dir, start_b, input_len-1)
                output_image_list.append(out)

        # Extend prediction
        for x in range(pred_len):
            out = draw_prediction(img_dir, start_b, -1, x, predictions, input_len - 1)
            write_img(input_len + x, out)
            output_image_list.append(out)

    # Pause
    for x in range(nb_pause):
        out = draw_prediction(img_dir, start_b, -1, pred_len - 1, predictions, input_len - 1, args.no_pred, args.no_gt)
        output_image_list.append(out)

    # Actual future
    for x in range(pred_len):
        if args.simultaneous:
            out = draw_prediction(img_dir, start_b, x, x, predictions, input_len + x, args.no_pred, args.no_gt)
        else:
            out = draw_prediction(img_dir, start_b, x, pred_len - 1, predictions, input_len + x, args.no_pred, args.no_gt)
        write_img(input_len + pred_len + x, out)
        output_image_list.append(out)

    # Pause
    for x in range(nb_pause):
        out = draw_prediction(img_dir, start_b, pred_len - 1, pred_len - 1,
                                      predictions, input_len + pred_len - 1, args.no_pred)
        output_image_list.append(out)

    if not args.no_video:
        if args.no_pred:
            video_name = os.path.join(save_dir, "{}_{}_{}_np.avi".format(args.vid, args.frame, args.pid))
            #video_name = os.path.join(save_dir, "{}_{}_{}_np.mp4".format(args.vid, args.frame, args.pid))
        else:
            video_name = os.path.join(save_dir, "{}_{}_{}.avi".format(args.vid, args.frame, args.pid))
            #video_name = os.path.join(save_dir, "{}_{}_{}.mp4".format(args.vid, args.frame, args.pid))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        width, height = int(sargs.width * args.img_ratio), int(sargs.height * args.img_ratio)
        writer = cv2.VideoWriter(video_name, fourcc, args.fps, (width, height))
        #writer = cv2.VideoWriter(video_name, 0x21, args.fps, (width, height))
        for img in output_image_list:
            img = cv2.resize(img, (width, height))
            writer.write(img)
        writer.release()
        logger.info("Wrote video into {}".format(video_name))

    logger.info("Elapsed time: {} (s), Saved at {}".format(time.time()-start, save_dir))
