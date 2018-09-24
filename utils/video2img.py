#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import os
import sys
import time
import cv2


def video2img(video_path):
    """
    Convert video to jpg images
    """
    start = time.time()
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    target_dir = os.path.join("data/videos", video_id, "images")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Broken or invalid video. Quit...")
        exit(1)

    frame_cnt, cnt = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
        cnt += 1

        frame = cv2.resize(frame, (1280, 720))
        cv2.imwrite(os.path.join(target_dir, "rgb_{0:05d}.jpg".format(frame_cnt)), frame)
        frame_cnt += 1

    cap.release()
    print("Outdir: {}, number of frame: {}".format(target_dir, frame_cnt))
    print("Elapsed time: {} s".format(time.time()-start))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python video2img.py <video_path>")
        exit(1)

    print(sys.argv[1])
    video2img(sys.argv[1])
