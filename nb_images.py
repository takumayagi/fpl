#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import os
import sys
import glob

indir_list = sys.argv[1]
vid_list = []
with open(indir_list) as f:
    for line in f:
        strs = line.strip("\n").split(",")
        vid_list.append(strs[0])

data_dir = os.getenv("TRAJ_DATA_DIR")
for vid in vid_list:
    image_list = sorted(glob.glob(os.path.join(data_dir, "videos", vid, "images", "*.jpg")))
    nb_images = len(image_list)
    print("{} {}".format(vid, nb_images))
