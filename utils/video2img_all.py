#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import glob
from video2img import video2img


for video_path in sorted(glob.glob("data/videos/*.mp4")):
    video2img(video_path)
