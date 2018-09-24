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
import glob
import sys
import datetime
import subprocess


def add_str(x, y, end=False):
    if end:
        return x + y
    else:
        return x + y + " "


if __name__ == "__main__":

    if len(sys.argv) < 5:
        print("Usage: python utils/eval.py <experiment_dir> <iter> run/test <gpu_id> <pred_len>")
        exit(1)

    dirname = sys.argv[1][:-1] if sys.argv[1].endswith("/") else sys.argv[1]
    eval_iter = sys.argv[2]
    decision = sys.argv[3]
    gpu_id = int(sys.argv[4])
    #dataset_path = sys.argv[5] if len(sys.argv) == 6 else None
    dataset_path = None
    pred_len = sys.argv[5] if len(sys.argv) >= 6 else 10
    dataset = sys.argv[6] if len(sys.argv) >= 7 else None
    print(pred_len)

    if not os.path.exists("evaluations"):
        os.makedirs("evaluations")

    experiment_id = os.path.basename(dirname)
    genpath = os.path.join("gen_scripts", "{}.sh".format(experiment_id))
    commands = []
    with open(genpath, "r") as f:
        for idx, line in enumerate(f):
            if idx <= 1:
                continue
            commands.append(line.replace('train_cv.py', 'eval_cv.py'))

    date = datetime.datetime.now()
    out_id = experiment_id + date.strftime("_%y%m%d_%H%M%S")

    result_list = list(sorted(glob.glob(os.path.join(dirname, "*/"))))
    while len(result_list) < len(commands):
        result_list.append("")

    assert len(commands) == len(result_list)

    out_commands = []
    for cmd, rdir in zip(commands, result_list):
        if rdir == "":
            continue
        model_path = os.path.join(rdir, "model_{}.npz".format(eval_iter))
        out_command = add_str(
            cmd.strip("\n"), "--root_dir {} --resume {}".format(os.path.join("evaluations", out_id), model_path))
        if dataset_path is None:
            out_commands.append(out_command)
        else:
            out_commands.append(add_str(out_command, "--in_data {}".format(dataset_path)))

    out_commands = [add_str(cmd, "--gpu {} --debug --pred_len {}".format(gpu_id, pred_len)) for cmd in out_commands]

    if dataset is not None:
        out_commands = [add_str(cmd, "--in_data {}".format(dataset)) for cmd in out_commands]

    for cmd in out_commands:
        print(cmd)

    if decision == "run":
        script_path = os.path.join("gen_scripts", "eval_{}.sh".format(out_id))
        print("Scripts written to {}".format(script_path))
        with open(script_path, "w") as f:
            f.write("#! /bin/sh\n")
            f.write("cd {}\n".format(os.getcwd()))
            for cmd in out_commands:
                f.write(cmd+"\n")
        cmd = "chmod +x {}".format(script_path)
        print(cmd)
        subprocess.call(cmd.split(" "))
        cmd = "sh {}".format(script_path)
        print(cmd)
        subprocess.call(cmd.split(" "))
    else:
        print(len(out_commands))
        print("Test finished")
