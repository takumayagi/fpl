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
import sys
import datetime
import json
import subprocess


def add_str(x, y, end=False):
    if end:
        return x + y
    else:
        return x + y + " "


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: python run.py <json_file> run/test <gpu_id>")
        exit(1)

    input_name = sys.argv[1]
    decision = sys.argv[2]
    gpu_id = int(sys.argv[3]) if len(sys.argv) == 4 else 0
    with open(input_name, "r") as f:
        data = json.load(f)

    if not os.path.exists("gen_scripts"):
        os.makedirs("gen_scripts")

    date = datetime.datetime.now()
    experiment_id = os.path.splitext(os.path.basename(input_name))[0] + date.strftime("_%y%m%d_%H%M%S")
    base_str = "python -u "
    base_str = add_str(base_str, data["script_name"])
    base_str = add_str(base_str, "--root_dir {}".format(os.path.join("experiments", experiment_id)))

    for key, value in data["fixed_args"].items():
        base_str = add_str(base_str, "--{}".format(key))
        if type(value) == list:
            base_str = add_str(base_str, " ".join(map(str, value)))
        else:
            base_str = add_str(base_str, str(value))

    if "combination_args" in data:
        commands = []
        for value_dict in data["combination_args"]:
            comb_str = base_str
            for key, value in value_dict.items():
                comb_str = add_str(comb_str, "--{}".format(key))
                if type(value) == list:
                    comb_str = add_str(comb_str, " ".join(map(str, value)))
                else:
                    comb_str = add_str(comb_str, str(value))
            commands.append(comb_str)
    else:
        commands = [base_str]

    if "dynamic_args" in data:
        for key, value_list in data["dynamic_args"].items():
            commands = [add_str(cmd, "--{} {}".format(key, " ".join(map(str, v)) if type(v) == list else v)) for v in value_list for cmd in commands]

    if decision == "runtest" and "test_args" in data:
        for key, value in data["test_args"].items():
            commands = [add_str(cmd, "--{} {}".format(key, " ".join(map(str, value)) if type(value) == list else value)) for cmd in commands]

    if "cv" in data and not decision == "runtest":
        nb_splits = data["cv"]
        commands = [add_str(cmd, "--nb_splits {} --eval_split {}".format(nb_splits, sp)) for cmd in commands for sp in range(nb_splits)]

    commands = [add_str(cmd, "--gpu {}".format(gpu_id)) for cmd in commands]

    for cmd in commands:
        print(cmd)

    if decision == "run" or decision == "runtest":
        script_path = os.path.join("gen_scripts", "{}.sh".format(experiment_id))
        print("Scripts written to {}".format(script_path))
        with open(script_path, "w") as f:
            f.write("#! /bin/sh\n")
            f.write("cd {}\n".format(os.getcwd()))
            for idx, cmd in enumerate(commands):
                f.write(cmd+"\n")
        cmd = "chmod +x {}".format(script_path)
        print(cmd)
        subprocess.call(cmd.split(" "))
        cmd = "sh {}".format(script_path)
        print(cmd)
        subprocess.call(cmd.split(" "))
    else:
        print(len(commands))
        print("Test finished")
