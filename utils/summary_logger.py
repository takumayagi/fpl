#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Takuma Yagi <yagi@ks.cs.titech.ac.jp>
#
# Distributed under terms of the MIT license.

import time
import datetime
import pandas as pd
import subprocess
import argparse


def get_commit_id():
    cmd = "git log -n 1 --format=%H"
    out = subprocess.check_output(cmd.split(" ")).decode('utf-8')
    return out.rstrip("\r\n")

class SummaryLogger(object):

    def __init__(self, args, logger, output_path):
        self._summ = vars(args)
        self._date_str = logger.dir_name
        self._output_path = output_path
        self._summ["log_path"] = logger.log_fn
        self._summ["commit_id"] = get_commit_id()

    def update(self, key, value):
        self._summ[key] = value

    def update_dict(self, update_dict):
        for key, value in update_dict.items():
            self._summ[key] = value

    def update_by_cond(self, key, value, time, cond="lower"):
        assert cond == "lower" or cond == "higher"
        sign = 1 if cond == "lower" else -1
        if key not in self._summ or sign * value < sign * self._summ[key]:
            self._summ[key] = value
            self._summ[key+"_tm"] = time

    def write(self):
        """Write summary to csv file
        If date_str already exists, overwrite it"""
        new_summary = pd.DataFrame([self._summ.values()], index=[self._date_str], columns=self._summ.keys())
        try:
            current = pd.read_csv(self._output_path, index_col=0)
        except IOError:
            new_summary.to_csv(self._output_path)
            return
        if self._date_str in current.index:
            # JOIN by the joined keys
            current = current.loc[:, current.columns.union(new_summary.columns)]
            current.update(new_summary)
        else:
            current = current.append(new_summary)
        current.to_csv(self._output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--cnn', type=str, default="CNN_M_2048")
    parser.add_argument('--optimizer', type=str, default="sgd")
    parser.add_argument('--batchsize', '-b', type=int, default=256)
    args = parser.parse_args()
    print(vars(args))

    summary_dict = initialize_summary(args, "log.txt")
    date = datetime.datetime.now()
    date_str = date.strftime('%y%m%d_%H%M%S')
    write_summary("summary.csv", summary_dict, date_str)
    summary_dict["added"] = 1
    write_summary("summary.csv", summary_dict, date_str)
    time.sleep(2)

    loss = 1.0
    acc = 0.5
    update_loss_acc(summary_dict, loss, acc, 1)
    loss = 0.5
    acc = 0.4
    update_loss_acc(summary_dict, loss, acc, 2)
    loss = [1.0, 0.5]
    acc = [0.4, 0.6]
    update_loss_acc(summary_dict, loss, acc, 2)

    print(summary_dict)
    assert summary_dict["loss"] == 0.5
    assert summary_dict["acc"] == 0.5
    assert summary_dict["loss_ep"] == 2
    assert summary_dict["acc_ep"] == 1
    assert summary_dict["loss_1_ep"] == 2
    assert summary_dict["loss_1"] == 1.0
    assert summary_dict["acc_2"] == 0.6

    date = datetime.datetime.now()
    date_str = date.strftime('%y%m%d_%H%M%S')
    write_summary("summary.csv", summary_dict, date_str)
