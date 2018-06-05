#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Takuma Yagi <yagi@ks.cs.titech.ac.jp>
#
# Distributed under terms of the MIT license.

import pandas as pd
import subprocess
from subprocess import CalledProcessError


def get_commit_id():
    cmd = "git log -n 1 --format=%H"
    try:
        out = subprocess.check_output(cmd.split(" ")).decode('utf-8')
    except CalledProcessError:
        return ""
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

    def update_by_cond(self, key, value, timing, cond="lower"):
        assert cond == "lower" or cond == "higher"
        sign = 1 if cond == "lower" else -1
        if key not in self._summ or sign * value < sign * self._summ[key]:
            self._summ[key] = value
            self._summ[key+"_tm"] = timing

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
