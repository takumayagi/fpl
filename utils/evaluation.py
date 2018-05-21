#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import numpy as np


def calc_mse(pred_y, true_y):
    return np.linalg.norm(pred_y - true_y, axis=pred_y.ndim-1)


def calc_weighted_mse(pred_y, true_y, scales):
    weights = 1. / scales
    if pred_y.ndim == 2:
        wade = np.linalg.norm(pred_y - true_y, axis=1) * weights
    else:
        wade = np.linalg.norm(pred_y - true_y, axis=2) * weights[:, np.newaxis]
    return wade


class Evaluator(object):
    def __init__(self, prefix, args):
        self.prefix = prefix
        self.nb_grids = args.nb_grids
        self.width = args.width
        self.height = args.height
        self.reset()

    def reset(self):
        self.loss = 0
        self.cnt = 0
        self.ade = 0
        self.fde = 0

    def update(self, loss, pred_y, batch):
        batch_size = len(pred_y)
        true_y = np.array([z[1] for z in batch])  # (B, T, 2)

        self.loss += loss * batch_size
        mse = calc_mse(pred_y[..., :2], true_y[..., :2])
        self.ade += np.mean(mse) * batch_size

        mse = calc_mse(pred_y[:, -1, :2], true_y[:, -1, :2])
        self.fde += np.mean(mse) * batch_size

        self.cnt += batch_size

    def __call__(self, name, normalize=True):
        if normalize:
            return getattr(self, name) / self.cnt if self.cnt != 0 else 0.0
        else:
            return getattr(self, name)

    def update_summary(self, summary, iter_cnt, targets):
        for name in targets:
            summary.update_by_cond(self.prefix + "_" + name, getattr(self, name) / self.cnt, iter_cnt + 1)
        summary.write()
