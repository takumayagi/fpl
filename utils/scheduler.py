#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

from logging import getLogger
logger = getLogger("main")


class DummyScheduler(object):
    def __init__(self):
        pass

    def update(self):
        pass

class MomentumSGDScheduler(object):

    def __init__(self, optimizer, decay_rate, decay_steps):
        self._itr = 0
        self._optimizer = optimizer
        self._momentum = optimizer.momentum
        self._optimizer.momentum = 0.7  # Stable start
        self._decay_rate = decay_rate
        self._decay_steps = decay_steps
        assert isinstance(decay_steps, list)
        assert 0 < decay_rate < 1.0

    def update(self):
        self._itr += 1
        if self._itr == 500:
            self._optimizer.momentum = self._momentum
        if self._itr in self._decay_steps:
            self._optimizer.lr *= self._decay_rate
            logger.info("Step {}: lr = {}".format(self._itr, self._optimizer.lr))


class AdamScheduler(object):

    def __init__(self, optimizer, decay_rate, decay_steps, start=0):
        self._itr = start
        self._optimizer = optimizer
        self._decay_rate = decay_rate
        self._decay_steps = decay_steps
        assert isinstance(decay_steps, list)
        assert 0 < decay_rate < 1.0

    def update(self):
        self._itr += 1
        if self._itr in self._decay_steps:
            self._optimizer.alpha *= self._decay_rate
            logger.info("Step {}: lr = {}".format(self._itr, self._optimizer.lr))
