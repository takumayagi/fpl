#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import os

import json
import time
import joblib

import numpy as np

import chainer
from chainer import Variable, optimizers, serializers, iterators, cuda
from chainer.dataset import convert

from utils.generic import get_args, get_model, write_prediction
from utils.dataset import SceneDatasetCV
from utils.summary_logger import SummaryLogger
from utils.scheduler import AdamScheduler
from utils.evaluation import Evaluator

from mllogger import MLLogger
logger = MLLogger(init=False)


if __name__ == "__main__":
    """
    Training with Cross-Validation
    """
    args = get_args()

    np.random.seed(args.seed)
    start = time.time()
    logger.initialize(args.root_dir)
    logger.info(vars(args))
    save_dir = logger.get_savedir()
    logger.info("Written to {}".format(save_dir))
    summary = SummaryLogger(args, logger, os.path.join(args.root_dir, "summary.csv"))
    summary.update("finished", 0)

    data_dir = "data"
    data = joblib.load(args.in_data)
    traj_len = data["trajectories"].shape[1]

    # Load training data
    train_splits = list(filter(lambda x: x != args.eval_split, range(args.nb_splits)))
    valid_split = args.eval_split + args.nb_splits
    train_dataset = SceneDatasetCV(data, args.input_len, args.offset_len, args.pred_len,
                                   args.width, args.height, data_dir, train_splits, args.nb_train,
                                   True, "scale" in args.model, args.ego_type)
    logger.info(train_dataset.X.shape)
    valid_dataset = SceneDatasetCV(data, args.input_len, args.offset_len, args.pred_len,
                                   args.width, args.height, data_dir, valid_split, -1,
                                   False, "scale" in args.model, args.ego_type)
    logger.info(valid_dataset.X.shape)

    # X: input, Y: output, poses, egomotions
    data_idxs = [0, 1, 2, 7]
    if data_idxs is None:
        logger.info("Invalid argument: model={}".format(args.model))
        exit(1)

    model = get_model(args)
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))
    scheduler = AdamScheduler(optimizer, 0.5, args.lr_step_list, 0)

    train_iterator = iterators.MultithreadIterator(train_dataset, args.batch_size, n_threads=args.nb_jobs)
    train_eval = Evaluator("train", args)
    valid_iterator = iterators.MultithreadIterator(valid_dataset, args.batch_size, False, False, n_threads=args.nb_jobs)
    valid_eval = Evaluator("valid", args)

    logger.info("Training...")
    train_eval.reset()
    st = time.time()

    # Training loop
    for iter_cnt, batch in enumerate(train_iterator):
        if iter_cnt == args.nb_iters:
            break
        chainer.config.train = True
        chainer.config.enable_backprop = True
        batch_array = [convert.concat_examples([x[idx] for x in batch], args.gpu) for idx in data_idxs]
        model.cleargrads()
        loss, pred_y, _ = model(tuple(map(Variable, batch_array)))
        loss.backward()
        loss.unchain_backward()
        optimizer.update()
        scheduler.update()
        train_eval.update(cuda.to_cpu(loss.data), pred_y, batch)

        # Validation & report
        if (iter_cnt + 1) % args.iter_snapshot == 0:
            logger.info("Validation...")
            if args.save_model:
                serializers.save_npz(os.path.join(save_dir, "model_{}.npz".format(iter_cnt + 1)), model)
            chainer.config.train = False
            chainer.config.enable_backprop = False
            model.cleargrads()
            prediction_dict = {
                "arguments": vars(args),
                "predictions": {}
            }

            valid_iterator.reset()
            valid_eval.reset()
            for itr, batch in enumerate(valid_iterator):
                batch_array = [convert.concat_examples([x[idx] for x in batch], args.gpu) for idx in data_idxs]
                loss, pred_y, _ = model.predict(tuple(map(Variable, batch_array)))
                valid_eval.update(cuda.to_cpu(loss.data), pred_y, batch)
                write_prediction(prediction_dict["predictions"], batch, pred_y)

            message_str = "Iter {}: train loss {} / ADE {} / FDE {}, valid loss {} / " \
                          "ADE {} / FDE {}, elapsed time: {} (s)"
            logger.info(message_str.format(
                iter_cnt + 1, train_eval("loss"), train_eval("ade"), train_eval("fde"),
                valid_eval("loss"), valid_eval("ade"), valid_eval("fde"), time.time()-st))
            train_eval.update_summary(summary, iter_cnt, ["loss", "ade", "fde"])
            valid_eval.update_summary(summary, iter_cnt, ["loss", "ade", "fde"])

            predictions = prediction_dict["predictions"]
            pred_list = [[pred for vk, v_dict in sorted(predictions.items())
                          for fk, f_dict in sorted(v_dict.items())
                          for pk, pred in sorted(f_dict.items()) if pred[8] == idx] for idx in range(4)]

            error_rates = [np.mean([pred[7] for pred in preds]) for preds in pred_list]
            logger.info("Towards {} / Away {} / Across {} / Other {}".format(*error_rates))

            prediction_path = os.path.join(save_dir, "prediction.json")
            with open(prediction_path, "w") as f:
                json.dump(prediction_dict, f)

            st = time.time()
            train_eval.reset()
        elif (iter_cnt + 1) % args.iter_display == 0:
            msg = "Iter {}: train loss {} / ADE {} / FDE {}"
            logger.info(msg.format(iter_cnt + 1, train_eval("loss"), train_eval("ade"), train_eval("fde")))

    summary.update("finished", 1)
    summary.write()
    logger.info("Elapsed time: {} (s), Saved at {}".format(time.time()-start, save_dir))
