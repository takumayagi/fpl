#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import os

import json
import time
import joblib

import numpy as np

import chainer
from chainer import Variable, iterators, cuda
from chainer.dataset import convert

from utils.generic import get_args, get_model, write_prediction
from utils.dataset import SceneDatasetCV
from utils.summary_logger import SummaryLogger
from utils.evaluation import Evaluator

from mllogger import MLLogger
logger = MLLogger(init=False)


if __name__ == "__main__":
    """
    Evaluation with Cross-Validation
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

    data_dir = os.getenv("TRAJ_DATA_DIR")
    data = joblib.load(args.in_data)
    traj_len = data["trajectories"].shape[1]

    # Load evaluation data
    valid_split = args.eval_split + args.nb_splits
    valid_dataset = SceneDatasetCV(data, args.input_len, args.offset_len, args.pred_len,
                                   arg.width, args.height, data_dir, valid_split, -1,
                                   False, "scale" in args.model, args.ego_type)
    logger.info(valid_dataset.X.shape)

    # X: input, Y: output, poses, egomotions
    data_idxs = [0, 1, 2, 7]
    if data_idxs is None:
        logger.info("Invalid argument: model={}".format(args.model))
        exit(1)

    model = get_model(args)

    prediction_dict = {
        "arguments": vars(args),
        "predictions": {}
    }
    valid_iterator = iterators.MultiprocessIterator(
        valid_dataset, args.batch_size, False, False, n_processes=args.nb_jobs)
    valid_eval = Evaluator("valid", args)

    logger.info("Evaluation...")
    chainer.config.train = False
    chainer.config.enable_backprop = False

    # Evaluation loop
    for itr, batch in enumerate(valid_iterator):
        batch_array = [convert.concat_examples([x[idx] for x in batch], args.gpu) for idx in data_idxs]
        loss, pred_y, prob = model.predict(tuple(map(Variable, batch_array)))
        valid_eval.update(cuda.to_cpu(loss.data), pred_y, batch)
        write_prediction(prediction_dict["predictions"], batch, pred_y)

    message_str = "Evaluation: valid loss {} / ADE {} / FDE {}"
    logger.info(message_str.format(valid_eval("loss"), valid_eval("ade"), valid_eval("fde")))
    valid_eval.update_summary(summary, -1, ["loss", "ade", "fde"])
    predictions = prediction_dict["predictions"]
    pred_list = [[pred for vk, v_dict in sorted(predictions.items())
                  for fk, f_dict in sorted(v_dict.items())
                  for pk, pred in sorted(f_dict.items()) if pred[8] == idx] for idx in range(4)]
    logger.info([len(x) for x in pred_list])

    error_rates = [np.mean([pred[7] for pred in preds]) for preds in pred_list]
    logger.info("Towards {} / Away {} / Across {} / Other {}".format(*error_rates))

    prediction_path = os.path.join(save_dir, "prediction.json")
    with open(prediction_path, "w") as f:
        json.dump(prediction_dict, f)

    summary.update("finished", 1)
    summary.write()
    logger.info("Elapsed time: {} (s), Saved at {}".format(time.time()-start, save_dir))
