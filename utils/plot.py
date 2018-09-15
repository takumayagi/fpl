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

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]]
COLORS = [[85.0, 0.0, 255.0], [0.0, 0.0, 255.0], [0.0, 85.0, 255.0], [0.0, 170.0, 255.0], [0.0, 255.0, 255.0], [0.0, 255.0, 170.0], [0.0, 255.0, 85.0], [0.0, 255.0, 0.0], [85.0, 255.0, 0.0], [170.0, 255.0, 0.0], [255.0, 255.0, 0.0], [255.0, 170.0, 0.0], [255.0, 85.0, 0.0], [255.0, 0.0, 0.0], [170.0, 0.0, 255.0], [255.0, 0.0, 170.0], [255.0, 0.0, 255.0], [255.0, 0.0, 85.0]]


def draw_line(img, traj, color, ratio, thickness=2, skip=1):
    for idx, (d1, d2) in enumerate(zip(traj[:-1], traj[1:])):
        if idx % skip != 0:
            continue
        cv2.line(img, (int(d1[0]*ratio), int(d1[1]*ratio)), (int(d2[0]*ratio), int(d2[1]*ratio)), color, thickness)
    return img

def draw_dotted_line(img, traj, color, ratio, thickness=2, r=0.85):
    for idx, (d1, d2) in enumerate(zip(traj[:-1], traj[1:])):
        if r < 0.5:
            w = (1 - r) / 4
            x1 = int(d1[0] * (1 - w) + d2[0] * w)
            y1 = int(d1[1] * (1 - w) + d2[1] * w)
            x2 = int(d1[0] * (1 - (w + r)) + d2[0] * (w + r))
            y2 = int(d1[1] * (1 - (w + r)) + d2[1] * (w + r))
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            x1 = int(d1[0] * (1 - (w * 3 + r)) + d2[0] * (w * 3 + r))
            y1 = int(d1[1] * (1 - (w * 3 + r)) + d2[1] * (w * 3 + r))
            x2 = int(d1[0] * w + d2[0] * (1 - w))
            y2 = int(d1[1] * w + d2[1] * (1 - w))
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
        else:
            x1 = int(d1[0] * r + d2[0] * (1 - r))
            y1 = int(d1[1] * r + d2[1] * (1 - r))
            x2 = int(d2[0] * r + d1[0] * (1 - r))
            y2 = int(d2[1] * r + d1[1] * (1 - r))
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img


def draw_x(img, point, color, ratio, thickness=2):
    x, y = int(point[0]*ratio), int(point[1]*ratio)
    scale = thickness * 2
    cv2.line(img, (x - scale, y - scale), (x + scale, y + scale), color, thickness)
    cv2.line(img, (x + scale, y - scale), (x - scale, y + scale), color, thickness)
    return img


def draw_rect(img, point, color, ratio, rsize=12):
    x, y = int(point[0]*ratio), int(point[1]*ratio)
    cv2.rectangle(img, (x - rsize // 2, y - rsize // 2), (x + rsize // 2, y + rsize // 2), color, -1)
    return img


def draw_scale(img, cx, cy, rad, color):
    cv2.circle(img, (cx, cy), rad, color, 1)
    cv2.line(img, (cx, cy - rad), (cx, cy + rad), color, 1)
    cv2.line(img, (cx - rad, cy), (cx + rad, cy), color, 1)
    return img


def draw_pose(img, point, pose, scale, ratio):
    canvas = np.zeros_like(img)
    cx, cy = point
    pose = pose.reshape((18, 2))
    for idx, (x, y) in enumerate(pose):
        cv2.circle(canvas, (int((x*scale+cx)*ratio), int((y*scale+cy)*ratio)), int(scale / 7 * ratio), COLORS[idx], -1)
    for idx1, idx2 in PAIRS:
        line_color = list(map(lambda x: (x[0] + x[1]) / 2, zip(COLORS[idx1], COLORS[idx2])))
        x1, y1 = pose[idx1]
        x2, y2 = pose[idx2]
        cv2.line(canvas, (int((x1*scale+cx)*ratio), int((y1*scale+cy)*ratio)),
                 (int((x2*scale+cx)*ratio), int((y2*scale+cy)*ratio)), line_color, int(scale / 12 * ratio))

    return cv2.addWeighted(img, 1.0, canvas, 0.5, 0.0)


class epoch_recorder(object):

    def __init__(self, outname, xname, yname, label_names):
        self.outname = outname
        self.xname = xname
        self.yname = yname
        self.label_names = label_names

        self.history = [[] for x in label_names]
        self.timing = [[] for x in label_names]

    def update(self, stats, timing=None, update_idxs=None):

        update_idxs = list(range(len(stats))) if update_idxs is None else update_idxs
        update_idxs = [update_idxs] if not isinstance(update_idxs, list) else update_idxs
        stats = [stats] if not isinstance(stats, list) else stats

        for stat, uidx in zip(stats, update_idxs):
            self.history[uidx].append(stat)
            if timing is not None:
                self.timing[uidx].append(timing)

        plt.figure()
        for idx, (hist, timings, label) in enumerate(zip(self.history, self.timing, self.label_names)):
            plt.plot(list(range(1, len(hist)+1)) if len(timings) == 0 else timings, hist, linewidth=3, label=label)
        plt.grid()
        plt.legend()
        plt.xlabel(self.xname)
        plt.ylabel(self.yname)
        plt.savefig(self.outname, format="png")
        plt.close()

def plot_trajectory():  # DEPRECATED
    pass

def plot_trajectory_eval(save_dir, data_dir, w, h, past, ground_truth, vid, frame, pid, predicted, egomotion, pose, pred_ego, scale, pred_pose, flipped=False):
    past_len = len(past)
    pred_len = len(ground_truth)

    impath = os.path.join(data_dir, "videos", vid, "images_384",
                          "rgb_{:05d}.jpg".format(frame + len(past) - 1))
    img = cv2.imread(impath)
    impath2 = os.path.join(data_dir, "videos", vid, "images_384",
                          "rgb_{:05d}.jpg".format(frame + past_len + pred_len - 1))
    img2 = cv2.imread(impath2)
    if img is None or img2 is None:
        print("Invalid image: {} {}".format(impath, impath2))
        return
    if predicted.shape[-1] == 3:
        scale = predicted[-1, 2] if predicted.ndim == 2 else predicted[2]

    past = past[...,:2]
    ground_truth = ground_truth[...,:2]
    predicted = predicted[...,:2]

    img = img[:, ::-1, :].copy() if flipped else img
    img2 = img2[:, ::-1, :].copy() if flipped else img2

    # Write frame number
    key = os.path.basename(impath)
    if predicted.ndim == 2:
        ade = np.mean(np.linalg.norm(predicted - ground_truth, axis=1))
        fde = np.linalg.norm(predicted[-1, :] - ground_truth[-1, :])
    else:
        ade = np.linalg.norm(predicted - ground_truth)
        fde = 0.0
    message_str = "{} {} ADE={:.2f} FDE={:.2f}".format(int(key[4:9]), pid, ade, fde)
    cv2.putText(img, message_str, (5, 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 0))

    out_h, out_w, _ = img.shape
    ratio = out_w / w
    bar_width = 3

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Past
    cv2.circle(img, (int(past[0][0]*ratio), int(past[0][1]*ratio)), 4, (255, 0, 0), -1)
    img = draw_line(img, past, (255, 0, 0), ratio)
    img = draw_x(img, past[-1], (255, 0, 0), ratio)
    img2 = draw_line(img2, past, (255, 0, 0), ratio)
    img2 = draw_x(img2, past[-1], (255, 0, 0), ratio)

    # Ground truth
    cv2.circle(img, (int(ground_truth[0][0]*ratio), int(ground_truth[0][1]*ratio)), 4, (0, 0, 255), -1)
    img = draw_line(img, ground_truth, (0, 0, 255), ratio)
    img = draw_x(img, ground_truth[-1], (0, 0, 255), ratio)
    img2 = draw_line(img2, ground_truth, (0, 0, 255), ratio)
    img2 = draw_x(img2, ground_truth[-1], (0, 0, 255), ratio)

    # Prediction
    if predicted.ndim == 1:
        img = draw_x(img, predicted, (0, 255, 0), ratio)
        img2 = draw_x(img2, predicted, (0, 255, 0), ratio)
    else:
        img = draw_line(img, predicted, (0, 255, 0), ratio)
        img = draw_x(img, predicted[-1], (0, 255, 0), ratio)
        img2 = draw_line(img2, predicted, (0, 255, 0), ratio)
        img2 = draw_x(img2, predicted[-1], (0, 255, 0), ratio)

    if egomotion is not None:
        egomotion = np.array(egomotion)
        # 円プロットにしきい値付きで表示
        # 逆変換が必要だが、情報がない->datasetに引数追加
        px, pz = 50, 70
        gx, gz = 50, 160
        rad = 40

        # Zが負=前進=上=座標負
        # Xが正=右=右=座標正:
        # Past
        ego_plot = egomotion[:, [3, 5]].copy() / 2
        ego_plot[ego_plot < -1] = -1
        ego_plot[ego_plot > 1] = 1
        ego_plot[:, 0] = ego_plot[:, 0] * 4
        ego_plot = ego_plot * rad + np.array([px, pz])
        img = draw_scale(img, px, pz, rad, (255, 255, 255))
        img = draw_line(img, ego_plot[:past_len], (255, 0, 0), 1.0)
        img = draw_x(img, ego_plot[past_len-1], (255, 0, 0), 1.0)

        #  GT
        ego_plot = egomotion[:, [3, 5]].copy() / 2
        ego_plot[ego_plot < -1] = -1
        ego_plot[ego_plot > 1] = 1
        ego_plot[:, 0] = ego_plot[:, 0] * 4
        ego_plot = ego_plot * rad + np.array([gx, gz])
        img = draw_scale(img, gx, gz, rad, (255, 255, 255))
        img = draw_line(img, ego_plot[past_len:past_len+pred_len], (0, 0, 255), 1.0)
        img = draw_x(img, ego_plot[-1], (0, 0, 255), 1.0)

    #  Pred
    if pred_ego is not None:
        if pred_ego.ndim == 2:
            ego_plot = pred_ego[:, [3, 5]].copy() / 2
            ego_plot[ego_plot < -1] = -1
            ego_plot[ego_plot > 1] = 1
            ego_plot[:, 0] = ego_plot[:, 0] * 4
            ego_plot = ego_plot * rad + np.array([gx, gz])
            img = draw_line(img, ego_plot, (0, 255, 0), 1.0)
            img = draw_x(img, ego_plot[-1], (0, 255, 0), 1.0)
            # print(np.linalg.norm(egomotion[-1, :] - pred_ego[-1, :]))
        else:
            ego_plot = pred_ego[[3, 5]].copy() / 2
            ego_plot[ego_plot < -1] = -1
            ego_plot[ego_plot > 1] = 1
            ego_plot[0] = ego_plot[0] * 4
            ego_plot = ego_plot * rad + np.array([gx, gz])
            img = draw_x(img, ego_plot, (0, 255, 0), 1.0)
            # print(np.linalg.norm(egomotion[-1, :] - pred_ego))

    # 予測位置に最後の姿勢を表示
    if pred_pose is not None:
        if pred_pose.ndim == 2:
            img = draw_pose(img, predicted[-1], pred_pose[-1], scale, ratio)
            img2 = draw_pose(img2, predicted[-1], pred_pose[-1], scale, ratio)
        else:
            img = draw_pose(img, predicted, pred_pose, scale, ratio)
            img2 = draw_pose(img2, predicted, pred_pose, scale, ratio)

    out_h = out_h // 4 * 3
    out_w = out_w // 4 * 3
    canvas = np.zeros((out_h, out_w * 2, 3), dtype=np.uint8)
    canvas[:, :out_w, :] = cv2.resize(img, (out_w, out_h))
    canvas[:, out_w:, :] = cv2.resize(img2, (out_w, out_h))

    # ポーズを戻すにはscale
    cv2.imwrite(os.path.join(save_dir, "pred_{}_{}_{}.jpg".format(vid, frame, pid)), canvas)


def plot_trajectory_full(save_dir, data_dir, w, h, past, ground_truth, vid, frame, pid, predicted, egomotion, pose, pred_ego, scale, pred_pose, flipped=False):
    past_len = len(past)
    pred_len = len(ground_truth)

    img_dir = os.path.join(data_dir, "videos", vid, "images")
    img = cv2.imread(os.path.join(img_dir, "rgb_{:05d}.jpg".format(frame)))
    img2 = cv2.imread(os.path.join(img_dir, "rgb_{:05d}.jpg".format(frame + 14)))
    img3 = cv2.imread(os.path.join(img_dir, "rgb_{:05d}.jpg".format(frame + 29)))
    if img is None or img2 is None:
        print("Invalid image: {} {}".format(impath, impath2))
        return

    past = past[...,:2]
    ground_truth = ground_truth[...,:2]
    predicted = predicted[...,:2]

    img = img[:, ::-1, :].copy() if flipped else img
    img2 = img2[:, ::-1, :].copy() if flipped else img2
    img3 = img3[:, ::-1, :].copy() if flipped else img3

    out_h, out_w, _ = img.shape
    ratio = out_w / w
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Past
    cv2.circle(img, (int(past[0][0]*ratio), int(past[0][1]*ratio)), 20, (255, 64, 64), -1)
    img = draw_line(img, past, (255, 64, 64), ratio, 10)
    img = draw_x(img, past[-1], (255, 64, 64), ratio, 12)
    img2 = draw_line(img2, past, (255, 64, 64), ratio, 10)
    img2 = draw_x(img2, past[-1], (255, 64, 64), ratio, 12)
    img3 = draw_line(img3, past, (255, 64, 64), ratio, 10)
    img3 = draw_x(img3, past[-1], (255, 64, 64), ratio, 12)

    # Ground truth
    #img = draw_dotted_line(img, ground_truth, (0, 0, 255), ratio, 10)
    #img = draw_x(img, ground_truth[-1], (0, 0, 255), ratio, 12)
    #img2 = draw_dotted_line(img2, ground_truth, (0, 0, 255), ratio, 10)
    #img2 = draw_x(img2, ground_truth[-1], (0, 0, 255), ratio, 12)
    cv2.circle(img3, (int(ground_truth[0][0]*ratio), int(ground_truth[0][1]*ratio)), 16, (0, 0, 255), -1)
    img3 = draw_dotted_line(img3, ground_truth, (0, 0, 255), ratio, 10)
    img3 = draw_x(img3, ground_truth[-1], (0, 0, 255), ratio, 12)

    # Prediction
    if predicted.ndim == 1:
        #img = draw_x(img, predicted, (0, 255, 0), ratio, 12)
        #img2 = draw_x(img2, predicted, (0, 255, 0), ratio, 12)
        img3 = draw_x(img3, predicted, (0, 255, 0), ratio, 12)
    else:
        #img = draw_dotted_line(img, predicted, (0, 255, 0), ratio, 10)
        #img = draw_x(img, predicted[-1], (0, 255, 0), ratio, 12)
        #img2 = draw_dotted_line(img2, predicted, (0, 255, 0), ratio, 10)
        #img2 = draw_x(img2, predicted[-1], (0, 255, 0), ratio, 12)
        img3 = draw_dotted_line(img3, predicted, (0, 255, 0), ratio, 10)
        img3 = draw_x(img3, predicted[-1], (0, 255, 0), ratio, 12)

    cv2.imwrite(os.path.join(save_dir, "pred_{}_{}_{}_1.jpg".format(vid, frame, pid)), img)
    cv2.imwrite(os.path.join(save_dir, "pred_{}_{}_{}_2.jpg".format(vid, frame, pid)), img2)
    cv2.imwrite(os.path.join(save_dir, "pred_{}_{}_{}_3.jpg".format(vid, frame, pid)), img3)



def calc_2d_normal_prob(x1, x2, mu1, mu2, s1, s2, rho):
    norm1 = x1 - mu1
    norm2 = x2 - mu2
    s1s2 = s1 * s2
    z = (norm1 / s1) ** 2 + (norm2 / s2) ** 2 - 2 * rho * norm1 * norm2 / s1s2
    neg_rho = 1 - rho ** 2
    return np.exp(-z / (2 * neg_rho)) / (2 * np.pi * s1s2 * np.sqrt(neg_rho))


def plot_prob_all(save_dir, w, h, mean, std, input_data, probs, predicted):
    if len(input_data) == 4:
        past, ground_truth, vid, frame = input_data
    else:
        past, ground_truth, vid, frame, _, _ = input_data

    cmap = plt.get_cmap("jet")
    X = np.array([[((x - mean[0]) / std[0], (y - mean[1]) / std[1])
                 for x in range(0, w, 4)] for y in range(0, h, 4)])

    canvas = np.zeros((X.shape[0], X.shape[1]))
    for prob in probs:
        o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr = prob
        Z = [calc_2d_normal_prob(X[:,:,0], X[:,:,1], mu1, mu2, s1, s2, rho)
            for mu1, mu2, s1, s2, rho in zip(o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr)]
        canvas = np.maximum(canvas, np.sum([pi * z for pi, z in zip(o_pi, Z)], axis=0))

    past_x = [int(x[0]) for x in past]
    past_y = [h - int(x[1]) for x in past]

    plt.figure()
    plt.imshow(canvas, cmap=cmap, vmin=0, vmax=np.max(canvas), extent=[0, w, 0, h], aspect='auto', interpolation="bilinear")
    plt.plot(past_x, past_y, "w")
    plt.savefig(os.path.join(save_dir, "pred_{}_{}_prob.jpg".format(vid, frame)))
    plt.close()


def plot_probability(save_dir, data_dir, w, h, pred_t, mean, std, input_data, prob, predicted, offset):
    impath = os.path.join(data_dir, "videos", vid, "images_384",
                          "rgb_{:05d}.jpg".format(frame+offset))
    img = cv2.imread(impath)
    img = img[:, ::-1, :] if flipped else img

    past, ground_truth, _, vid, frame = input_data[:5]
    o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr = prob
    cmap = plt.get_cmap("jet")

    X = np.array([[((x - mean[0]) / std[0], (y - mean[1]) / std[1])
                 for x in range(0, w, 4)] for y in range(0, h, 4)])
    Z = [calc_2d_normal_prob(X[:,:,0], X[:,:,1], mu1, mu2, s1, s2, rho)
            for mu1, mu2, s1, s2, rho in zip(o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr)]
    Z = np.sum([pi * z for pi, z in zip(o_pi, Z)], axis=0)

    out_h, out_w, _ = img.shape
    img_ratio = out_h / h

    gt_x, gt_y = (ground_truth[pred_t]*img_ratio).astype(np.int)
    pred_x, pred_y = (predicted * img_ratio).astype(np.int)
    past = (past * img_ratio).astype(np.int)
    #past_x = [int(x[0]*img_ratio) for x in past]
    #past_y = [int((h - x[1])*img_ratio) for x in past]

    Z = Z / np.max(Z) * 255
    Z = Z[:, :, np.newaxis].astype(np.uint8)
    Z = cv2.resize(Z, (out_w, out_h))
    cmap = cv2.applyColorMap(Z, cv2.COLORMAP_JET)
    out = cv2.addWeighted(img, 0.5, cmap, 0.5, 0)

    for d1, d2 in zip(past[:-1], past[1:]):
        cv2.line(out, (int(d1[0]), int(d1[1])), (int(d2[0]), int(d2[1])), (255, 255, 255), 2)

    cv2.circle(out, (gt_x, gt_y), 4, (0, 0, 255), -1)
    cv2.circle(out, (pred_x, pred_y), 4, (0, 255, 0), -1)

    cv2.imwrite(os.path.join(save_dir, "pred_{}_{}_{}.jpg".format(vid, frame, pred_t)), img)


def draw_ade(save_dir, data):
    np.save(os.path.join(save_dir, "hist_ade.npy"), data)
    plt.figure(figsize=(12, 6))
    plt.hist(data, bins=np.arange(0, 400, 10))
    plt.title("ADE histogram: mean={}, std={}".format(np.mean(data), np.std(data)))
    plt.xlabel("Average Displacement Error (px)")
    plt.ylabel("Frequency")
    plt.grid(which='major',color='black',linestyle='-')
    plt.grid(which='minor',color='black',linestyle='-')
    plt.savefig(os.path.join(save_dir, "hist_ade.png"))
    plt.close()


def draw_wade(save_dir, data):
    np.save(os.path.join(save_dir, "hist_wade.npy"), data)
    plt.figure(figsize=(12, 6))
    plt.hist(data, bins=np.arange(0, 4.0, 0.1))
    plt.title("WADE histogram: mean={}, std={}".format(np.mean(data), np.std(data)))
    plt.xlabel("Weighted Average Displacement Error")
    plt.ylabel("Frequency")
    plt.grid(which='major',color='black',linestyle='-')
    plt.grid(which='minor',color='black',linestyle='-')
    plt.savefig(os.path.join(save_dir, "hist_wade.png"))
    plt.close()


def draw_szade(save_dir, data):
    np.save(os.path.join(save_dir, "hist_szade.npy"), data)
    plt.figure(figsize=(12, 6))
    plt.hist(data, bins=np.arange(0, 100.0, 4))
    plt.title("SZADE histogram: mean={}, std={}".format(np.mean(data), np.std(data)))
    plt.xlabel("Average Size Displacement Error")
    plt.ylabel("Frequency")
    plt.grid(which='major',color='black',linestyle='-')
    plt.grid(which='minor',color='black',linestyle='-')
    plt.savefig(os.path.join(save_dir, "hist_szade.png"))
    plt.close()


def draw_swade(save_dir, data, sidx):
    if len(data) == 0:
        return
    np.save(os.path.join(save_dir, "hist_swade_{}.npy".format(sidx)), data)
    plt.figure(figsize=(12, 6))
    plt.hist(data, bins=np.arange(0, 4.0, 0.1))
    plt.title("SWADE histogram: mean={}, std={}".format(np.mean(data), np.std(data)))
    plt.xlabel("Scale Weighted Average Displacement Error: scale={}".format(sidx))
    plt.ylabel("Frequency")
    plt.grid(which='major',color='black',linestyle='-')
    plt.grid(which='minor',color='black',linestyle='-')
    plt.savefig(os.path.join(save_dir, "hist_sade_{}.png".format(sidx)))
    plt.close()

#def plot_trajectory(save_dir, data_dir, w, h, past, ground_truth, vid, frame, flipped, predicted, egomotion, pose):
"""
def plot_egomotion(img, egomotion):

    trans_center = [200, 200]
    rot_center = [200, 400]

    pass
"""

if __name__ == "__main__":
    pass
    """
    data_dir =
    video_id = "GOPR0250U20"
    impath = os
    img = cv2.imr
    """
