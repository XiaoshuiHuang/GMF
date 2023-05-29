# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
import os

import sys
sys.path.append(os.path.abspath("../"))
import logging
import argparse
import numpy as np
import open3d as o3d
from sklearn.metrics import recall_score, precision_score, f1_score
import torch
import torch.nn as nn

from config import get_config

from core.deep_global_registration import DeepGlobalRegistration

from dataloader.kitti_loader import KITTINMPairDataset
from dataloader.base_loader import CollationFunctionFactory
from util.pointcloud import make_open3d_point_cloud, make_open3d_feature, pointcloud_to_spheres
from util.timer import AverageMeter, Timer

from scripts.test_3dmatch import rte_rre

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    handlers=[ch])

def transform(pts, trans):
  """
  Applies the SE3 transformations, support torch.Tensor and np.ndarry.  Equation: trans_pts = R @ pts + t
  Input
      - pts: [num_pts, 3] or [bs, num_pts, 3], pts to be transformed
      - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
  Output
      - pts: [num_pts, 3] or [bs, num_pts, 3] transformed pts
  """

  trans_pts = trans[:3, :3] @ pts.T + trans[:3, 3:4]
  return trans_pts.T

class ClassificationLoss(nn.Module):
  def __init__(self, balanced=True):
    super(ClassificationLoss, self).__init__()
    self.balanced = balanced

  def forward(self, pred, gt, weight=None):
    """
    Classification Loss for the inlier confidence
    Inputs:
        - pred: [bs, num_corr] predicted logits/labels for the putative correspondences
        - gt:   [bs, num_corr] ground truth labels
    Outputs:(dict)
        - loss          (weighted) BCE loss for inlier confidence
        - precision:    inlier precision (# kept inliers / # kepts matches)
        - recall:       inlier recall (# kept inliers / # all inliers)
        - f1:           (precision * recall * 2) / (precision + recall)
        - logits_true:  average logits for inliers
        - logits_false: average logits for outliers
    """
    num_pos = torch.relu(torch.sum(gt) - 1) + 1
    num_neg = torch.relu(torch.sum(1 - gt) - 1) + 1
    if weight is not None:
      loss = nn.BCEWithLogitsLoss(reduction='none')(pred, gt.float())
      loss = torch.mean(loss * weight)
    elif self.balanced is False:
      loss = nn.BCEWithLogitsLoss(reduction='mean')(pred, gt.float())
    else:
      loss = nn.BCEWithLogitsLoss(pos_weight=num_neg * 1.0 / num_pos, reduction='mean')(pred, gt.float())

    # compute precision, recall, f1
    pred_labels = pred > 0
    gt, pred_labels, pred = gt.detach().cpu().numpy(), pred_labels.detach().cpu().numpy(), pred.detach().cpu().numpy()
    precision = precision_score(gt[0], pred_labels[0])
    recall = recall_score(gt[0], pred_labels[0])
    f1 = f1_score(gt[0], pred_labels[0])
    mean_logit_true = np.sum(pred * gt) / max(1, np.sum(gt))
    mean_logit_false = np.sum(pred * (1 - gt)) / max(1, np.sum(1 - gt))

    eval_stats = {
      "loss": loss,
      "precision": float(precision),
      "recall": float(recall),
      "f1": float(f1),
      "logit_true": float(mean_logit_true),
      "logit_false": float(mean_logit_false)
    }
    return eval_stats

TE_THRESH = 0.6  # m
RE_THRESH = 5  # deg
VISUALIZE = False


def visualize_pair(xyz0, xyz1, T, voxel_size):
    pcd0 = pointcloud_to_spheres(xyz0,
                                 voxel_size,
                                 np.array([0, 0, 1]),
                                 sphere_size=0.6)
    pcd1 = pointcloud_to_spheres(xyz1,
                                 voxel_size,
                                 np.array([0, 1, 0]),
                                 sphere_size=0.6)
    pcd0.transform(T)
    o3d.visualization.draw_geometries([pcd0, pcd1])


def analyze_stats(stats):
    print('Total result mean')
    print(stats.mean(0))

    sel_stats = stats[stats[:, 0] > 0]
    print(sel_stats.mean(0))


def evaluate(config, data_loader, method,f=None):
    data_timer = Timer()

    test_iter = data_loader.__iter__()
    N = len(test_iter)

    success_rte_sum = 0
    success_rre_sum = 0
    success_precision = 0
    success_recall = 0
    success_f1 = 0
    success_num = 0

    class_loss = ClassificationLoss()

    stats = np.zeros((N, 5))  # bool succ, rte, rre, time, drive id

    for i in range(len(data_loader)):
        data_timer.tic()
        try:
            data_dict = test_iter.next()
        except ValueError as exc:
            pass
        data_timer.toc()

        drive = data_dict['extra_packages'][0]['drive']
        xyz0, xyz1,p_image,q_image = \
            data_dict['pcd0'][0], \
            data_dict['pcd1'][0],\
            data_dict['p_image'], \
            data_dict['q_image'], \

        T_gt = data_dict['T_gt'][0].numpy()
        xyz0np, xyz1np = xyz0.numpy(), xyz1.numpy()

        # T = method.register(xyz0np, xyz1np)
        T, xyz0_corr, xyz1_corr = method.register(
            xyz0np,
            xyz1np,
            p_image=p_image,
            q_image=q_image,
            use_corr=True
        )

        stats[i, :3] = rte_rre(T, T_gt, TE_THRESH, RE_THRESH)
        stats[i, 3] = method.reg_timer.diff + method.feat_timer.diff
        stats[i, 4] = drive

        if stats[i, 0] == 0:
            logging.info(f"Failed with RTE: {stats[i, 1]}, RRE: {stats[i, 2]}")
        else:
            success_num+=1
            extra_package = data_dict['extra_packages'][0]
            content = f"extra_package:{extra_package}\npred_T:\n{T}\nGT_T:\n{T_gt}\n"
            content.replace(",", "").replace("tensor","").replace("[","").replace("]","")
            f.write(content)
            f.write("*************************************************************************\n\n")

        frag1 = xyz0_corr.cpu().numpy()
        frag2 = xyz1_corr.cpu().numpy()

        frag1_warp = transform(frag1, T_gt)
        distance = np.sqrt(np.sum(np.power(frag1_warp - frag2, 2), axis=1))
        gt_labels = (distance < 0.6).astype(np.int)

        frag1_warp = transform(frag1, T)
        distance = np.sqrt(np.sum(np.power(frag1_warp - frag2, 2), axis=1))
        pred_labels = (distance < 0.6).astype(np.int)

        pred_labels = torch.unsqueeze(torch.from_numpy(pred_labels).cuda().float(), 0)
        gt_labels = torch.unsqueeze(torch.from_numpy(gt_labels).cuda().float(), 0)
        class_stats = class_loss(pred_labels, gt_labels)

        success_rre_sum += stats[i, 2]
        success_rte_sum += stats[i, 1]
        success_precision += float(class_stats['precision'])
        success_recall += float(class_stats['recall'])
        success_f1 += float(class_stats['f1'])

        if i % 10 == 0:
            succ_rate, rte, rre, avg_time, _ = stats[:i + 1].mean(0)
            logging.info(
                f"{i} / {N}: Data time: {data_timer.avg}, Feat time: {method.feat_timer.avg},"
                + f" Reg time: {method.reg_timer.avg}, RTE: {rte}," +
                f" RRE: {rre}, Success: {succ_rate * 100} %")

        if VISUALIZE and i % 10 == 9:
            visualize_pair(xyz0, xyz1, T, config.voxel_size)

    succ_rate, rte, rre, avg_time, _ = stats.mean(0)
    print(stats.mean(0))
    logging.info(
        f"Data time: {data_timer.avg}, Feat time: {method.feat_timer.avg}," +
        f" Reg time: {method.reg_timer.avg}, RTE: {rte}," +
        f" RRE: {rre}, Success: {succ_rate * 100} %")

    # Save results
    # filename = f'kitti-stats_{method.__class__.__name__}'
    # if config.out_filename is not None:
    #     filename += f'_{config.out_filename}'
    # if isinstance(method, FCGFWrapper):
    #     filename += '_' + method.method
    #     if 'ransac' in method.method:
    #         filename += f'_{config.ransac_iter}'
    # if os.path.isdir(config.out_dir):
    #     out_file = os.path.join(config.out_dir, filename)
    # else:
    #     out_file = filename  # save it on the current directory
    # print(f'Saving the stats to {out_file}')
    # np.savez(out_file, stats=stats)
    # analyze_stats(stats)

    mean_success = success_num / N
    mean_rte = success_rte_sum/N
    mean_rre = success_rre_sum/N * 100
    success_precision = success_precision/N
    success_recall = success_recall/N
    success_f1 = success_f1/N

    print(f"RR:{mean_success},rre:{mean_rre},rte:{mean_rte}")
    print(f"precision:{success_precision},recall:{success_recall},f1:{success_f1}")


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test_path = "/DISK/qwt/datasets/kitti/data_odometry_velodyne"

    checkpoint_path = "../weights/Kitti_fcgf.pth"

    config = get_config()

    config.weights = checkpoint_path
    config.threed_match_dir = test_path
    print(f"dataset : {test_path}")

    dgr = DeepGlobalRegistration(config)

    dset = KITTINMPairDataset('test',
                              transform=None,
                              random_rotation=False,
                              random_scale=False,
                              config=config)

    data_loader = torch.utils.data.DataLoader(
        dset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=CollationFunctionFactory(concat_correspondences=False,
                                            collation_type='collate_pair'),
        pin_memory=False,
        drop_last=False)
    filename = "/DISK/qwt/PointDSC_Ours/select/KItti/DGR/log.txt"
    with open(file=filename, mode="w") as f:
        evaluate(config, data_loader, dgr,f=f)
    print(f"dataset : {test_path}")