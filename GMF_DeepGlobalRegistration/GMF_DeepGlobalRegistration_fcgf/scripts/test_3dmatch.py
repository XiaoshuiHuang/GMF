# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
# Run with python -m scripts.test_3dmatch_refactor
import os
import sys
from sklearn.metrics import recall_score, precision_score, f1_score
import torch.nn as nn
sys.path.append(os.path.abspath("../"))
import math
import logging
import open3d as o3d
import numpy as np
import time
import torch
import copy

sys.path.append('.')
import MinkowskiEngine as ME
from config import get_config
from model import load_model

from dataloader.data_loaders import ThreeDMatchTrajectoryDataset
from core.knn import find_knn_gpu
from core.deep_global_registration import DeepGlobalRegistration

from util.timer import Timer
from util.pointcloud import make_open3d_point_cloud

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
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

# Criteria
def rte_rre(T_pred, T_gt, rte_thresh, rre_thresh, eps=1e-16):
  if T_pred is None:
    return np.array([0, np.inf, np.inf])

  rte = np.linalg.norm(T_pred[:3, 3] - T_gt[:3, 3])
  rre = np.arccos(
      np.clip((np.trace(T_pred[:3, :3].T @ T_gt[:3, :3]) - 1) / 2, -1 + eps,
              1 - eps)) * 180 / math.pi
  return np.array([rte < rte_thresh and rre < rre_thresh, rte, rre])


def analyze_stats(stats, mask, method_names):
  mask = (mask > 0).squeeze(1)
  stats = stats[:, mask, :]

  print('Total result mean')
  for i, method_name in enumerate(method_names):
    print(method_name)
    print(stats[i].mean(0))

  print('Total successful result mean')
  for i, method_name in enumerate(method_names):
    sel = stats[i][:, 0] > 0
    sel_stats = stats[i][sel]
    print(method_name)
    print(sel_stats.mean(0))


def create_pcd(xyz, color):
  # n x 3
  n = xyz.shape[0]
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(xyz)
  pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (n, 1)))
  pcd.estimate_normals(
      search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
  return pcd


def draw_geometries_flip(pcds):
  pcds_transform = []
  flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
  for pcd in pcds:
    pcd_temp = copy.deepcopy(pcd)
    pcd_temp.transform(flip_transform)
    pcds_transform.append(pcd_temp)
  o3d.visualization.draw_geometries(pcds_transform)


def evaluate(methods, method_names, data_loader, config, debug=False):

  tot_num_data = len(data_loader.dataset)
  data_loader_iter = iter(data_loader)

  # Accumulate success, rre, rte, time, sid
  mask = np.zeros((tot_num_data, 1)).astype(int)
  stats = np.zeros((len(methods), tot_num_data, 5))
  stats_class = np.zeros([tot_num_data, 3])
  failed_nums = 0
  class_loss = ClassificationLoss()
  dataset = data_loader.dataset
  subset_names = open(dataset.DATA_FILES[dataset.phase]).read().split()

  for batch_idx in range(tot_num_data):
    batch = data_loader_iter.next()

    # Skip too sparse point clouds
    sname, xyz0, xyz1, trans,p_image,q_image = batch[0]

    sid = subset_names.index(sname)
    T_gt = np.linalg.inv(trans)

    for i, method in enumerate(methods):
      start = time.time()
      T,xyz0_corr,xyz1_corr = method.register(
        xyz0,
        xyz1,
        p_image=p_image,
        q_image=q_image,
        use_corr=True
      )
      end = time.time()

      # Visualize
      if debug:
        print(method_names[i])
        pcd0 = create_pcd(xyz0, np.array([1, 0.706, 0]))
        pcd1 = create_pcd(xyz1, np.array([0, 0.651, 0.929]))

        pcd0.transform(T)
        draw_geometries_flip([pcd0, pcd1])
        pcd0.transform(np.linalg.inv(T))

      stats[i, batch_idx, :3] = rte_rre(T, T_gt, config.success_rte_thresh,
                                        config.success_rre_thresh)
      stats[i, batch_idx, 3] = end - start
      stats[i, batch_idx, 4] = sid

      frag1 = xyz0_corr.cpu().numpy()
      frag2 = xyz1_corr.cpu().numpy()

      # T_gt_tensor = torch.from_numpy(T_gt).float()
      # T_tensor = torch.from_numpy(T).float()

      frag1_warp = transform(frag1, T_gt)
      distance = np.sqrt(np.sum(np.power(frag1_warp - frag2, 2), axis=1))
      gt_labels = (distance < 0.10).astype(np.int)

      frag1_warp = transform(frag1, T)
      distance = np.sqrt(np.sum(np.power(frag1_warp - frag2, 2), axis=1))
      pred_labels = (distance < 0.10).astype(np.int)


      pred_labels = torch.unsqueeze(torch.from_numpy(pred_labels).cuda().float(),0)
      gt_labels = torch.unsqueeze(torch.from_numpy(gt_labels).cuda().float(),0)
      class_stats = class_loss(pred_labels, gt_labels)

      stats_class[i, 0] = float(class_stats['precision'])  # output inlier precision
      stats_class[i, 1] = float(class_stats['recall'])  # output inlier recall
      stats_class[i, 2] = float(class_stats['f1'])

      mask[batch_idx] = 1

      if stats[i, batch_idx, 0] == 0:
        print(f"{method_names[i]}: failed")
        failed_nums += 1

    if batch_idx % 10 == 9:
      print('Summary {} / {}'.format(batch_idx, tot_num_data))
      analyze_stats(stats, mask, method_names)

  # Save results
  filename = f'3dmatch-stats_{method.__class__.__name__}'
  if os.path.isdir(config.out_dir):
    out_file = os.path.join(config.out_dir, filename)
  else:
    out_file = filename  # save it on the current directory
  print(f'Saving the stats to {out_file}')
  np.savez(out_file, stats=stats, names=method_names)
  analyze_stats(stats, mask, method_names)
  success_num = tot_num_data - failed_nums
  pair_success = success_num/tot_num_data
  # Analysis per scene
  for i, method in enumerate(methods):
    print(f'Scene-wise mean {method}')
    scene_vals = np.zeros((len(subset_names), 3))
    for sid, sname in enumerate(subset_names):
      curr_scene = stats[i, :, 4] == sid
      scene_vals[sid] = (stats[i, curr_scene, :3]).mean(0)

    print('All scenes')
    print(scene_vals)
    print('Scene average')
    all_scenes = scene_vals.mean(0)
    all_scenes[0] = pair_success
    print(all_scenes)
    print(f"precision:{stats_class[:,0].mean(0)},recall:{stats_class[:,1].mean(0)},f1:{stats_class[:,2].mean(0)}")

if __name__ == '__main__':


  checkpoint_path = "../weights/3DMatch_fcgf.pth"
  test_path = "/DISK/qwt/datasets/3dmatch/3DMatch_test"
  # test_path = "/DISK/qwt/datasets/3dmatch/3DLoMatch_test"
  config = get_config()

  config.weights = checkpoint_path
  config.threed_match_dir = test_path

  print(config)

  dgr = DeepGlobalRegistration(config)

  methods = [dgr]
  method_names = ['DGR']

  dset = ThreeDMatchTrajectoryDataset(phase='test',
                                      transform=None,
                                      random_scale=False,
                                      random_rotation=False,
                                      config=config)

  data_loader = torch.utils.data.DataLoader(dset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=1,
                                            collate_fn=lambda x: x,
                                            pin_memory=False,
                                            drop_last=True)

  evaluate(methods, method_names, data_loader, config, debug=False)
