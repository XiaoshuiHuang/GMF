import os
from os.path import join, exists
import pickle
import glob
import random
import numpy as np
import matplotlib.image as image
import torch.utils.data as data
from utils.uio import process_image
from utils.pointcloud import make_point_cloud, estimate_normal
from utils.SE3 import rotation_matrix,\
                      translation_matrix,\
                      transform,decompose_trans,\
                      integrate_trans,concatenate

class ThreeDMatchTrainVal(data.Dataset):
    def __init__(
            self,
            root,
            split,
            descriptor='fcgf',
            in_dim=6,
            inlier_threshold=0.10,
            num_node=1000,
            use_mutual=True,
            downsample=0.03,
            augment_axis=3,
            augment_rotation=1.0,
            augment_translation=0.5,
            config=None
    ):
        self.root = root
        self.split = split
        self.descriptor = descriptor
        assert descriptor in ['fpfh', 'fcgf']
        self.in_dim = in_dim
        self.inlier_threshold = inlier_threshold
        self.num_node = num_node
        self.use_mutual = use_mutual
        self.downsample = downsample
        self.augment_axis = augment_axis
        self.augment_rotation = augment_rotation
        self.augment_translation = augment_translation
        self.config = config

        # 使用重叠率超过30%的数据集进行训练
        OVERLAP_RATIO = 0.3 
        DATA_FILES = {
            'train': './misc/split/train_3dmatch.txt',
            'val': './misc/split/val_3dmatch.txt',
            # 'test': './mic/test_3dmatch.txt'
        }
        # subset_names = open(DATA_FILES[split]).read().split()
        # self.files = []
        # self.length = 0

        # 使用FCGF处理的数据集
        # for name in subset_names:
        #     fname = name + "*%.2f.txt" % OVERLAP_RATIO
        #     fnames_txt = glob.glob(root + "/" + fname)
        #     assert len(fnames_txt) > 0, f"Make sure that the path {root} has data {fname}"
        #     for fname_txt in fnames_txt:
        #         with open(fname_txt) as f:
        #             content = f.readlines()
        #         fnames = [x.strip().split() for x in content]
        #         for fname in fnames:
        #             self.files.append([fname[0], fname[1]])
        #             self.length += 1

        subset_names = open(DATA_FILES[split]).read().split()
        self.files = []
        self.length = 0
        for name in subset_names:
            fname = name
            fnames_txt = glob.glob(pathname=os.path.join(self.config.overlap_path, fname + "*"))
            assert len(fnames_txt) > 0, f"Make sure that the path {self.config.overlap_path} has data {fname}"
            for fname_txt in fnames_txt:
                with open(fname_txt) as f:
                    content = f.readlines()
                fnames = [x.strip().split() for x in content]
                for fname in fnames:
                    self.files.append([fname[0], fname[1]])
                    self.length += 1
        return
        
    def __getitem__(self, index):
        # load meta data
        src_id, tgt_id = self.files[index][0], self.files[index][1]
        if random.random() > 0.5:
            src_id, tgt_id = tgt_id, src_id
        
        # 加载点坐标以及预计算好的描述符
        if self.descriptor == 'fcgf':

            file0 = os.path.join(self.root, src_id.replace(".ply","_fcgf.npz"))
            file1 = os.path.join(self.root, tgt_id.replace(".ply","_fcgf.npz"))
            src_data = np.load(file0)
            tgt_data = np.load(file1)
            src_keypts = src_data['xyz']
            tgt_keypts = tgt_data['xyz']
            src_features = src_data['feature']
            tgt_features = tgt_data['feature']

        elif self.descriptor == 'fpfh':

            file0 = os.path.join(self.root, src_id.replace(".ply","_fpfh.npz"))
            file1 = os.path.join(self.root, tgt_id.replace(".ply","_fpfh.npz"))
            src_data = np.load(file0)
            tgt_data = np.load(file1)
            src_keypts = src_data['xyz']
            tgt_keypts = tgt_data['xyz']
            src_features = src_data['feature']
            tgt_features = tgt_data['feature']
            np.nan_to_num(src_features)
            np.nan_to_num(tgt_features)
            src_features = src_features / (np.linalg.norm(src_features, axis=1, keepdims=True) + 1e-6)
            tgt_features = tgt_features / (np.linalg.norm(tgt_features, axis=1, keepdims=True) + 1e-6)

        # image
        image_file0 = os.path.join(self.root, src_id.replace(".ply", "_0.png"))
        image_file1 = os.path.join(self.root, tgt_id.replace(".ply", "_0.png"))
        if (not os.path.exists(image_file0)):
            image_file0 = os.path.join(self.root, src_id.replace(".ply", "_0.jpg"))
        if (not os.path.exists(image_file1)):
            image_file1 = os.path.join(self.root, tgt_id.replace(".ply", "_0.jpg"))
        p_image = image.imread(image_file0)
        if (p_image.shape[0] != self.config.image_H or p_image.shape[1] != self.config.image_W):
            p_image = process_image(image=p_image, aim_H=self.config.image_H, aim_W=self.config.image_W)
        p_image = np.transpose(p_image, axes=(2, 0, 1))
        q_image = image.imread(image_file1)
        if (q_image.shape[0] != self.config.image_H or q_image.shape[1] != self.config.image_W):
            q_image = process_image(image=q_image, aim_H=self.config.image_H, aim_W=self.config.image_W)
        q_image = np.transpose(q_image, axes=(2, 0, 1))

        # 计算 Ground Truth 的变换
        orig_trans = np.eye(4).astype(np.float32)                
        # 数据增强 (添加数据增强用于原坐标变换)
        src_keypts += np.random.rand(src_keypts.shape[0], 3) * 0.005
        tgt_keypts += np.random.rand(tgt_keypts.shape[0], 3) * 0.005
        aug_R = rotation_matrix(self.augment_axis, self.augment_rotation)
        aug_T = translation_matrix(self.augment_translation)
        aug_trans = integrate_trans(aug_R, aug_T)
        tgt_keypts = transform(tgt_keypts, aug_trans)
        gt_trans = concatenate(aug_trans, orig_trans)

        # 随机选出指定数量的关键点（1000），源点云和目标点云都选出关键点
        N_src = src_features.shape[0]
        N_tgt = tgt_features.shape[0]
        if self.num_node == 'all':
            src_sel_ind = np.arange(N_src)
            tgt_sel_ind = np.arange(N_tgt)
        else:
            # 随机选出1000个点的索引
            src_sel_ind = np.random.choice(N_src, self.num_node)
            tgt_sel_ind = np.random.choice(N_tgt, self.num_node)
        # 根据指定索引选出特征
        src_desc = src_features[src_sel_ind, :]
        tgt_desc = tgt_features[tgt_sel_ind, :]
        # 根据指定索引选出关键点
        src_keypts = src_keypts[src_sel_ind, :]
        tgt_keypts = tgt_keypts[tgt_sel_ind, :]            
        
        # 在特征空间中搜索对应的点集合(find min distance corresdences in 1000 point pairs)
        distance = np.sqrt(2 - 2 * (src_desc @ tgt_desc.T) + 1e-6)
        source_idx = np.argmin(distance, axis=1)
        source_dis = np.min(distance, axis=1)
        # Flase
        if self.use_mutual:
            target_idx = np.argmin(distance, axis=0)
            mutual_nearest = (target_idx[source_idx] == np.arange(source_idx.shape[0]))
            corr = np.concatenate([np.where(mutual_nearest == 1)[0][:,None], source_idx[mutual_nearest][:,None]], axis=-1)
        else:
            # 从随机选出的源点云（1000）以及目标点云（1000）中，根据特征找出两者的对应点（可能会过少）
            corr = np.concatenate([np.arange(source_idx.shape[0])[:, None], source_idx[:, None]], axis=-1)
        if len(corr) < 10:
            # skip pairs with too few correspondences.
            return  self.__getitem__(int(np.random.choice(self.__len__(),1)))
        
        # 计算Ground Truth的标签（这些对应点中，是否属于匹配正确的标签）
        frag1 = src_keypts[corr[:, 0]]
        frag2 = tgt_keypts[corr[:, 1]]
        frag1_warp = transform(frag1, gt_trans)
        distance = np.sqrt(np.sum(np.power(frag1_warp - frag2, 2), axis=1))
        # The labels is the answer wheather each correspondence is inilier in 1000 point pairs.
        labels = (distance < self.inlier_threshold).astype(np.int)

        # prepare input to the network
        if self.split == 'train' and np.mean(labels) > 0.5:
            # add random outlier to input data (deprecated)
            num_outliers = int(0 * len(corr))
            src_outliers = np.random.randn(num_outliers, 3) * np.mean(src_keypts, axis=0)
            tgt_outliers = np.random.randn(num_outliers, 3) * np.mean(tgt_keypts, axis=0)
            input_src_keypts = np.concatenate( [src_keypts[corr[:, 0]], src_outliers], axis=0)
            input_tgt_keypts = np.concatenate( [tgt_keypts[corr[:, 1]], tgt_outliers], axis=0)
            labels = np.concatenate([labels, np.zeros(num_outliers)], axis=0)
        else:
            input_src_keypts = src_keypts[corr[:, 0]]
            input_tgt_keypts = tgt_keypts[corr[:, 1]]

        # input_src_keypts -- input_src_desc
        input_src_desc = src_desc[corr[:, 0]]
        # input_tgt_keypts -- input_tgt_desc
        input_tgt_desc = tgt_desc[corr[:, 1]]

        if self.in_dim == 3:
            corr_pos = input_src_keypts - input_tgt_keypts
        elif self.in_dim == 6:
            corr_pos = np.concatenate([input_src_keypts, input_tgt_keypts], axis=-1)
            # move the center of each point cloud to (0,0,0).
            corr_pos = corr_pos - corr_pos.mean(0)
        elif self.in_dim == 9:
            corr_pos = np.concatenate([input_src_keypts, input_tgt_keypts, input_src_keypts-input_tgt_keypts], axis=-1)
        elif self.in_dim == 70:
            corr_pos = np.concatenate([input_src_keypts, input_tgt_keypts], axis=-1)
            # move the center of each point cloud to (0,0,0).
            corr_pos = corr_pos - corr_pos.mean(0)
            corr_pos = np.concatenate([corr_pos, src_desc[corr[:,0]], tgt_desc[corr[:,1]]], axis=-1)
        elif self.in_dim == 12:   
            src_pcd = make_point_cloud(src_keypts)
            tgt_pcd = make_point_cloud(tgt_keypts)
            estimate_normal(src_pcd, radius=self.downsample*2)
            estimate_normal(tgt_pcd, radius=self.downsample*2)
            src_normal = np.array(src_pcd.normals)
            tgt_normal = np.array(tgt_pcd.normals)  
            src_normal = src_normal[src_sel_ind, :]
            tgt_normal = tgt_normal[tgt_sel_ind, :]   
            input_src_normal = src_normal[corr[:, 0]]
            input_tgt_normal = tgt_normal[corr[:, 1]]
            corr_pos = np.concatenate([input_src_keypts, input_src_normal, input_tgt_keypts, input_tgt_normal], axis=-1)

        '''
            corr_pos.astype(np.float32)             :             根据特征选出的对应点(1000 point pairs),[1000,6]
            input_src_keypts.astype(np.float32)     :             对应点中的源点集合,[1000,3],corr[:,:3]
            input_tgt_keypts.astype(np.float32)     :             对应点中的目标点集合,[1000,3],corr[:,3:]
            gt_trans.astype(np.float32)             :             Ground Truth变换矩阵,[4,4]
            labels.astype(np.float32)               :             对应点的Ground Truth的标签（是否为内点）[1000]
            p_image.astype(np.float32)              :             src image, [1,3,120,160]
            q_image.astype(np.float32)              :             tgt image, [1,3,120,160]
            input_src_desc.astype(np.float32)       :             src desc, [1000,32]
            input_tgt_desc.astype(np.float32)       :             tgt desc, [1000,32]
        '''
        li= corr_pos.astype(np.float32), \
            input_src_keypts.astype(np.float32), \
            input_tgt_keypts.astype(np.float32), \
            gt_trans.astype(np.float32), \
            labels.astype(np.float32),\
            p_image.astype(np.float32),\
            q_image.astype(np.float32),\
            input_src_desc.astype(np.float32),\
            input_tgt_desc.astype(np.float32)

        return li
            
    def __len__(self):
        return self.length


class ThreeDMatchTest(data.Dataset):
    def __init__(self, 
                root, 
                descriptor='fcgf',
                in_dim=6,
                inlier_threshold=0.10,
                num_node=5000, 
                use_mutual=True,
                downsample=0.03, 
                augment_axis=0, 
                augment_rotation=1.0,
                augment_translation=0.01,
                select_scene=None,
                config=None
                ):
        self.root = root
        self.descriptor = descriptor
        assert descriptor in ['fcgf', 'fpfh']
        self.in_dim = in_dim
        self.inlier_threshold = inlier_threshold
        self.num_node = num_node
        self.use_mutual = use_mutual
        self.downsample = downsample
        self.augment_axis = augment_axis
        self.augment_rotation = augment_rotation
        self.augment_translation = augment_translation
        self.config = config
        # assert augment_axis == 0
        # assert augment_rotation == 0
        # assert augment_translation == 0
        
        # containers
        self.gt_trans = {}
        
        self.scene_list = [
            '7-scenes-redkitchen',
            'sun3d-home_at-home_at_scan1_2013_jan_1',
            'sun3d-home_md-home_md_scan9_2012_sep_30',
            'sun3d-hotel_uc-scan3',
            'sun3d-hotel_umd-maryland_hotel1',
            'sun3d-hotel_umd-maryland_hotel3',
            'sun3d-mit_76_studyroom-76-1studyroom2',
            'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
        ]
        if select_scene in self.scene_list:
            self.scene_list = [select_scene]
        
        # load ground truth transformation
        for scene in self.scene_list:
            scene_path = f'{self.root}/{scene}'
            # gt_path = f'{self.root}/gt_result/{scene}-evaluation'
            for k, v in self.__loadlog__(scene_path).items():
                self.gt_trans[f'{scene}@{k}'] = v
        return
                  
    def __getitem__(self, index):
        # load meta data
        key = list(self.gt_trans.keys())[index]      
        scene = key.split('@')[0]
        src_id = key.split('@')[1].split('_')[0]
        tgt_id = key.split('@')[1].split('_')[1]
        
        # load point coordinates and pre-computed per-point local descriptors
        if self.descriptor == 'fcgf':

            src_path = f"{self.root}/{scene}/seq-01/cloud_bin_{src_id}_fcgf.npz"
            tgt_path = f"{self.root}/{scene}/seq-01/cloud_bin_{tgt_id}_fcgf.npz"
            src_data = np.load(src_path)
            tgt_data = np.load(tgt_path)
            src_keypts = src_data['xyz']
            tgt_keypts = tgt_data['xyz']
            src_features = src_data['feature']
            tgt_features = tgt_data['feature']

        elif self.descriptor == 'fpfh':

            src_data = np.load(f"{self.root}/{scene}/seq-01/cloud_bin_{src_id}_fpfh.npz")
            tgt_data = np.load(f"{self.root}/{scene}/seq-01/cloud_bin_{tgt_id}_fpfh.npz")
            src_keypts = src_data['xyz']
            tgt_keypts = tgt_data['xyz']
            src_features = src_data['feature']
            tgt_features = tgt_data['feature']
            src_features = src_features / (np.linalg.norm(src_features, axis=1, keepdims=True) + 1e-6)
            tgt_features = tgt_features / (np.linalg.norm(tgt_features, axis=1, keepdims=True) + 1e-6)

        # image
        image_file0 = f"{self.root}/{scene}/seq-01/cloud_bin_{src_id}_0.png"
        image_file1 = f"{self.root}/{scene}/seq-01/cloud_bin_{tgt_id}_0.png"
        if (not os.path.exists(image_file0)):
            image_file0 = image_file0.replace(".png",".jpg")
        if (not os.path.exists(image_file1)):
            image_file1 = image_file1.replace(".png",".jpg")
        p_image = image.imread(image_file0)
        if (p_image.shape[0] != self.config.image_H or p_image.shape[1] != self.config.image_W):
            p_image = process_image(image=p_image, aim_H=self.config.image_H, aim_W=self.config.image_W)
        p_image = np.transpose(p_image, axes=(2, 0, 1))
        q_image = image.imread(image_file1)
        if (q_image.shape[0] != self.config.image_H or q_image.shape[1] != self.config.image_W):
            q_image = process_image(image=q_image, aim_H=self.config.image_H, aim_W=self.config.image_W)
        q_image = np.transpose(q_image, axes=(2, 0, 1))

        # compute ground truth transformation
        orig_trans = np.linalg.inv(self.gt_trans[key])  # the given ground truth trans is target-> source   
        # data augmentation
        aug_R = rotation_matrix(self.augment_axis, self.augment_rotation)
        aug_T = translation_matrix(self.augment_translation)
        aug_trans = integrate_trans(aug_R, aug_T)
        tgt_keypts = transform(tgt_keypts, aug_trans)
        gt_trans = concatenate(aug_trans, orig_trans)

        # select {self.num_node} numbers of keypoints
        N_src = src_features.shape[0]
        N_tgt = tgt_features.shape[0]
        # use all point during test.
        if self.num_node == 'all':
            src_sel_ind = np.arange(N_src)
            tgt_sel_ind = np.arange(N_tgt)
        else:
            src_sel_ind = np.random.choice(N_src, self.num_node)
            tgt_sel_ind = np.random.choice(N_tgt, self.num_node)
        src_desc = src_features[src_sel_ind, :]
        tgt_desc = tgt_features[tgt_sel_ind, :]
        src_keypts = src_keypts[src_sel_ind, :]
        tgt_keypts = tgt_keypts[tgt_sel_ind, :]
        
        # construct the correspondence set by mutual nn in feature space.
        distance = np.sqrt(2 - 2 * (src_desc @ tgt_desc.T) + 1e-6)
        source_idx = np.argmin(distance, axis=1)
        if self.use_mutual:
            target_idx = np.argmin(distance, axis=0)
            mutual_nearest = (target_idx[source_idx] == np.arange(source_idx.shape[0]))
            corr = np.concatenate([np.where(mutual_nearest == 1)[0][:,None], source_idx[mutual_nearest][:,None]], axis=-1)
        else:
            corr = np.concatenate([np.arange(source_idx.shape[0])[:, None], source_idx[:, None]], axis=-1)
             
        # build the ground truth label
        frag1 = src_keypts[corr[:, 0]]
        frag2 = tgt_keypts[corr[:, 1]]
        frag1_warp = transform(frag1, gt_trans)
        distance = np.sqrt(np.sum(np.power(frag1_warp - frag2, 2), axis=1))
        labels = (distance < self.inlier_threshold).astype(np.int)
        
        # prepare input to the network
        input_src_keypts = src_keypts[corr[:, 0]]
        input_tgt_keypts = tgt_keypts[corr[:, 1]]

        # input_src_keypts -- input_src_desc
        input_src_desc = src_desc[corr[:, 0]]
        # input_tgt_keypts -- input_tgt_desc
        input_tgt_desc = tgt_desc[corr[:, 1]]
        
        if self.in_dim == 3:
            corr_pos = input_src_keypts - input_tgt_keypts
        elif self.in_dim == 6:
            corr_pos = np.concatenate([input_src_keypts, input_tgt_keypts], axis=-1)
            # move the center of each point cloud to (0,0,0).
            corr_pos = corr_pos - corr_pos.mean(0)
        elif self.in_dim == 9:
            corr_pos = np.concatenate([input_src_keypts, input_tgt_keypts, input_src_keypts-input_tgt_keypts], axis=-1)
        elif self.in_dim == 70:
            corr_pos = np.concatenate([input_src_keypts, input_tgt_keypts], axis=-1)
            # move the center of each point cloud to (0,0,0).
            corr_pos = corr_pos - corr_pos.mean(0)
            corr_pos = np.concatenate([corr_pos, src_desc[corr[:,0]], tgt_desc[corr[:,1]]], axis=-1)
        elif self.in_dim == 12:
            src_pcd = make_point_cloud(src_keypts)
            tgt_pcd = make_point_cloud(tgt_keypts)
            estimate_normal(src_pcd, radius=self.downsample*2)
            estimate_normal(tgt_pcd, radius=self.downsample*2)
            src_normal = np.array(src_pcd.normals)
            tgt_normal = np.array(tgt_pcd.normals)  
            src_normal = src_normal[src_sel_ind, :]
            tgt_normal = tgt_normal[tgt_sel_ind, :]   
            input_src_normal = src_normal[corr[:, 0]]
            input_tgt_normal = tgt_normal[corr[:, 1]]
            corr_pos = np.concatenate([input_src_keypts, input_src_normal, input_tgt_keypts, input_tgt_normal], axis=-1)
        

        '''
            corr_pos.astype(np.float32)             :             根据特征选出的对应点(1000 point pairs),[1000,6]
            input_src_keypts.astype(np.float32)     :             对应点中的源点集合,[1000,3],corr[:,:3]
            input_tgt_keypts.astype(np.float32)     :             对应点中的目标点集合,[1000,3],corr[:,3:]
            gt_trans.astype(np.float32)             :             Ground Truth变换矩阵,[4,4]
            labels.astype(np.float32)               :             对应点的Ground Truth的标签（是否为内点）[1000]
            p_image.astype(np.float32)              :             src image, [1,3,120,160]
            q_image.astype(np.float32)              :             tgt image, [1,3,120,160]
            input_src_desc.astype(np.float32)       :             src desc, [1000,32]
            input_tgt_desc.astype(np.float32)       :             tgt desc, [1000,32]
        '''
        li= corr_pos.astype(np.float32), \
            input_src_keypts.astype(np.float32), \
            input_tgt_keypts.astype(np.float32), \
            gt_trans.astype(np.float32), \
            labels.astype(np.float32),\
            p_image.astype(np.float32),\
            q_image.astype(np.float32),\
            input_src_desc.astype(np.float32),\
            input_tgt_desc.astype(np.float32)

        return li
    
    def __len__(self):
        return self.gt_trans.keys().__len__()
    
    def __loadlog__(self, gtpath):
        with open(os.path.join(gtpath, 'gt.log')) as f:
            content = f.readlines()
        result = {}
        i = 0
        while i < len(content):
            line = content[i].replace("\n", "").split("\t")[0:3]
            trans = np.zeros([4, 4])
            trans[0] = np.fromstring(content[i+1], dtype=float, sep=' \t')
            trans[1] = np.fromstring(content[i+2], dtype=float, sep=' \t')
            trans[2] = np.fromstring(content[i+3], dtype=float, sep=' \t')
            trans[3] = np.fromstring(content[i+4], dtype=float, sep=' \t')
            i = i + 5
            result[f'{int(line[0])}_{int(line[1])}'] = trans
        return result

class ThreeDLOMatchTest(data.Dataset):
    def __init__(self, 
            root, 
            descriptor='fcgf',
            in_dim=6,
            inlier_threshold=0.10,
            num_node=5000, 
            use_mutual=True,
            downsample=0.03, 
            augment_axis=0, 
            augment_rotation=1.0,
            augment_translation=0.01,
            select_scene=None,
            ):
        self.root = root
        self.descriptor = descriptor
        assert descriptor in ['fcgf', 'fpfh']
        self.in_dim = in_dim
        self.inlier_threshold = inlier_threshold
        self.num_node = num_node
        self.use_mutual = use_mutual
        self.downsample = downsample
        self.augment_axis = augment_axis
        self.augment_rotation = augment_rotation
        self.augment_translation = augment_translation

        with open('misc/3DLoMatch.pkl', 'rb') as f:
            self.infos = pickle.load(f)
    
    def __len__(self):
        return len(self.infos['rot'])

    def __getitem__(self,item): 
        # get meta data
        gt_trans = integrate_trans(self.infos['rot'][item], self.infos['trans'][item])  
        scene = self.infos['src'][item].split('/')[1]
        src_id = self.infos['src'][item].split('/')[-1].split('_')[-1].replace('.pth', '')
        tgt_id = self.infos['tgt'][item].split('/')[-1].split('_')[-1].replace('.pth', '')

        # load point coordinates and pre-computed per-point local descriptors
        if self.descriptor == 'fcgf':
            src_data = np.load(f"{self.root}/fragments/{scene}/cloud_bin_{src_id}_fcgf.npz")
            tgt_data = np.load(f"{self.root}/fragments/{scene}/cloud_bin_{tgt_id}_fcgf.npz")
            src_keypts = src_data['xyz']
            tgt_keypts = tgt_data['xyz']
            src_features = src_data['feature']
            tgt_features = tgt_data['feature']
        elif self.descriptor == 'fpfh':
            src_data = np.load(f"{self.root}/fragments/{scene}/cloud_bin_{src_id}_fpfh.npz")
            tgt_data = np.load(f"{self.root}/fragments/{scene}/cloud_bin_{tgt_id}_fpfh.npz")
            src_keypts = src_data['xyz']
            tgt_keypts = tgt_data['xyz']
            src_features = src_data['feature']
            tgt_features = tgt_data['feature']
            src_features = src_features / (np.linalg.norm(src_features, axis=1, keepdims=True) + 1e-6)
            tgt_features = tgt_features / (np.linalg.norm(tgt_features, axis=1, keepdims=True) + 1e-6)

        # select {self.num_node} numbers of keypoints
        N_src = src_features.shape[0]
        N_tgt = tgt_features.shape[0]
        # use all point during test.
        if self.num_node == 'all':
            src_sel_ind = np.arange(N_src)
            tgt_sel_ind = np.arange(N_tgt)
        else:
            src_sel_ind = np.random.choice(N_src, self.num_node)
            tgt_sel_ind = np.random.choice(N_tgt, self.num_node)
        src_desc = src_features[src_sel_ind, :]
        tgt_desc = tgt_features[tgt_sel_ind, :]
        src_keypts = src_keypts[src_sel_ind, :]
        tgt_keypts = tgt_keypts[tgt_sel_ind, :]
        
        # construct the correspondence set by mutual nn in feature space.
        distance = np.sqrt(2 - 2 * (src_desc @ tgt_desc.T) + 1e-6)
        source_idx = np.argmin(distance, axis=1)
        if self.use_mutual:
            target_idx = np.argmin(distance, axis=0)
            mutual_nearest = (target_idx[source_idx] == np.arange(source_idx.shape[0]))
            corr = np.concatenate([np.where(mutual_nearest == 1)[0][:,None], source_idx[mutual_nearest][:,None]], axis=-1)
        else:
            corr = np.concatenate([np.arange(source_idx.shape[0])[:, None], source_idx[:, None]], axis=-1)
        
        # build the ground truth label
        frag1 = src_keypts[corr[:, 0]]
        frag2 = tgt_keypts[corr[:, 1]]
        frag1_warp = transform(frag1, gt_trans)
        distance = np.sqrt(np.sum(np.power(frag1_warp - frag2, 2), axis=1))
        labels = (distance < self.inlier_threshold).astype(np.int)
              
        # prepare input to the network
        input_src_keypts = src_keypts[corr[:, 0]]
        input_tgt_keypts = tgt_keypts[corr[:, 1]]

        if self.in_dim == 6:
            corr_pos = np.concatenate([input_src_keypts, input_tgt_keypts], axis=-1)
            # move the center of each point cloud to (0,0,0).
            corr_pos = corr_pos - corr_pos.mean(0)
        
        return corr_pos.astype(np.float32), \
            input_src_keypts.astype(np.float32), \
            input_tgt_keypts.astype(np.float32), \
            gt_trans.astype(np.float32), \
            labels.astype(np.float32),


if __name__ == "__main__":
    dset = ThreeDMatchTrainVal(root='/data/3DMatch', 
                       split='train',   
                       descriptor='fcgf',
                       num_node='all', 
                       use_mutual=False,
                       augment_axis=0, 
                       augment_rotation=0, 
                       augment_translation=0.00
                       )
    print(len(dset))  
    for i in range(dset.__len__()):
        ret_dict = dset[i]
