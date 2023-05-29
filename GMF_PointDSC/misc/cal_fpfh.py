import os
import open3d as o3d
import numpy as np
import shutil
import glob

kitti_cache = {}
kitti_icp_cache = {}

def process_kitti(voxel_size=0.30, split='train'):
    def odometry_to_positions(odometry):
        T_w_cam0 = odometry.reshape(3, 4)
        T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
        return T_w_cam0

    def get_video_odometry(root, drive, indices=None, ext='.txt', return_all=False):
        data_path = root + '/poses/%02d.txt' % drive
        if data_path not in kitti_cache:
            kitti_cache[data_path] = np.genfromtxt(data_path)
        if return_all:
            return kitti_cache[data_path]
        else:
            return kitti_cache[data_path][indices]

    def _get_velodyne_fn(root, drive, t):
        fname = root + '/sequences/%02d/velodyne/%06d.bin' % (drive, t)
        return fname

    def apply_transform(pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    MIN_DIST = 10
    root = '/ssd2/xuyang/KITTI/dataset/'
    R = np.array([
        7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
        -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
    ]).reshape(3, 3)
    T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
    velo2cam = np.hstack([R, T])
    velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T

    subset_names = open(f'misc/split/{split}_kitti.txt').read().split()
    files = []
    for dirname in subset_names:
        drive_id = int(dirname)
        # inames = get_all_scan_ids(root, drive_id)
        # for start_time in inames:
        #     for time_diff in range(2, max_time_diff):
        #         pair_time = time_diff + start_time
        #         if pair_time in inames:
        #             files.append((drive_id, start_time, pair_time))
        fnames = glob.glob(root + '/sequences/%02d/velodyne/*.bin' % drive_id)
        assert len(fnames) > 0, f"Make sure that the path {root} has data {dirname}"
        inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

        all_odo = get_video_odometry(root, drive_id, return_all=True)
        all_pos = np.array([odometry_to_positions(odo) for odo in all_odo])
        Ts = all_pos[:, :3, 3]
        pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3)) ** 2
        pdist = np.sqrt(pdist.sum(-1))
        more_than_10 = pdist > MIN_DIST
        curr_time = inames[0]
        while curr_time in inames:
            next_time = np.where(more_than_10[curr_time][curr_time:curr_time + 100])[0]
            if len(next_time) == 0:
                curr_time += 1
            else:
                next_time = next_time[0] + curr_time - 1

            if next_time in inames:
                files.append((drive_id, curr_time, next_time))
                curr_time = next_time + 1
        # Remove problematic sequence
        for item in [
            (8, 15, 58),
        ]:
            if item in files:
                files.pop(files.index(item))

    # begin extracting features
    for idx in range(len(files)):
        drive = files[idx][0]
        t0, t1 = files[idx][1], files[idx][2]
        all_odometry = get_video_odometry(root, drive, [t0, t1])
        positions = [odometry_to_positions(odometry) for odometry in all_odometry]
        fname0 = _get_velodyne_fn(root, drive, t0)
        fname1 = _get_velodyne_fn(root, drive, t1)

        # XYZ and reflectance
        xyzr0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)
        xyzr1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)
        xyz0 = xyzr0[:, :3]
        xyz1 = xyzr1[:, :3]

        key = '%d_%d_%d' % (drive, t0, t1)
        filename = root + 'icp/' + key + '.npy'
        if key not in kitti_icp_cache:
            if not os.path.exists(filename):
                # work on the downsampled xyzs, 0.05m == 5cm
                sel0 = ME.utils.sparse_quantize(xyz0 / 0.05, return_index=True)
                sel1 = ME.utils.sparse_quantize(xyz1 / 0.05, return_index=True)

                M = (velo2cam @ positions[0].T @ np.linalg.inv(positions[1].T)
                     @ np.linalg.inv(velo2cam)).T
                xyz0_t = apply_transform(xyz0[sel0], M)
                pcd0 = make_point_cloud(xyz0_t)
                pcd1 = make_point_cloud(xyz1[sel1])
                reg = o3d.registration.registration_icp(
                    pcd0, pcd1, 0.2, np.eye(4),
                    o3d.registration.TransformationEstimationPointToPoint(),
                    o3d.registration.ICPConvergenceCriteria(max_iteration=200))
                pcd0.transform(reg.transformation)
                # pcd0.transform(M2) or self.apply_transform(xyz0, M2)
                M2 = M @ reg.transformation
                # o3d.draw_geometries([pcd0, pcd1])
                # write to a file
                np.save(filename, M2)
            else:
                M2 = np.load(filename)
            kitti_icp_cache[key] = M2
        else:
            M2 = kitti_icp_cache[key]

        # pc = o3d.io.read_point_cloud(ply_path)
        # points = np.asarray(pc.points)
        # pcd = pc.voxel_down_sample(voxel_size=voxel_size)
        # xyz_down = np.asarray(pcd.points)
        # estimate the normals and compute fpfh descriptor
        pcd0.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd0,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 5,
                max_nn=100
            )
        )
        features0 = np.array(fpfh.data).T

        pcd1.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd1,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 5,
                max_nn=100
            )
        )
        features0 = np.array(fpfh.data).T

        # xyz_down0, features0 = extract_features(
        #     model,
        #     xyz=xyz0,
        #     rgb=None,
        #     normal=None,
        #     voxel_size=voxel_size,
        #     skip_check=True,
        # )
        # xyz_down1, features1 = extract_features(
        #     model,
        #     xyz=xyz1,
        #     rgb=None,
        #     normal=None,
        #     voxel_size=voxel_size,
        #     skip_check=True,
        # )
        filename = f"{root}/feat_{split}/drive{drive}-pair{t0}_{t1}"
        np.savez_compressed(
            filename,
            xyz0=xyz_down0.astype(np.float32),
            xyz1=xyz_down1.astype(np.float32),
            features0=features0.detach().cpu().numpy().astype(np.float32),
            features1=features1.detach().cpu().numpy().astype(np.float32),
            gt_trans=M2
        )
        print(filename)

def process_3dmatch_train(voxel_size=0.05):

    # root = "/DISK/qwt/datasets/threedmatch/"
    root = "/DISK/qwt/datasets/Ours_train_0_01/train"
    # save_path = "/DISK/qwt/datasets/threedmatch/threedmatch_feat"
    save_path = "/DISK/qwt/pointDSC/FPFH_feat_train"
    # scene
    scenes = [scene.split("/")[-1] for scene in os.listdir(root)]
    for scene in scenes:
        seqs_path = os.path.join(root,scene,"seq-*")
        # seq-*
        seqs = [seq.split("/")[-1] for seq in glob.glob(seqs_path)]
        for seq in seqs:
            plys_path = os.path.join(root,scene,seq,"*.ply")
            # cloud_bin_*
            plys = [ply.split("/")[-1].split(".")[0] for ply in glob.glob(plys_path)]
            for ply in plys:
                ply_path = os.path.join(root,scene,seq,f"{ply}.ply")
                pc = o3d.io.read_point_cloud(ply_path)
                points = np.asarray(pc.points)

                pcd = pc.voxel_down_sample(voxel_size=voxel_size)
                xyz_down = np.asarray(pcd.points)

                # estimate the normals and compute fpfh descriptor
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
                fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    pcd,
                    o3d.geometry.KDTreeSearchParamHybrid(
                        radius=voxel_size * 5,
                        max_nn=100
                    )
                )
                features = np.array(fpfh.data).T

                folder_path = os.path.join(save_path,scene,seq)
                if(not os.path.exists(folder_path)):
                    os.makedirs(folder_path)
                ply_save_path = os.path.join(folder_path,f"{ply}_fpfh.npz")
                np.savez_compressed(
                    ply_save_path,
                    points=points.astype(np.float32),
                    xyz=xyz_down.astype(np.float32),
                    feature=features.astype(np.float32)
                )
                suffix = "png"
                if(scene.__contains__("analysis")):
                    suffix = "jpg"
                png_path = os.path.join(root,scene,seq,f"{ply}_0.{suffix}")
                png_save_path = os.path.join(folder_path,f"{ply}_0.{suffix}")
                shutil.copy(png_path,png_save_path)

                print(f"Saving : {ply_save_path}")
                print(f"Saving : {png_save_path}")
                print("-------------------------------------------------------")


def process_3dmatch_test(voxel_size=0.05):
    scenes = [
        '7-scenes-redkitchen',
        'sun3d-home_at-home_at_scan1_2013_jan_1',
        'sun3d-home_md-home_md_scan9_2012_sep_30',
        'sun3d-hotel_uc-scan3',
        'sun3d-hotel_umd-maryland_hotel1',
        'sun3d-hotel_umd-maryland_hotel3',
        'sun3d-mit_76_studyroom-76-1studyroom2',
        'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
    ]

    root = "/DISK/qwt/datasets/Ours_train_0_01/test"
    save_path = "/DISK/qwt/pointDSC/FPFH_feat_test"
    # scene
    # scenes = [scene.split("/")[-1] for scene in os.listdir(root)]
    for scene in scenes:
        seqs_path = os.path.join(root, scene, "seq-*")
        # seq-*
        seqs = [seq.split("/")[-1] for seq in glob.glob(seqs_path)]
        for seq in seqs:
            plys_path = os.path.join(root, scene, seq, "*.ply")
            # cloud_bin_*
            plys = [ply.split("/")[-1].split(".")[0] for ply in glob.glob(plys_path)]
            for ply in plys:

                # ------------------ feat ------------------
                ply_path = os.path.join(root, scene, seq, f"{ply}.ply")
                pc = o3d.io.read_point_cloud(ply_path)
                points = np.asarray(pc.points)

                pcd = pc.voxel_down_sample(voxel_size=voxel_size)
                xyz_down = np.asarray(pcd.points)

                # estimate the normals and compute fpfh descriptor
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
                fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    pcd,
                    o3d.geometry.KDTreeSearchParamHybrid(
                        radius=voxel_size * 5,
                        max_nn=100
                    )
                )
                features = np.array(fpfh.data).T

                folder_path = os.path.join(save_path, scene, seq)
                if (not os.path.exists(folder_path)):
                    os.makedirs(folder_path)
                ply_save_path = os.path.join(folder_path, f"{ply}_fpfh.npz")
                np.savez_compressed(
                    ply_save_path,
                    points=points.astype(np.float32),
                    xyz=xyz_down.astype(np.float32),
                    feature=features.astype(np.float32)
                )
                # ------------------ feat ------------------

                # ------------------ png ------------------
                suffix = "png"
                if (scene.__contains__("analysis")):
                    suffix = "jpg"
                png_path = os.path.join(root, scene, seq, f"{ply}_0.{suffix}")
                png_save_path = os.path.join(folder_path, f"{ply}_0.{suffix}")
                shutil.copy(png_path, png_save_path)
                # ------------------ png ------------------

                # ------------------ gt.log ------------------
                gtlog_save_path = os.path.join(save_path, scene, "gt.log")
                if (not os.path.exists(gtlog_save_path)):
                    gtlog_path = os.path.join(root, scene, "gt.log")
                    shutil.copy(gtlog_path, gtlog_save_path)
                    print(f"Saving : {gtlog_save_path}")
                # ------------------ gt.log ------------------

                print(f"Saving : {ply_save_path}")
                print(f"Saving : {png_save_path}")
                print("-------------------------------------------------------")


def process_redwood(voxel_size=0.05):
    scene_list = [
        'livingroom1-simulated',
        'livingroom2-simulated',
        'office1-simulated',
        'office2-simulated'
    ]
    for scene in scene_list:
        scene_path = os.path.join("/data/Augmented_ICL-NUIM/", scene + '/fragments')
        pcd_list = os.listdir(scene_path)
        for pcd_path in pcd_list:
            if not pcd_path.endswith('.ply'):
                continue
            full_path = os.path.join(scene_path, pcd_path)
            orig_pcd = o3d.io.read_point_cloud(full_path)
            # voxel downsample 
            pcd = orig_pcd.voxel_down_sample(voxel_size=voxel_size)

            # estimate the normals and compute fpfh descriptor
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
            fpfh = o3d.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
            fpfh_np = np.array(fpfh.data).T

            # save the data for training.
            np.savez_compressed(
                full_path.replace('.ply', '_fpfh'),
                points=np.array(orig_pcd.points).astype(np.float32),
                xyz=np.array(pcd.points).astype(np.float32),
                feature=fpfh_np.astype(np.float32),
            )
            print(full_path, fpfh_np.shape)


if __name__ == '__main__':
    # process_3dmatch_train(voxel_size=0.05)
    # process_3dmatch_test(voxel_size=0.05)
    # process_redwood(voxel_size=0.05)
    process_kitti()
