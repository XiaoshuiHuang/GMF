import os
import time
import shutil
import json 
from config_3DMatch import get_config
from easydict import EasyDict as edict
from libs.loss import TransformationLoss, ClassificationLoss, SpectralMatchingLoss
from datasets.ThreeDMatch import ThreeDMatchTrainVal
from datasets.dataloader import get_dataloader
from libs.trainer import Trainer
from models.PointDSC import PointDSC
from torch import optim


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    config = get_config()
    dconfig = vars(config)

    for k in dconfig:
        print(f"    {k}: {dconfig[k]}")
    config = edict(dconfig)

    os.makedirs(config.snapshot_dir, exist_ok=True)
    os.makedirs(config.tboard_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)

    shutil.copy2(os.path.join('.', 'train_3DMatch.py'), os.path.join(config.snapshot_dir, 'train.py'))
    shutil.copy2(os.path.join('.', 'libs/trainer.py'), os.path.join(config.snapshot_dir, 'trainer.py'))
    shutil.copy2(os.path.join('.', 'models/PointDSC.py'), os.path.join(config.snapshot_dir, 'model.py'))  # for the model setting.
    shutil.copy2(os.path.join('.', 'libs/loss.py'), os.path.join(config.snapshot_dir, 'loss.py'))
    shutil.copy2(os.path.join('.', 'datasets/ThreeDMatch.py'), os.path.join(config.snapshot_dir, 'dataset.py'))
    json.dump(
        config,
        open(os.path.join(config.snapshot_dir, 'config.json'), 'w'),
        indent=4,
    )

    config.model = PointDSC(
        in_dim=config.in_dim,                           # 6
        num_layers=config.num_layers,                   # 12
        num_channels=config.num_channels,               # 128
        num_iterations=config.num_iterations,           # 10
        inlier_threshold=config.inlier_threshold,       # 0.10
        sigma_d=config.sigma_d,                         # 0.10
        ratio=config.ratio,                             # 0.1
        k=config.k,
    )

    if config.optimizer == 'SGD':
        config.optimizer = optim.SGD(
            config.model.parameters(), 
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            )
    elif config.optimizer == 'ADAM':
        config.optimizer = optim.Adam(
            config.model.parameters(), 
            lr=config.lr,
            betas=(0.9, 0.999),
            # momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    config.scheduler = optim.lr_scheduler.ExponentialLR(
        config.optimizer,
        gamma=config.scheduler_gamma,
    )

    train_set = ThreeDMatchTrainVal(
                        root=config.root,
                        descriptor=config.descriptor,
                        split='train', 
                        in_dim=config.in_dim,
                        inlier_threshold=config.inlier_threshold,
                        num_node=config.num_node, 
                        use_mutual=config.use_mutual,
                        downsample=config.downsample,
                        augment_axis=config.augment_axis,
                        augment_rotation=config.augment_rotation,
                        augment_translation=config.augment_translation,
                        config=config
    )
    val_set = ThreeDMatchTrainVal(
                        root=config.root,
                        split='val',
                        descriptor=config.descriptor,
                        in_dim=config.in_dim,
                        inlier_threshold=config.inlier_threshold,
                        num_node=config.num_node,
                        use_mutual=config.use_mutual,
                        downsample=config.downsample,
                        augment_axis=config.augment_axis,
                        augment_rotation=config.augment_rotation,
                        augment_translation=config.augment_translation,
                        config=config
    )
    config.train_loader = get_dataloader(
                        dataset=train_set,
                        batch_size=config.batch_size,
                        num_workers=config.num_workers,
    )
    config.val_loader = get_dataloader(
                        dataset=val_set,
                        batch_size=config.batch_size,
                        num_workers=config.num_workers,
    )
    
    # create evaluation
    config.evaluate_metric = {
        "ClassificationLoss": ClassificationLoss(balanced=config.balanced),
        "SpectralMatchingLoss": SpectralMatchingLoss(balanced=config.balanced),
        "TransformationLoss": TransformationLoss(re_thre=config.re_thre, te_thre=config.te_thre),
    }
    config.metric_weight = {
        "ClassificationLoss": config.weight_classification,
        "SpectralMatchingLoss": config.weight_spectralmatching,
        "TransformationLoss": config.weight_transformation,
    }


    trainer = Trainer(config)
    trainer.train()