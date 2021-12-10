import torch
from .sceneflow import SceneFlowDataset
from .kitti2012 import KITTI2012Dataset
from .kitti2015 import KITTI2015Dataset


def build_dataset(args, training):
    if training:
        data_type = args.data_type_train
        data_root = args.data_root_train
        data_list = args.data_list_train
        data_size = args.data_size_train
    else:
        data_type = args.data_type_val
        data_root = args.data_root_val
        data_list = args.data_list_val
        data_size = args.data_size_val

    datasets = []
    for d_type, d_root, d_list in zip(data_type, data_root, data_list):
        if d_type == "SceneFlow":
            dataset = SceneFlowDataset(
                image_list=d_list,
                root=d_root,
                crop_size=data_size,
                training=training,
                augmentation=args.data_augmentation,
            )
        elif d_type == "KITTI2015":
            dataset = KITTI2015Dataset(
                image_list=d_list,
                root=d_root,
                crop_size=data_size,
                training=training,
                augmentation=args.data_augmentation,
            )
        elif d_type == "KITTI2012":
            dataset = KITTI2012Dataset(
                image_list=d_list,
                root=d_root,
                crop_size=data_size,
                training=training,
                augmentation=args.data_augmentation,
            )
        else:
            raise NotImplementedError

        datasets.append(dataset)

    return torch.utils.data.ConcatDataset(datasets)
