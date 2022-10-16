# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy

import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from hfai.datasets.coco import CocoReader


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """
    size = cfg.INPUT.MAX_SIZE_TRAIN
    if is_train:
        transforms = [
            T.RandomFlip(horizontal=True),  # flip first
            T.ResizeScale(
                min_scale=0.1, max_scale=2.0, 
                target_height=size, target_width=size
            ),
            T.FixedSizeCrop(crop_size=(size, size), pad=False),
        ]
    else:
        transforms = [
            T.ResizeShortestEdge(short_edge_length=size, max_size=size),
        ]
    return transforms


class SimpleBaselineDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by SimpleBaseline.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """
    def __init__(self, cfg, is_train=True):
        self.tfm_gens = build_transform_gen(cfg, is_train)
        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train

        split = "train" if is_train else "val"
        self.reader = CocoReader(split)
        from pycocotools.coco import COCO
        coco_api = COCO(f"coco/annotations/instances_{split}2017.json")
        self.img_ids = sorted(coco_api.imgs.keys())

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        image_id = dataset_dict["image_id"]
        index = self.img_ids.index(image_id)
        image = self.reader.read_imgs([index])[0]
        utils.check_image_size(dataset_dict, image)

        image, tfms = T.apply_transform_gens(self.tfm_gens, image)
        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of 
        # pickle & mp.Queue. Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                anno.pop("keypoints", None)

            annos = [
                utils.transform_instance_annotations(obj, tfms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape, "bitmask")
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        return dataset_dict
