# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn

from detectron2.modeling import (
    META_ARCH_REGISTRY,
    build_backbone,
    detector_postprocess
)
from detectron2.structures import Boxes, ImageList, Instances

from .decoder import Decoder
from .loss import SetCriterion


@META_ARCH_REGISTRY.register()
class SimpleBaseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.SimpleBaseline.NUM_CLASSES
        self.num_queries = cfg.MODEL.SimpleBaseline.NUM_QUERIES
        self.hidden_dim = cfg.MODEL.SimpleBaseline.HIDDEN_DIM

        self.backbone = build_backbone(cfg)
        self.queries = nn.Embedding(self.num_queries, self.hidden_dim)
        self.decoder = Decoder(cfg, input_shape=self.backbone.output_shape())

        self.criterion = SetCriterion(cfg=cfg)

        mu = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        sigma = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - mu) / sigma

    def forward(self, batched_inputs):
        imgs, img_box = self.preprocess_image(batched_inputs)

        src = self.backbone(imgs.tensor)
        features = list()
        for f in self.in_features:
            feature = src[f]
            features.append(feature)

        init_boxes = img_box[:, None, :].repeat(1, self.num_queries, 1)
        output = self.decoder(features, init_boxes, self.queries.weight)

        if self.training:
            targets = [x["instances"].to(self.device) for x in batched_inputs]
            loss_dict = self.criterion(output, targets)
            return loss_dict
        else:
            pred_logits = output[-1]["pred_logits"]
            pred_boxes = output[-1]["pred_boxes"]
            pred_masks = output[-1]["pred_masks"]
            results = self.inference(pred_logits, pred_boxes, pred_masks, 
                                     imgs.image_sizes)

            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, imgs.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})

            return processed_results

    def inference(self, pred_logits, pred_boxes, pred_masks, image_sizes):
        assert len(pred_logits) == len(image_sizes)
        results = []

        scores = torch.sigmoid(pred_logits)
        labels = torch.arange(self.num_classes, device=self.device)
        labels = labels.unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)

        B, L, C = pred_logits.shape
        H, W = pred_masks.shape[-2:]
        pred_masks = pred_masks.view(B, L, C, H, W)

        for i, (score, box, mask, image_size) in enumerate(
                zip(scores, pred_boxes, pred_masks, image_sizes)):
            result = Instances(image_size)
            score, topk_indices = score.flatten(0, 1).topk(100, sorted=False)
            result.scores = score

            labels_per_image = labels[topk_indices]
            result.pred_classes = labels_per_image

            box = box.view(-1, 1, 4).repeat(1, self.num_classes, 1)
            box = box.view(-1, 4)[topk_indices]
            result.pred_boxes = Boxes(box)

            mask = mask.view(-1, H, W)[topk_indices]
            result.pred_masks = mask.unsqueeze(1)

            results.append(result)

        return results

    def preprocess_image(self, inputs):
        """Normalize, pad and batch the input images.
        """
        imgs = [self.normalizer(x["image"].to(self.device)) for x in inputs]
        imgs = ImageList.from_tensors(
            imgs, 
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints
        )

        img_box = list()
        for item in inputs:
            h, w = item["image"].shape[-2:]
            img_box.append(torch.tensor([0, 0, w, h], dtype=torch.float32))
        img_box = torch.stack(img_box).to(self.device)

        return imgs, img_box
