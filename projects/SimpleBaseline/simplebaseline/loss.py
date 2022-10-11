# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.distributed as dist
import torch.nn.functional as F
from fvcore.nn import sigmoid_focal_loss_jit
from scipy.optimize import linear_sum_assignment

from detectron2.utils.comm import get_world_size
from detectron2.structures import BitMasks

from .box_ops import generalized_box_iou
from .dice_loss import DiceLoss


class SetCriterion:
    def __init__(self, cfg):
        self.matcher = HungarianMatcher(cfg)
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.num_queries = cfg.MODEL.SimpleBaseline.NUM_QUERIES
        self.mask_size = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION

        self.focal_loss_alpha = cfg.MODEL.SimpleBaseline.ALPHA
        self.focal_loss_gamma = cfg.MODEL.SimpleBaseline.GAMMA
        self.focal_weight = cfg.MODEL.SimpleBaseline.FOCAL_WEIGHT
        self.l1_weight = cfg.MODEL.SimpleBaseline.L1_WEIGHT
        self.giou_weight = cfg.MODEL.SimpleBaseline.GIOU_WEIGHT
        self.loss_dice = DiceLoss(cfg.MODEL.SimpleBaseline.DICE_WEIGHT)

    def prepare_targets(self, targets):
        new_targets = []
        for target in targets:
            h, w = target.image_size
            image_size = torch.as_tensor([w, h, w, h], dtype=torch.float)
            new_targets.append({
                "image_size": image_size.unsqueeze(0).to(self.device),
                "labels": target.gt_classes.to(self.device),
                "boxes": target.gt_boxes.tensor.to(self.device),
            })
        return new_targets

    def prepare_mask_targets(self, targets_, indices, pred):
        gt_masks = []
        for idx in range(len(targets_)):
            target_idx_per_img = indices[idx][1]
            query_idx_per_img = indices[idx][0]
            pred_box_per_img = pred['pred_boxes'][idx, query_idx_per_img]

            gt_masks_per_img = targets_[idx].gt_masks.tensor[target_idx_per_img]
            gt_masks_per_img_ = BitMasks(gt_masks_per_img)
            _gt_masks_per_img = gt_masks_per_img_.crop_and_resize(
                pred_box_per_img, pred['pred_masks'].size(2)
            )
            gt_masks.append(_gt_masks_per_img)
        return torch.cat(gt_masks)

    def __call__(self, preds, targets_):
        targets = self.prepare_targets(targets_)

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([float(num_boxes)], device=self.device)
        if dist.is_available() and dist.is_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        for i, pred in enumerate(preds[::-1]):
            indices = self.matcher(pred, targets)
            query_idx = torch.cat([idx for (idx, _) in indices])
            batch_idx = torch.cat(
                [torch.full_like(idx, j) for j, (idx, _) in enumerate(indices)]
            )

            # focal loss
            logits = pred['pred_logits']
            logits = logits.flatten(0, 1)
            labels = torch.zeros_like(logits)

            flattened_idx = batch_idx * self.num_queries + query_idx
            permutated_label_idx = torch.cat(
                [t["labels"][j] for t, (_, j) in zip(targets, indices)]
            )
            labels[flattened_idx, permutated_label_idx] = 1

            loss_focal = sigmoid_focal_loss_jit(
                logits,
                labels,
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="sum"
            )
            loss_focal = loss_focal / num_boxes * self.focal_weight
            losses[f'loss_focal_{i}'] = loss_focal

            # l1 loss + giou loss
            pred_boxes = pred['pred_boxes'][batch_idx, query_idx]
            tgt_boxes = torch.cat(
                [t['boxes'][j] for t, (_, j) in zip(targets, indices)]
            )
            loss_giou = 1 - torch.diag(generalized_box_iou(pred_boxes, tgt_boxes))
            losses[f'loss_giou_{i}'] = loss_giou.sum() / num_boxes * self.giou_weight

            normalizer = torch.cat(
                [v["image_size"].repeat(len(v["boxes"]), 1) for v in targets]
            )
            pred_boxes_ = pred_boxes / normalizer
            tgt_boxes_ = tgt_boxes / normalizer
            loss_l1 = F.l1_loss(pred_boxes_, tgt_boxes_, reduction='none')
            losses[f'loss_l1_{i}'] = loss_l1.sum() / num_boxes * self.l1_weight

            # dice loss
            pred_masks = pred['pred_masks'][flattened_idx, permutated_label_idx]
            tgt_masks = self.prepare_mask_targets(
                targets_, indices, pred
            )
            losses[f'loss_dice_{i}'] = self.loss_dice(pred_masks, tgt_masks, 
                                                      avg_factor=num_boxes)

        return losses


class HungarianMatcher:
    def __init__(self, cfg):
        self.focal_weight = cfg.MODEL.SimpleBaseline.FOCAL_WEIGHT
        self.l1_weight = cfg.MODEL.SimpleBaseline.L1_WEIGHT
        self.giou_weight = cfg.MODEL.SimpleBaseline.GIOU_WEIGHT
        self.focal_loss_alpha = cfg.MODEL.SimpleBaseline.ALPHA
        self.focal_loss_gamma = cfg.MODEL.SimpleBaseline.GAMMA

    @torch.no_grad()
    def __call__(self, preds, targets):
        """
        Args:
            preds:
              - pred_logits: [batch_size, num_queries, num_classes]
              - pred_boxes: [batch_size, num_queries, 4]
            targets: a list of (length = batch_size)
              - labels: [num_target_boxes]
              - boxes: [num_target_boxes, 4]
              - image_size: [1, 4]
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j):
              - index_i: the indices of the selected predictions (in order)
              - index_j: the indices of the corresponding targets (in order)
        """
        bs, num_queries = preds["pred_logits"].shape[:2]

        pred_prob = preds["pred_logits"].flatten(0, 1).sigmoid()
        pred_boxes = preds["pred_boxes"].flatten(0, 1)

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_boxes = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, 
        # we don't use the NLL, but approximate it in 1 - prob[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        alpha = self.focal_loss_alpha
        gamma = self.focal_loss_gamma
        neg_cost = (1 - alpha) * (pred_prob ** gamma) * \
            (-(1 - pred_prob + 1e-8).log())
        pos_cost = alpha * ((1 - pred_prob) ** gamma) * \
            (-(pred_prob + 1e-8).log())
        cost_focal = pos_cost[:, tgt_ids] - neg_cost[:, tgt_ids]

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(pred_boxes, tgt_boxes)

        # Compute the L1 cost between boxes
        normalizer = torch.cat([v["image_size"] for v in targets])
        normalizer = normalizer.unsqueeze(1).repeat(1, num_queries, 1)
        pred_boxes = pred_boxes / normalizer.flatten(0, 1)
        normalizer = torch.cat(
            [v["image_size"].repeat(len(v["boxes"]), 1) for v in targets]
        )
        tgt_boxes = tgt_boxes / normalizer
        cost_l1 = torch.cdist(pred_boxes, tgt_boxes, p=1)

        # Final cost matrix
        C = self.l1_weight * cost_l1 + self.focal_weight * cost_focal \
            + self.giou_weight * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(C.split(sizes, -1))
        ]
        return [(torch.as_tensor(i, dtype=torch.int64), 
                 torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
