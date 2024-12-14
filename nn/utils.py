from collections import namedtuple
import math
from copy import deepcopy
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import dist2bbox, make_anchors
from .base import BaseModel
import torch.nn.init as init
import math
from functools import partial
from typing import Callable
from torch import optim



def select_candidates_in_gts(xy_centers: torch.Tensor, gt_bboxes: torch.Tensor, eps: float = 1e-9, roll_out: bool = False) -> torch.Tensor:
    """
    Selects candidate anchor centers within ground truth bounding boxes.

    This function calculates the minimum distance (delta) between each anchor center and all four edges (left, top, right, bottom)
    of each ground truth bounding box in a batch. It then returns a boolean tensor indicating whether the minimum distance
    for each anchor-ground truth pair is greater than a small epsilon value. This essentially selects anchor centers that
    are "inside" the ground truth bounding boxes.

    Args:
        xy_centers (torch.Tensor): A tensor of shape (h*w, 4) representing anchor center coordinates (x, y).
        gt_bboxes (torch.Tensor): A tensor of shape (b, n_boxes, 4) representing ground truth bounding boxes (x1, y1, x2, y2) in a batch.
            - b: number of images in the batch
            - n_boxes: number of ground truth boxes per image
        eps (float, optional): A small epsilon value to prevent division by zero and numerical instability. Defaults to 1e-9.
        roll_out (bool, optional): If True, performs the calculation in a loop for each image in the batch.
            This can be memory-intensive for large batches. Defaults to False.

    Returns:
        torch.Tensor: A boolean tensor of shape (b, n_boxes, h*w) indicating which anchor centers are considered candidates
            within each ground truth bounding box in the batch. True indicates a candidate, False indicates not a candidate.
    """

    n_anchors = xy_centers.shape[0]  # Total number of anchor centers
    bs, n_boxes, _ = gt_bboxes.shape  # Batch size, number of ground truth boxes per image, and unused dimension

    if roll_out:
        # Perform calculation in a loop for each image (memory intensive for large batches)
        bbox_deltas = torch.empty((bs, n_boxes, n_anchors), device=gt_bboxes.device)
        for b in range(bs):
            # Split ground truth bounding boxes into left-top and right-bottom corners
            lt, rb = gt_bboxes[b].view(-1, 1, 4).chunk(2, 2)  # lt: (n_boxes, 1, 4), rb: (n_boxes, 1, 4)

            # Calculate minimum distance (delta) to all edges for each anchor center
            bbox_deltas[b] = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(n_boxes, n_anchors, -1).amin(2).gt_(eps)
        return bbox_deltas

    else:
        # Efficient vectorized calculation for all images in the batch
        # Split ground truth bounding boxes into left-top and right-bottom corners
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # lt: (bs * n_boxes, 1, 4), rb: (bs * n_boxes, 1, 4)

        # Calculate minimum distance (delta) to all edges for each anchor-ground truth pair
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        return bbox_deltas.amin(3).gt_(eps)  # Minimum delta across last dimension (edges) > epsilon

def select_highest_overlaps(mask_pos: torch.Tensor, overlaps: torch.Tensor, n_max_boxes: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Selects the ground truth box with the highest IoU for each anchor when an anchor is assigned to multiple ground truths.

    Args:
        mask_pos (torch.Tensor): A boolean mask of shape (b, n_max_boxes, h*w).
            mask_pos[b, i, j] = True indicates that anchor j is potentially assigned to ground truth box i in batch b.
        overlaps (torch.Tensor): A tensor of shape (b, n_max_boxes, h*w) containing the IoUs between each anchor and each ground truth box.
            overlaps[b, i, j] is the IoU between anchor j and ground truth box i in batch b.
        n_max_boxes (int): The maximum number of ground truth boxes per image.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - target_gt_idx (torch.Tensor): A tensor of shape (b, h*w) indicating the index of the assigned ground truth box for each anchor.
            - fg_mask (torch.Tensor): A boolean mask of shape (b, h*w) indicating which anchors are assigned to any ground truth box (foreground).
            - mask_pos (torch.Tensor): The updated positive mask after resolving conflicts, shape (b, n_max_boxes, h*w).
    """

    # Calculate the foreground mask by summing across the n_max_boxes dimension.
    # fg_mask[b, j] will be > 1 if anchor j is assigned to multiple GTs in batch b.
    fg_mask = mask_pos.sum(-2)  # (b, h*w)

    # Check if any anchor is assigned to multiple ground truth boxes.
    if fg_mask.max() > 1:
        # Create a mask indicating where anchors are assigned to multiple GTs.
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, n_max_boxes, 1])  # (b, n_max_boxes, h*w)

        # Find the index of the ground truth box with the highest overlap (IoU) for each anchor.
        max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)

        # Create a one-hot encoding of the max overlap indices.
        is_max_overlaps = F.one_hot(max_overlaps_idx, n_max_boxes)  # (b, h*w, n_max_boxes)

        # Permute the dimensions to match mask_pos for element-wise operations.
        is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)  # (b, n_max_boxes, h*w)

        # Update mask_pos to keep only the assignment with the highest overlap.
        # If there was a conflict (mask_multi_gts is True), use the one-hot encoding of the max overlap index.
        # Otherwise, keep the original mask_pos value.
        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)  # (b, n_max_boxes, h*w)

        # Recalculate the foreground mask after resolving conflicts.
        fg_mask = mask_pos.sum(-2)  # (b, h*w)

    # Find the index of the assigned ground truth box for each anchor based on the updated mask_pos.
    target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)

    return target_gt_idx, fg_mask, mask_pos

def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, xywh: bool = True, GIoU: bool = False, DIoU: bool = False, CIoU: bool = False, eps: float = 1e-7) -> torch.Tensor:
    """
    Calculates Intersection over Union (IoU) and optionally other related metrics (GIoU, DIoU, CIoU) between two sets of bounding boxes.

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing bounding boxes.
            If xywh is True, the format is (x_center, y_center, width, height).
            If xywh is False, the format is (x1, y1, x2, y2) (top-left and bottom-right coordinates).
        box2 (torch.Tensor): A tensor of shape (N, 4) representing bounding boxes in the same format as box1.
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. Defaults to True.
        GIoU (bool, optional): If True, calculates Generalized IoU. Defaults to False.
        DIoU (bool, optional): If True, calculates Distance IoU. Defaults to False.
        CIoU (bool, optional): If True, calculates Complete IoU. Defaults to False.
        eps (float, optional): A small epsilon value to prevent division by zero. Defaults to 1e-7.

    Returns:
        torch.Tensor: A tensor of shape (N,) containing the IoU (or GIoU, DIoU, CIoU) values for each pair of bounding boxes.
    """

    # Convert boxes to (x1, y1, x2, y2) format if they are in (x, y, w, h) format
    if xywh:
        # Split box1 and box2 into their respective components
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        # Calculate half-widths and half-heights
        w1_half, h1_half, w2_half, h2_half = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        # Calculate x1, y1, x2, y2 for box1
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_half, x1 + w1_half, y1 - h1_half, y1 + h1_half
        # Calculate x1, y1, x2, y2 for box2
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_half, x2 + w2_half, y2 - h2_half, y2 + h2_half
    else:  # Already in (x1, y1, x2, y2) format
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps


    # Calculate intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Calculate union area
    union = w1 * h1 + w2 * h2 - inter + eps

    # Calculate IoU
    iou = inter / union

    # Calculate GIoU, DIoU, or CIoU if requested
    if CIoU or DIoU or GIoU:
        # Calculate convex (smallest enclosing box) width and height
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)

        if CIoU or DIoU:  # Distance or Complete IoU
            c2 = cw ** 2 + ch ** 2 + eps  # Convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # Center distance squared
            if CIoU: # Complete IoU
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # Convex area
        return iou - (c_area - union) / c_area  # GIoU
    return iou  # IoU

def bbox2dist(anchor_points: torch.Tensor, bbox: torch.Tensor, reg_max: float) -> torch.Tensor:
    """
    Transforms bounding boxes (format xyxy) into distances from anchor points to bounding box edges (format ltrb).

    This function takes anchor points and bounding boxes as input and calculates the distances between each anchor point and
    the four edges (left, top, right, bottom) of the bounding box. The output is a tensor containing these distances in the format ltrb
    (left top right bottom).

    Args:
        anchor_points (torch.Tensor): A tensor of shape (N, 2) representing anchor point coordinates (x, y).
        bbox (torch.Tensor): A tensor of shape (N, 4) representing bounding boxes in xyxy format (x1, y1, x2, y2).
        reg_max (float): The maximum value for the distances (ltrb).

    Returns:
        torch.Tensor: A tensor of shape (N, 4) containing the distances (ltrb) from each anchor point to the bounding box edges.
    """

    # Split the bounding box coordinates into separate tensors for x1, y1, x2, and y2
    x1y1, x2y2 = torch.split(bbox, 2, dim=-1)  # x1y1: (N, 2), x2y2: (N, 2)

    # Calculate the distances from the anchor points to each edge of the bounding box
    # and concatenate them into a single tensor
    ltrb = torch.cat((anchor_points - x1y1, x2y2 - anchor_points), dim=-1)  # ltrb: (N, 4)

    # Clamp the distances to be within the range (0, reg_max - 0.01)
    # This ensures that the distances stay positive and avoid potential issues at the boundaries.
    return ltrb.clamp(0, reg_max - 0.01)

def xywh2xyxy(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Converts bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format.

    This function takes bounding box coordinates in (x, y, width, height) format and converts them to (x1, y1, x2, y2) format,
    where (x1, y1) represents the top-left corner and (x2, y2) represents the bottom-right corner of the bounding box.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format. The input can be
            either a NumPy ndarray or a PyTorch Tensor. The shape is expected to be (..., 4), where the last dimension
            represents the bounding box coordinates.

    Returns:
        np.ndarray | torch.Tensor: The bounding box coordinates in (x1, y1, x2, y2) format. The output type matches the input type
            (NumPy ndarray or PyTorch Tensor) and has the same shape (..., 4).
    """

    # Create a copy of the input to avoid modifying the original data.
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)

    # Calculate the x-coordinate of the top-left corner (x1).
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x_center - width / 2

    # Calculate the y-coordinate of the top-left corner (y1).
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y_center - height / 2

    # Calculate the x-coordinate of the bottom-right corner (x2).
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x_center + width / 2

    # Calculate the y-coordinate of the bottom-right corner (y2).
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y_center + height / 2

    return y

class TaskAlignedAssigner(nn.Module):
    """
    Assigns ground truth objects to predicted anchor boxes using a task-aligned assignment strategy.

    This assigner combines predicted class scores and IoU overlaps to create an alignment metric,
    selecting the top-k candidates for each ground truth object. It handles cases with varying
    numbers of ground truth objects and provides an optional "rollout" strategy for specific scenarios.
    """

    def __init__(self, topk: int = 13, num_classes: int = 80, alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9, roll_out_thr: int = 0):
        """
        Initializes the TaskAlignedAssigner.

        Args:
            topk (int): Number of top candidates to consider for each ground truth object.
            num_classes (int): Number of object classes.
            alpha (float): Exponent for the predicted class scores in the alignment metric.
            beta (float): Exponent for the IoU overlaps in the alignment metric.
            eps (float): Small epsilon value for numerical stability.
            roll_out_thr (int): Threshold for the number of ground truth boxes to trigger the rollout strategy.
        """
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes  # Background class index
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.roll_out_thr = roll_out_thr

    @torch.no_grad()
    def forward(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        anc_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted class scores (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truths (bs, n_max_boxes, 1).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                Tuple containing target labels, target bboxes, target scores, foreground mask, and target ground truth indices.
        """
        bs = pd_scores.size(0)
        n_max_boxes = gt_bboxes.size(1)
        roll_out = n_max_boxes > self.roll_out_thr if self.roll_out_thr else False

        if n_max_boxes == 0:  # Handle cases with no ground truth objects
            device = gt_bboxes.device
            empty_tensor = torch.zeros_like(pd_scores[..., 0]).to(device)
            return (
                torch.full_like(pd_scores[..., 0], self.bg_idx).to(device),
                torch.zeros_like(pd_bboxes).to(device),
                torch.zeros_like(pd_scores).to(device),
                empty_tensor,
                empty_tensor,
            )

        mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt)
        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, n_max_boxes)
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        anc_points: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates the positive mask for anchor-ground truth assignment.
        """
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes, roll_out=self.roll_out)
        mask_topk = self.select_topk_candidates(align_metric * mask_in_gts, topk_mask=mask_gt.repeat([1, 1, self.topk]).bool())
        mask_pos = mask_topk * mask_in_gts * mask_gt
        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores: torch.Tensor, pd_bboxes: torch.Tensor, gt_labels: torch.Tensor, gt_bboxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the alignment metric and IoU overlaps.
        """
        bs = pd_scores.size(0)
        n_max_boxes = gt_bboxes.size(1)
        if self.roll_out:
            align_metric = torch.empty((bs, n_max_boxes, pd_scores.shape[1]), device=pd_scores.device)
            overlaps = torch.empty((bs, n_max_boxes, pd_scores.shape[1]), device=pd_scores.device)
            ind_0 = torch.empty(n_max_boxes, dtype=torch.long)
            for b in range(bs):
                ind_0[:], ind_2 = b, gt_labels[b].squeeze(-1).long()
                bbox_scores = pd_scores[ind_0, :, ind_2]
                overlaps[b] = bbox_iou(gt_bboxes[b].unsqueeze(1), pd_bboxes[b].unsqueeze(0), xywh=False, CIoU=True).squeeze(2).clamp(0)
                align_metric[b] = bbox_scores.pow(self.alpha) * overlaps[b].pow(self.beta)
        else:
            ind = torch.zeros([2, bs, n_max_boxes], dtype=torch.long)
            ind[0] = torch.arange(end=bs).view(-1, 1).repeat(1, n_max_boxes)
            ind[1] = gt_labels.long().squeeze(-1)
            bbox_scores = pd_scores[ind[0], :, ind[1]]
            overlaps = bbox_iou(gt_bboxes.unsqueeze(2), pd_bboxes.unsqueeze(1), xywh=False, CIoU=True).squeeze(3).clamp(0)
            align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def select_topk_candidates(self, metrics: torch.Tensor, largest: bool = True, topk_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Selects the top-k candidate anchors based on given metrics.

        This function selects the top-k candidates from the provided metrics tensor,
        considering an optional mask to filter out invalid candidates. It returns a
        tensor indicating which anchors are among the selected top-k.

        Args:
            metrics (torch.Tensor): The metric values for each anchor. Shape (b, max_num_obj, num_anchors).
            largest (bool): If True (default), selects the top-k largest values. If False, selects the top-k smallest values.
            topk_mask (torch.Tensor, optional): A mask to filter candidates. Shape (b, max_num_obj, topk) or None. Defaults to None.

        Returns:
            torch.Tensor: A tensor indicating which anchors are in the top-k. Shape (b, max_num_obj, num_anchors).
        """
        num_anchors = metrics.shape[-1]
        topk = min(self.topk, num_anchors) # handle case where num_anchors < topk
        topk_metrics, topk_idxs = torch.topk(metrics, topk, dim=-1, largest=largest)

        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True) > self.eps).tile([1, 1, topk])

        topk_idxs[~topk_mask] = 0

        if self.roll_out:
            is_in_topk = torch.zeros_like(metrics, dtype=torch.float32) #change to zeros and float32
            for b in range(len(topk_idxs)):
                is_in_topk[b] = F.one_hot(topk_idxs[b], num_anchors).sum(dim=-2).to(is_in_topk.dtype) #sum with dim
        else:
            is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(dim=-2).to(metrics.dtype) #sum with dim

        is_in_topk = torch.where(is_in_topk > 1, 0, is_in_topk)
        return is_in_topk
    
    def get_targets(self, gt_labels: torch.Tensor, gt_bboxes: torch.Tensor, target_gt_idx: torch.Tensor, fg_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates target labels, bounding boxes, and scores for assigned anchors.

        This function creates the training targets for the object detection model based on the
        assigned ground truth objects.

        Args:
            gt_labels (torch.Tensor): Ground truth labels (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes (bs, n_max_boxes, 4).
            target_gt_idx (torch.Tensor): Indices of assigned ground truth objects (bs, num_anchors).
            fg_mask (torch.Tensor): Foreground mask indicating assigned anchors (bs, num_anchors).

        Returns:
            tuple: A tuple containing:
                - target_labels (torch.Tensor): Assigned ground truth labels (bs, num_anchors).
                - target_bboxes (torch.Tensor): Assigned ground truth boxes (bs, num_anchors, 4).
                - target_scores (torch.Tensor): Target scores (bs, num_anchors, num_classes).
        """
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None] # add device
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes

        target_labels = gt_labels.long().flatten()[target_gt_idx]
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx]
        target_labels.clamp_(0, self.num_classes-1) #clamp labels to be in the range [0, num_classes-1]
        target_scores = F.one_hot(target_labels, self.num_classes)  # (b, h*w, num_classes)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes).bool()  # (b, h*w, num_classes)
        target_scores = torch.where(fg_scores_mask, target_scores, torch.zeros_like(target_scores)) #use torch.where and zeros_like

        return target_labels, target_bboxes, target_scores

class BboxLoss(nn.Module):
    """
    Computes bounding box loss, including IoU loss and Distribution Focal Loss (DFL).

    This class calculates the combined loss for bounding box regression, incorporating
    Intersection over Union (IoU) loss and, optionally, Distribution Focal Loss (DFL) for
    more precise box localization.
    """

    def __init__(self, reg_max: int = 16, use_dfl: bool = False):
        """
        Initializes the BboxLoss module.

        Args:
            reg_max (int): Maximum range for the distribution of bounding box offsets in DFL.
            use_dfl (bool): Whether to use Distribution Focal Loss (DFL).
        """
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist: torch.Tensor, pred_bboxes: torch.Tensor, anchor_points: torch.Tensor,
                target_bboxes: torch.Tensor, target_scores: torch.Tensor, target_scores_sum: torch.Tensor,
                fg_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the bounding box loss.

        Args:
            pred_dist (torch.Tensor): Predicted distribution of box offsets (b, num_anchors, 4*(reg_max + 1)).
            pred_bboxes (torch.Tensor): Predicted bounding boxes (b, num_anchors, 4).
            anchor_points (torch.Tensor): Anchor points (num_anchors, 2).
            target_bboxes (torch.Tensor): Target bounding boxes (b, num_anchors, 4).
            target_scores (torch.Tensor): Target scores (b, num_anchors, num_classes).
            target_scores_sum (torch.Tensor): Sum of target scores for normalization (scalar).
            fg_mask (torch.Tensor): Foreground mask indicating assigned anchors (b, num_anchors).

        Returns:
            tuple: A tuple containing:
                - loss_iou (torch.Tensor): IoU loss.
                - loss_dfl (torch.Tensor): Distribution Focal Loss (or 0 if not used).
        """
        weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max) #calculate target ltrb
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0, device=pred_dist.device) #set device

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the Distribution Focal Loss (DFL).

        This function calculates the DFL based on the predicted distribution and the target
        bounding box offsets.

        Args:
            pred_dist (torch.Tensor): Predicted distribution (n, reg_max + 1).
            target (torch.Tensor): Target bounding box offsets (n,).

        Returns:
            torch.Tensor: The DFL loss for each element in the batch (n, 1).
        """
        tl = target.long()  # Target left
        tr = tl + 1  # Target right
        wl = (tr - target).unsqueeze(-1)  # Weight left
        wr = (1 - (tr - target)).unsqueeze(-1) # Weight right
        loss = (F.cross_entropy(pred_dist, tl.view(-1), reduction="none").unsqueeze(-1) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction="none").unsqueeze(-1) * wr)
        return loss.mean(dim=-2, keepdim=True) #mean over regmax dim

# Criterion class for computing training losses
class Loss(nn.Module):
    """
    Computes the combined loss for object detection, including classification,
    bounding box regression (IoU), and optionally Distribution Focal Loss (DFL).
    """

    def __init__(self, model: BaseModel):
        """
        Initializes the Loss module.

        Args:
            model (nn.Module): The detection model containing necessary attributes
                like stride, num_classes, reg_max, and no.
        """
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.stride = model.head.stride
        self.nc = model.head.nc
        self.reg_max = model.head.ch
        self.use_dfl = self.reg_max > 1
        self.roll_out_thr = 64 

        self.assigner = TaskAlignedAssigner(
            topk=10, num_classes=self.nc, alpha=0.5, beta=6.0, roll_out_thr=self.roll_out_thr
        )
        self.bbox_loss = BboxLoss(self.reg_max - 1, use_dfl=self.use_dfl)
        self.proj = torch.arange(self.reg_max, dtype=torch.float)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """
        Preprocesses ground truth targets.

        Args:
            targets (torch.Tensor): Raw target tensor (n, 6) [image_id, class, xywh].
            batch_size (int): Batch size.
            scale_tensor (torch.Tensor): Tensor for scaling boxes.

        Returns:
            torch.Tensor: Preprocessed targets (batch_size, max_targets_per_image, 5) [class, xyxy].
        """
        if targets.numel() == 0:  # Use numel() for empty tensor check
            return torch.zeros(batch_size, 0, 5, device=targets.device)

        i = targets[:, 0].long()  # Image indices, make sure it's long
        _, counts = i.unique(return_counts=True)
        out = torch.zeros(batch_size, counts.max(), 5, device=targets.device)
        for j in range(batch_size):
            matches = i == j
            n = matches.sum()
            if n > 0:  # More explicit check
                out[j, :n] = targets[matches, 1:]
        out[..., 1:5] = xywh2xyxy(out[..., 1:5] * scale_tensor)  # Use * for element-wise multiplication
        return out

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """Decodes predicted distribution to bounding boxes."""
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.to(pred_dist.device).type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds: tuple, batch: torch.Tensor) -> torch.Tensor:
        """Computes the overall loss."""
        device = preds[1].device
        loss = torch.zeros(3, device=device)
        feats = preds[2] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        targets = torch.cat((batch[:, 0].view(-1, 1), batch[:, 1].view(-1, 1), batch[:, 2:]), 1)
        targets = self.preprocess(targets.to(device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt
        )

        target_bboxes /= stride_tensor
        target_scores_sum = max(target_scores.sum(), 1)  # Avoid division by zero

        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum() > 0: #Check if any fg_mask is True to avoid error
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                            target_scores_sum, fg_mask)

        loss[0] *= 7.5  # box gain
        loss[1] *= 0.5  # cls gain
        loss[2] *= 1.5  # dfl gain
        return loss.sum()


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model
    
def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)

class ModelEMA(object):
    """
    Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models

    Keeps a moving average of everything in the model state_dict (parameters and buffers).

    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999, tau: int = 2000, updates: int = 0):
        """
        Creates an EMA model.

        Args:
            model (torch.nn.Module): The model to track with EMA.
            decay (float, optional): EMA decay rate. Defaults to 0.9999.
            tau (int, optional): EMA update ramp-up timescale. Defaults to 2000.
            updates (int, optional): Number of EMA updates performed so far. Defaults to 0.
        """

        # Create EMA model with parameters in FP32
        self.ema = deepcopy(de_parallel(model)).eval()
        self.ema.float()  # Ensure EMA parameters are in FP32

        self.updates = updates
        self.decay = namedtuple('Decay', ['fn'])(lambda x: decay * (1 - math.exp(-x / tau)))  # Decay function with ramp-up

        # Freeze EMA parameters
        for param in self.ema.parameters():
            param.requires_grad = False

    def update(self, model: torch.nn.Module):
        """
        Updates the EMA model with the current model state.

        Args:
            model (torch.nn.Module): The model to update the EMA from.
        """

        with torch.no_grad():
            self.updates += 1
            decay = self.decay.fn(self.updates)

            # Get model state dict (de-parallelized)
            model_state_dict = de_parallel(model).state_dict()

            # Update EMA parameters
            for key, ema_val in self.ema.state_dict().items():
                if ema_val.dtype.is_floating_point:
                    ema_val *= decay
                    ema_val += (1.0 - decay) * model_state_dict[key].detach()

    def update_attr(self, model: torch.nn.Module, include: tuple = (), exclude: tuple = ('process_group', 'reducer')):
        """
        Updates EMA attributes from the model.

        Args:
            model (torch.nn.Module): The model to copy attributes from.
            include (tuple, optional): Attributes to explicitly include. Defaults to empty tuple.
            exclude (tuple, optional): Attributes to exclude. Defaults to ('process_group', 'reducer').
        """

        copy_attr(self.ema, model, include, exclude)

def weights_init(net: nn.Module, init_type: str = 'normal', init_gain: float = 0.02) -> None:
    """
    Initializes the weights of a PyTorch network.

    Args:
        net (torch.nn.Module): The network to initialize.
        init_type (str, optional): The weight initialization type. Defaults to 'normal'.
            - 'normal': Normal distribution initialization.
            - 'xavier': Xavier normal initialization.
            - 'kaiming': Kaiming normal initialization.
            - 'orthogonal': Orthogonal initialization.
        init_gain (float, optional): The gain factor for some initialization methods. Defaults to 0.02.
    """

    def init_fn(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                if classname.find('Conv') != -1:
                    init.xavier_normal_(m.weight.data, gain=init_gain)
                else:
                    init.xavier_uniform_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'Initialization method [{init_type}] is not implemented')
        elif classname.find('BatchNorm') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    print(f'Initializing network with {init_type} type')
    net.apply(init_fn)

def get_lr_scheduler(
    lr_decay_type: str,
    lr: float,
    min_lr: float,
    total_iters: int,
    warmup_iters_ratio: float = 0.05,
    warmup_lr_ratio: float = 0.1,
    no_aug_iter_ratio: float = 0.05,
    step_num: int = 10,
) -> Callable:
    """
    Gets a learning rate scheduler function based on the specified type.

    Args:
        lr_decay_type (str): The type of learning rate decay. Supported types are:
            - "cos": Cosine annealing with warmup and no-augmentation period.
            - other: Step decay.
        lr (float): Initial learning rate.
        min_lr (float): Minimum learning rate.
        total_iters (int): Total number of iterations.
        warmup_iters_ratio (float, optional): Ratio of warmup iterations to total iterations. Defaults to 0.05.
        warmup_lr_ratio (float, optional): Ratio of warmup learning rate to initial learning rate. Defaults to 0.1.
        no_aug_iter_ratio (float, optional): Ratio of no-augmentation iterations to total iterations. Defaults to 0.05.
        step_num (int, optional): Number of steps for step decay. Defaults to 10.

    Returns:
        Callable: A learning rate scheduler function that takes the current iteration as input and returns the learning rate.
        Raises ValueError if lr_decay_type is not supported.
    """

    def warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        """cosine annealing with warmup and no-aug period."""
        if iters <= warmup_total_iters:
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        """Step learning rate decay."""
        if step_size < 1:
            raise ValueError("step_size must be at least 1.")
        n = iters // step_size
        out_lr = lr * decay_rate**n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(int(warmup_iters_ratio * total_iters), 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(int(no_aug_iter_ratio * total_iters), 1), 15)
        return partial(warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    elif lr_decay_type == "step":
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1)) if step_num > 1 else 0.0 # handle case where step_num=1
        step_size = total_iters / step_num
        return partial(step_lr, lr, decay_rate, step_size)
    else:
        raise ValueError(f"Unsupported lr_decay_type: {lr_decay_type}")

def set_optimizer_lr(optimizer: optim.Optimizer, lr_scheduler_func: Callable, epoch: int) -> None:
    """
    Sets the learning rate of the optimizer using a scheduler function.

    Args:
        optimizer (optim.Optimizer): The optimizer whose learning rate should be updated.
        lr_scheduler_func (Callable): A function that takes the current epoch/iteration
            and returns the desired learning rate.
        epoch (int): The current epoch or iteration number.
    """
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
