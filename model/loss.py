import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import dist2bbox, make_anchors
from .base import BaseModel


def select_candidates_in_gts(xy_centers: torch.Tensor, gt_bboxes: torch.Tensor, eps: float = 1e-9, roll_out: bool = False) -> torch.Tensor:
    """
    Selects candidate anchor centers within ground truth bounding boxes.

    Args:
        xy_centers (torch.Tensor): Anchor center coordinates (h*w, 4).
        gt_bboxes (torch.Tensor): Ground truth bounding boxes (b, n_boxes, 4).
        eps (float): Small epsilon for numerical stability.
        roll_out (bool): Whether to use a loop for computation.

    Returns:
        torch.Tensor: Boolean mask indicating candidate anchor centers.
    """
    n_anchors = xy_centers.shape[0]
    bs, n_boxes, _ = gt_bboxes.shape

    if roll_out:
        bbox_deltas = torch.empty((bs, n_boxes, n_anchors), device=gt_bboxes.device)
        for b in range(bs):
            lt, rb = gt_bboxes[b].view(-1, 1, 4).chunk(2, 2)
            bbox_deltas[b] = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(n_boxes, n_anchors, -1).amin(2).gt_(eps)
        return bbox_deltas
    else:
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        return bbox_deltas.amin(3).gt_(eps)

def select_highest_overlaps(mask_pos: torch.Tensor, overlaps: torch.Tensor, n_max_boxes: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Selects the ground truth box with the highest IoU for each anchor.

    Args:
        mask_pos (torch.Tensor): Boolean mask indicating positive anchors.
        overlaps (torch.Tensor): IoU values between anchors and ground truths.
        n_max_boxes (int): Maximum number of ground truth boxes.

    Returns:
        tuple: Updated target ground truth indices, foreground mask, and positive mask.
    """
    fg_mask = mask_pos.sum(-2)
    if fg_mask.max() > 1:
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, n_max_boxes, 1])
        max_overlaps_idx = overlaps.argmax(1)
        is_max_overlaps = F.one_hot(max_overlaps_idx, n_max_boxes).permute(0, 2, 1).to(overlaps.dtype)
        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
        fg_mask = mask_pos.sum(-2)
    target_gt_idx = mask_pos.argmax(-2)
    return target_gt_idx, fg_mask, mask_pos

def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, xywh: bool = True, GIoU: bool = False, DIoU: bool = False, CIoU: bool = False, eps: float = 1e-7) -> torch.Tensor:
    """
    Calculates IoU and related metrics between two sets of bounding boxes.

    Args:
        box1 (torch.Tensor): First set of bounding boxes.
        box2 (torch.Tensor): Second set of bounding boxes.
        xywh (bool): Whether boxes are in (x, y, w, h) format.
        GIoU (bool): Whether to calculate Generalized IoU.
        DIoU (bool): Whether to calculate Distance IoU.
        CIoU (bool): Whether to calculate Complete IoU.
        eps (float): Small epsilon for numerical stability.

    Returns:
        torch.Tensor: IoU or related metric values.
    """
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_half, h1_half, w2_half, h2_half = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_half, x1 + w1_half, y1 - h1_half, y1 + h1_half
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_half, x2 + w2_half, y2 - h2_half, y2 + h2_half
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
        if CIoU or DIoU:
            c2 = cw ** 2 + ch ** 2 + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            if CIoU:
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
            return iou - rho2 / c2
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area
    return iou

def bbox2dist(anchor_points: torch.Tensor, bbox: torch.Tensor, reg_max: float) -> torch.Tensor:
    """
    Transforms bounding boxes into distances from anchor points.

    Args:
        anchor_points (torch.Tensor): Anchor points (N, 2).
        bbox (torch.Tensor): Bounding boxes (N, 4).
        reg_max (float): Maximum distance value.

    Returns:
        torch.Tensor: Distances (ltrb) from anchor points to bounding box edges.
    """
    x1y1, x2y2 = torch.split(bbox, 2, dim=-1)
    ltrb = torch.cat((anchor_points - x1y1, x2y2 - anchor_points), dim=-1)
    return ltrb.clamp(0, reg_max - 0.01)

def xywh2xyxy(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Converts bounding boxes from (x, y, w, h) to (x1, y1, x2, y2).

    Args:
        x (np.ndarray | torch.Tensor): Input bounding boxes.

    Returns:
        np.ndarray | torch.Tensor: Converted bounding boxes.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

class TaskAlignedAssigner(nn.Module):
    """
    Assigns ground truth objects to predicted anchor boxes using a task-aligned strategy.
    """

    def __init__(self, topk: int = 13, num_classes: int = 80, alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9, roll_out_thr: int = 0):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
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
        bs = pd_scores.size(0)
        n_max_boxes = gt_bboxes.size(1)
        roll_out = n_max_boxes > self.roll_out_thr if self.roll_out_thr else False

        if n_max_boxes == 0:
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
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes, roll_out=self.roll_out)
        mask_topk = self.select_topk_candidates(align_metric * mask_in_gts, topk_mask=mask_gt.repeat([1, 1, self.topk]).bool())
        mask_pos = mask_topk * mask_in_gts * mask_gt
        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores: torch.Tensor, pd_bboxes: torch.Tensor, gt_labels: torch.Tensor, gt_bboxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        num_anchors = metrics.shape[-1]
        topk = min(self.topk, num_anchors)
        topk_metrics, topk_idxs = torch.topk(metrics, topk, dim=-1, largest=largest)

        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True) > self.eps).tile([1, 1, topk])

        topk_idxs[~topk_mask] = 0

        if self.roll_out:
            is_in_topk = torch.zeros_like(metrics, dtype=torch.float32)
            for b in range(len(topk_idxs)):
                is_in_topk[b] = F.one_hot(topk_idxs[b], num_anchors).sum(dim=-2).to(is_in_topk.dtype)
        else:
            is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(dim=-2).to(metrics.dtype)

        is_in_topk = torch.where(is_in_topk > 1, 0, is_in_topk)
        return is_in_topk
    
    def get_targets(self, gt_labels: torch.Tensor, gt_bboxes: torch.Tensor, target_gt_idx: torch.Tensor, fg_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes

        target_labels = gt_labels.long().flatten()[target_gt_idx]
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx]
        target_labels.clamp_(0, self.num_classes-1)
        target_scores = F.one_hot(target_labels, self.num_classes)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes).bool()
        target_scores = torch.where(fg_scores_mask, target_scores, torch.zeros_like(target_scores))

        return target_labels, target_bboxes, target_scores

class BboxLoss(nn.Module):
    """
    Computes bounding box loss, including IoU loss and Distribution Focal Loss (DFL).
    """

    def __init__(self, reg_max: int = 16, use_dfl: bool = False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist: torch.Tensor, pred_bboxes: torch.Tensor, anchor_points: torch.Tensor,
                target_bboxes: torch.Tensor, target_scores: torch.Tensor, target_scores_sum: torch.Tensor,
                fg_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        tl = target.long()
        tr = tl + 1
        wl = (tr - target).unsqueeze(-1)
        wr = (1 - (tr - target)).unsqueeze(-1)
        loss = (F.cross_entropy(pred_dist, tl.view(-1), reduction="none").unsqueeze(-1) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction="none").unsqueeze(-1) * wr)
        return loss.mean(dim=-2, keepdim=True)

class Loss(nn.Module):
    """
    Computes the combined loss for object detection.
    """

    def __init__(self, model: BaseModel):
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
        if targets.numel() == 0:
            return torch.zeros(batch_size, 0, 5, device=targets.device)

        i = targets[:, 0].long()
        _, counts = i.unique(return_counts=True)
        out = torch.zeros(batch_size, counts.max(), 5, device=targets.device)
        for j in range(batch_size):
            matches = i == j
            n = matches.sum()
            if n > 0:
                out[j, :n] = targets[matches, 1:]
        out[..., 1:5] = xywh2xyxy(out[..., 1:5] * scale_tensor)
        return out

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.to(pred_dist.device).type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds: tuple, batch: torch.Tensor) -> torch.Tensor:
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
        target_scores_sum = max(target_scores.sum(), 1)

        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum() > 0:
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                            target_scores_sum, fg_mask)

        loss[0] *= 7.5
        loss[1] *= 0.5
        loss[2] *= 1.5
        return loss.sum()
