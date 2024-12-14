import numpy as np
import torch
from torchvision.ops import nms
from typing import Tuple, List, Optional
import pkg_resources as pkg

def check_version(current: str = "0.0.0", minimum: str = "0.0.0", name: str = "version ", pinned: bool = False) -> bool:
    """Checks if the current version meets the minimum requirement."""
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    return (current == minimum) if pinned else (current >= minimum)

TORCH_1_10 = check_version(torch.__version__, '1.10.0')

def make_anchors(feats: List[torch.Tensor], strides: List[int], grid_cell_offset: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate anchors from feature maps.

    Args:
        feats (List[torch.Tensor]): List of feature maps (tensors).
        strides (List[int]): List of corresponding strides for each feature map.
        grid_cell_offset (float, optional): Offset for the grid cell center. Defaults to 0.5.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing anchor points and stride tensors.
    """
    assert feats is not None, "Feature maps (feats) cannot be None."
    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device

    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(w, device=device, dtype=dtype) + grid_cell_offset  # Shift x
        sy = torch.arange(h, device=device, dtype=dtype) + grid_cell_offset  # Shift y
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))

    return torch.cat(anchor_points), torch.cat(stride_tensor)

def dist2bbox(distance: torch.Tensor, anchor_points: torch.Tensor, xywh: bool = True, dim: int = -1) -> torch.Tensor:
    """Transform distance (ltrb) to bounding box (xywh or xyxy).

    Args:
        distance (torch.Tensor): Distances from anchor points (left, top, right, bottom).
        anchor_points (torch.Tensor): Anchor points (x, y).
        xywh (bool, optional): If True, return boxes in xywh format; otherwise, xyxy. Defaults to True.
        dim (int, optional): Dimension along which to concatenate. Defaults to -1.

    Returns:
        torch.Tensor: Bounding boxes in the specified format.
    """
    lt, rb = torch.split(distance, 2, dim)  # Left-top, right-bottom
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

class DecodeBox:
    def __init__(self, num_classes: int, input_shape: Tuple[int, int]):
        self.num_classes = num_classes
        self.bbox_attrs = 4 + num_classes
        self.input_shape = input_shape

    def decode_box(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Decodes the box predictions.

        Args:
            inputs (Tuple): A tuple containing: dbox (distances), cls (class probabilities), origin_cls, anchors, and strides.

        Returns:
            torch.Tensor: Decoded box predictions.
        """
        dbox, cls, origin_cls, anchors, strides = inputs
        dbox = dist2bbox(dbox, anchors.unsqueeze(0), xywh=True, dim=1) * strides
        y = torch.cat((dbox, cls.sigmoid()), 1).permute(0, 2, 1)
        y[:, :, :4] = y[:, :, :4] / torch.tensor(
            [self.input_shape[1], self.input_shape[0], self.input_shape[1], self.input_shape[0]],
            dtype=y.dtype, device=y.device,
        )
        return y

    def correct_boxes(
            self, box_xy: np.ndarray, box_wh: np.ndarray, input_shape: Tuple[int, int],
            image_shape: Tuple[int, int], letterbox_image: bool
        ) -> np.ndarray:
        """Corrects bounding boxes for image resizing and letterboxing.

        Args:
            box_xy (np.ndarray): Box centers (x, y).
            box_wh (np.ndarray): Box widths and heights (w, h).
            input_shape (Tuple[int, int]): Input shape of the model.
            image_shape (Tuple[int, int]): Original image shape.
            letterbox_image (bool): Whether letterboxing was used.

        Returns:
            np.ndarray: Corrected bounding boxes (x1, y1, x2, y2).
        """
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            new_shape = np.round(image_shape * np.min(input_shape / image_shape))
            offset = (input_shape - new_shape) / 2.0 / input_shape
            scale = input_shape / new_shape

            box_yx = (box_yx - offset) * scale
            box_hw *= scale

        box_mins = box_yx - (box_hw / 2.0)
        box_maxes = box_yx + (box_hw / 2.0)
        boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes
    
    def non_max_suppression(
        self,
        prediction: torch.Tensor,
        num_classes: int,
        input_shape: Tuple[int, int],
        image_shape: Tuple[int, int],
        letterbox_image: bool,
        conf_thres: float = 0.5,
        nms_thres: float = 0.4,
    ) -> Optional[List[np.ndarray]]:
        """Performs Non-Maximum Suppression (NMS) on the predictions.

        Args:
            prediction (torch.Tensor): Raw predictions from the model.
            num_classes (int): Number of classes.
            input_shape (Tuple[int, int]): Input shape of the model.
            image_shape (Tuple[int, int]): Original image shape.
            letterbox_image (bool): Whether letterboxing was used.
            conf_thres (float, optional): Confidence threshold. Defaults to 0.5.
            nms_thres (float, optional): NMS IoU threshold. Defaults to 0.4.

        Returns:
            Optional[List[np.ndarray]]: A list of numpy arrays, where each array contains the detected boxes for an image.
                Returns None if no detections are found.
        """
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output: List[Optional[np.ndarray]] = [None] * len(prediction)  # Initialize output list correctly
        for i, image_pred in enumerate(prediction):
            class_conf, class_pred = torch.max(image_pred[:, 4 : 4 + num_classes], 1, keepdim=True)
            conf_mask = (class_conf[:, 0] >= conf_thres).squeeze()

            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]

            if not image_pred.size(0):
                continue

            detections = torch.cat((image_pred[:, :4], class_conf.float(), class_pred.float()), 1)
            unique_labels = detections[:, -1].unique() # no need to move to cpu first

            if prediction.is_cuda:
                detections = detections.cuda()
                unique_labels = unique_labels.cuda()

            for c in unique_labels:
                detections_class = detections[detections[:, -1] == c]
                keep = nms(detections_class[:, :4], detections_class[:, 4], nms_thres)
                max_detections = detections_class[keep]

                output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

            if output[i] is not None:
                output[i] = output[i].cpu().numpy()
                box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]
                output[i][:, :4] = self.correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)

        return output