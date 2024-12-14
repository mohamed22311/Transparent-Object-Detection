import numpy as np
import torch
from torchvision.ops import nms
import pkg_resources as pkg
from typing import Tuple, List, Optional

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
        #Use 'ij' indexing consistently regardless of torch version
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
        super().__init__()
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
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from PIL import Image

    def get_anchors_and_decode(input, input_shape, anchors, anchors_mask, num_classes):
        """Decodes predictions and generates anchors."""
        batch_size = input.size(0)
        input_height = input.size(2)
        input_width = input.size(3)

        stride_h = input_shape[0] / input_height
        stride_w = input_shape[1] / input_width
        scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in anchors[anchors_mask[2]]]

        prediction = input.view(batch_size, len(anchors_mask[2]), num_classes + 5, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = torch.sigmoid(prediction[..., 2])
        h = torch.sigmoid(prediction[..., 3])
        conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
            batch_size * len(anchors_mask[2]), 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
            batch_size * len(anchors_mask[2]), 1, 1).view(y.shape).type(FloatTensor)

        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data * 2. - 0.5 + grid_x
        pred_boxes[..., 1] = y.data * 2. - 0.5 + grid_y
        pred_boxes[..., 2] = (w.data * 2) ** 2 * anchor_w
        pred_boxes[..., 3] = (h.data * 2) ** 2 * anchor_h

        point_h = 5
        point_w = 5

        box_xy = pred_boxes[..., 0:2].cpu().numpy() * 32
        box_wh = pred_boxes[..., 2:4].cpu().numpy() * 32
        grid_x = grid_x.cpu().numpy() * 32
        grid_y = grid_y.cpu().numpy() * 32
        anchor_w = anchor_w.cpu().numpy() * 32
        anchor_h = anchor_h.cpu().numpy() * 32

        fig = plt.figure()
        ax = fig.add_subplot(121)
        try:
            img = Image.open("img/street.jpg").resize([640, 640])
        except FileNotFoundError:
            print("Error: 'img/street.jpg' not found. Please provide a valid image.")
            return
        plt.imshow(img, alpha=0.5)
        plt.ylim(-30, 650)
        plt.xlim(-30, 650)
        plt.scatter(grid_x, grid_y)
        plt.scatter(point_h * 32, point_w * 32, c='black')
        plt.gca().invert_yaxis()

        anchor_left = grid_x - anchor_w / 2
        anchor_top = grid_y - anchor_h / 2

        rect1 = plt.Rectangle([anchor_left[0, 0, point_h, point_w],anchor_top[0, 0, point_h, point_w]],
            anchor_w[0, 0, point_h, point_w],anchor_h[0, 0, point_h, point_w],color="r",fill=False)
        rect2 = plt.Rectangle([anchor_left[0, 1, point_h, point_w],anchor_top[0, 1, point_h, point_w]],
            anchor_w[0, 1, point_h, point_w],anchor_h[0, 1, point_h, point_w],color="r",fill=False)
        rect3 = plt.Rectangle([anchor_left[0, 2, point_h, point_w],anchor_top[0, 2, point_h, point_w]],
            anchor_w[0, 2, point_h, point_w],anchor_h[0, 2, point_h, point_w],color="r",fill=False)

        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)

        ax = fig.add_subplot(122)
        plt.imshow(img, alpha=0.5)
        plt.ylim(-30, 650)
        plt.xlim(-30, 650)
        plt.scatter(grid_x, grid_y)
        plt.scatter(point_h * 32, point_w * 32, c='black')
        plt.scatter(box_xy[0, :, point_h, point_w, 0], box_xy[0, :, point_h, point_w, 1], c='r')
        plt.gca().invert_yaxis()

        pre_left = box_xy[...,0] - box_wh[...,0] / 2
        pre_top = box_xy[...,1] - box_wh[...,1] / 2

        rect1 = plt.Rectangle([pre_left[0, 0, point_h, point_w], pre_top[0, 0, point_h, point_w]],
            box_wh[0, 0, point_h, point_w,0], box_wh[0, 0, point_h, point_w,1],color="r",fill=False)
        rect2 = plt.Rectangle([pre_left[0, 1, point_h, point_w], pre_top[0, 1, point_h, point_w]],
            box_wh[0, 1, point_h, point_w,0], box_wh[0, 1, point_h, point_w,1],color="r",fill=False)
        rect3 = plt.Rectangle([pre_left[0, 2, point_h, point_w], pre_top[0, 2, point_h, point_w]],
            box_wh[0, 2, point_h, point_w,0], box_wh[0, 2, point_h, point_w,1],color="r",fill=False)

        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)

        plt.show()

    feat = torch.from_numpy(np.random.normal(0.2, 0.5, [4, 255, 20, 20])).float()
    anchors = np.array([[116, 90], [156, 198], [373, 326], [30,61], [62,45], [59,119], [10,13], [16,30], [33,23]])
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]