import cv2
import os
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils.utils import cvtColor, preprocess_input
from random import sample, shuffle
from typing import List, Tuple


class ModelDataset(Dataset):
    """Dataset for object detection."""

    def __init__(self, annotation_lines: List[str], input_shape: Tuple[int, int], num_classes: int, epoch_length: int,
                 mosaic: bool, mixup: bool, mosaic_prob: float, mixup_prob: float, train: bool, special_aug_ratio: float = 0.7):
        """
        Initializes the dataset.

        Args:
            annotation_lines (list): List of annotation lines (e.g., from a text file).
            input_shape (tuple): Input image shape (height, width).
            num_classes (int): Number of object classes.
            epoch_length (int): Length of an epoch.
            mosaic (bool): Whether to use mosaic augmentation.
            mixup (bool): Whether to use mixup augmentation.
            mosaic_prob (float): Probability of applying mosaic augmentation.
            mixup_prob (float): Probability of applying mixup augmentation.
            train (bool): Whether the dataset is used for training.
            special_aug_ratio (float, optional): Ratio of epochs to apply special augmentations. Defaults to 0.7.
        """
        super().__init__()
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epoch_length = epoch_length
        self.mosaic = mosaic
        self.mosaic_prob = mosaic_prob
        self.mixup = mixup
        self.mixup_prob = mixup_prob
        self.train = train
        self.special_aug_ratio = special_aug_ratio

        self.epoch_now = -1
        self.length = len(self.annotation_lines)
        self.bbox_attrs = 5 + num_classes

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        index = index % self.length

        # Apply mosaic augmentation
        if self.mosaic and self.rand() < self.mosaic_prob and self.epoch_now < self.epoch_length * self.special_aug_ratio:
            lines = sample(self.annotation_lines, 3)
            lines.append(self.annotation_lines[index])
            shuffle(lines)
            image, box = self.get_random_data_with_Mosaic(lines, self.input_shape)

            # Apply mixup augmentation
            if self.mixup and self.rand() < self.mixup_prob:
                lines = sample(self.annotation_lines, 1)
                image_2, box_2 = self.get_random_data(lines[0], self.input_shape, random=self.train)
                image, box = self.get_random_data_with_MixUp(image, box, image_2, box_2)
        else:
            # Regular data loading without mosaic/mixup
            image, box = self.get_random_data(self.annotation_lines[index], self.input_shape, random=self.train)

        # Preprocess image and bounding boxes
        image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box = np.array(box, dtype=np.float32)

        # Normalize bounding boxes
        nL = len(box)
        labels_out = np.zeros((nL, 6))
        if nL:
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
            labels_out[:, 1] = box[:, -1]
            labels_out[:, 2:] = box[:, :4]

        return image, labels_out

    def rand(self, a: float = 0, b: float = 1) -> float:
        """Generate a random float between a and b."""
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line: str, input_shape: Tuple[int, int], jitter: float = .3, hue: float = .1, sat: float = 0.7, val: float = 0.4, random: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gets and augments a single image and its bounding boxes.

        Args:
            annotation_line (str): Annotation line from the dataset file.
            input_shape (tuple): Input image shape (height, width).
            jitter (float, optional): Jittering factor for aspect ratio. Defaults to .3.
            hue (float, optional): Hue augmentation factor. Defaults to .1.
            sat (float, optional): Saturation augmentation factor. Defaults to 0.7.
            val (float, optional): Value augmentation factor. Defaults to 0.4.
            random (bool, optional): Whether to apply random augmentations. Defaults to True.

        Returns:
            tuple: Tuple containing the augmented image data and bounding boxes.
        """
        line = annotation_line.split()
        image = Image.open(line[0])
        image = cvtColor(image)
        iw, ih = image.size
        h, w = input_shape
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        if not random:
            # Non-random resizing and padding
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]

            return image_data, box

        # Random augmentations
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data = np.array(image, np.uint8)

        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, box

    def merge_bboxes(self, bboxes: List[np.ndarray], cutx: int, cuty: int) -> List[list]:
        """Merges bounding boxes from different parts of a mosaic image.

        Args:
            bboxes (list): List of bounding boxes, where each element is a NumPy array.
            cutx (int): X-coordinate of the cut.
            cuty (int): Y-coordinate of the cut.

        Returns:
            list: List of merged bounding boxes.
        """
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                tmp_box = [x1, y1, x2, y2, box[-1]]
                merge_bbox.append(tmp_box)
        return merge_bbox

    def get_random_data_with_Mosaic(self, annotation_line: List[str], input_shape: Tuple[int, int], jitter: float = 0.3, hue: float = .1, sat: float = 0.7, val: float = 0.4) -> Tuple[np.ndarray, list]:
        """
        Generates mosaic augmented data.

        Args:
            annotation_line (list): List of annotation lines.
            input_shape (tuple): Input image shape (height, width).
            jitter (float, optional): Jittering factor. Defaults to 0.3.
            hue (float, optional): Hue augmentation factor. Defaults to .1.
            sat (float, optional): Saturation augmentation factor. Defaults to 0.7.
            val (float, optional): Value augmentation factor. Defaults to 0.4.

        Returns:
            tuple: Tuple containing the mosaic image and merged bounding boxes.
        """
        h, w = input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)

        image_datas = []
        box_datas = []
        index = 0
        for line in annotation_line:
            line_content = line.split()
            image = Image.open(line_content[0])
            image = cvtColor(image)

            iw, ih = image.size
            box = np.array([np.array(list(map(int, box.split(',')))) for box in line_content[1:]])

            flip = self.rand() < .5
            if flip and len(box) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0, 2]] = iw - box[:, [2, 0]]

            new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
            scale = self.rand(.4, 1)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

            if index == 0:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y) - nh
            elif index == 1:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y)
            elif index == 2:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y)
            elif index == 3:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y) - nh

            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)

            index = index + 1
            box_data = []
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data = np.zeros((len(box), 5))
                box_data[:len(box)] = box

            image_datas.append(image_data)
            box_datas.append(box_data)

        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        new_image = np.array(new_image, np.uint8)

        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        dtype = new_image.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)

        return new_image, new_boxes

    def get_random_data_with_MixUp(self, image_1: np.ndarray, box_1: np.ndarray, image_2: np.ndarray, box_2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Mixes up two images and their bounding boxes.

        Args:
            image_1 (np.ndarray): First image.
            box_1 (np.ndarray): Bounding boxes of the first image.
            image_2 (np.ndarray): Second image.
            box_2 (np.ndarray): Bounding boxes of the second image.

        Returns:
            tuple: Mixed up image and bounding boxes.
        """
        new_image = np.array(image_1, np.float32) * 0.5 + np.array(image_2, np.float32) * 0.5
        if len(box_1) == 0:
            new_boxes = box_2
        elif len(box_2) == 0:
            new_boxes = box_1
        else:
            new_boxes = np.concatenate([box_1, box_2], axis=0)
        return new_image, new_boxes

def dataset_collate(batch: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collates a batch of images and bounding boxes.
    
    Args:
        batch (list): List of tuples containing images and bounding boxes.
        
    Returns:
        tuple: Tuple containing the batched images and bounding boxes.
    """
    images = []
    bboxes = []
    for i, (img, box) in enumerate(batch):
        images.append(img)
        box[:, 0] = i  # Add batch index
        bboxes.append(box)

    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.FloatTensor)
    return images, bboxes



class COCODataset(Dataset):
    def __init__(self, dataset_path, input_shape, train=True, transform=None):
        """
        Initialize the COCODataset.
        :param dataset_path: Path to the COCO dataset directory.
        :param input_shape: Input image shape (height, width).
        :param train: Whether the dataset is for training (default: True).
        :param transform: Optional transformations to apply to the images and annotations.
        """
        self.dataset_path = dataset_path
        self.input_shape = input_shape
        self.train = train
        self.transform = transform

        # Load COCO annotations
        annotation_file = os.path.join(dataset_path, "annotations.json")
        with open(annotation_file, "r") as f:
            self.coco_annotations = json.load(f)

        # Extract image and annotation information
        self.image_ids = [img['id'] for img in self.coco_annotations['images']]
        self.image_paths = {img['id']: os.path.join(dataset_path, img['file_name']) for img in self.coco_annotations['images']}
        self.annotations = self._parse_annotations()

    def _parse_annotations(self):
        """
        Parse the COCO annotations into a dictionary format.
        :return: A dictionary mapping image IDs to their annotations.
        """
        annotations = {img_id: [] for img_id in self.image_ids}
        for ann in self.coco_annotations['annotations']:
            img_id = ann['image_id']
            bbox = ann['bbox']  # [x_min, y_min, width, height]
            category_id = ann['category_id']
            annotations[img_id].append((bbox, category_id))
        return annotations

    def __len__(self):
        """
        Return the number of images in the dataset.
        """
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Get an image and its annotations by index.
        :param idx: Index of the image.
        :return: A tuple containing the image tensor and its annotations.
        """
        # Get image ID and path
        img_id = self.image_ids[idx]
        img_path = self.image_paths[img_id]

        # Load the image
        image = Image.open(img_path).convert("RGB")

        # Get annotations for the image
        bboxes, labels = [], []
        for bbox, category_id in self.annotations[img_id]:
            x_min, y_min, width, height = bbox
            x_max, y_max = x_min + width, y_min + height
            bboxes.append([x_min, y_min, x_max, y_max])
            labels.append(category_id)

        # Convert to numpy arrays
        bboxes = np.array(bboxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        # Apply transformations if provided
        if self.transform:
            image, bboxes, labels = self.transform(image, bboxes, labels)

        # Convert to tensors
        image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1)  # Convert to CHW format
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        return image, bboxes, labels

    def collate_fn(self, batch):
        """
        Custom collate function to handle variable-sized annotations.
        :param batch: A list of samples from the dataset.
        :return: A tuple containing batched images, bounding boxes, and labels.
        """
        images, bboxes, labels = zip(*batch)
        images = torch.stack(images, dim=0)
        return images, bboxes, labels