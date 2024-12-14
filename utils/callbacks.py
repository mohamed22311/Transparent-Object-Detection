import datetime
import os
from typing import List, Tuple

import torch
import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from .utils import cvtColor, preprocess_input, resize_image
from .bbox_utils import DecodeBox
from .utils_map import get_coco_map, get_map


class LossHistory:
    """
    A class for logging and plotting training and validation loss.

    Args:
        log_dir (str): Directory to save the logs and plots.
        model (torch.nn.Module): The model being trained (currently unused, but kept for potential future use).
        input_shape (Tuple[int, int]): The input shape of the model (currently unused).
    """

    def __init__(self, log_dir: str, model, input_shape: Tuple[int, int]):
        self.log_dir = log_dir
        self.losses = []
        self.val_loss = []

        os.makedirs(self.log_dir, exist_ok=True)  # Create directory if it doesn't exist
        self.writer = SummaryWriter(self.log_dir)

    def append_loss(self, epoch: int, loss: float, val_loss: float):
        """
        Appends the training and validation loss for the current epoch.

        Args:
            epoch (int): The current epoch number.
            loss (float): The training loss for the epoch.
            val_loss (float): The validation loss for the epoch.
        """

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss) + "\n")  # More concise writing
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss) + "\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        """Plots the training and validation loss curves."""
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()  # Clear the current axes
        plt.close("all") # Close all figures to prevent memory leaks


class EvalCallback:
    """Callback for evaluating the model and calculating mAP.

    Args:
        net (torch.nn.Module): The model to evaluate.
        input_shape (Tuple[int, int]): Input image shape (height, width).
        class_names (List[str]): List of class names.
        num_classes (int): Number of classes.
        val_lines (List[str]): List of validation annotation lines.
        log_dir (str): Log directory.
        cuda (bool): Whether to use CUDA.
        map_out_path (str, optional): Path to output mAP files. Defaults to ".temp_map_out".
        max_boxes (int, optional): Maximum number of boxes per image. Defaults to 100.
        confidence (float, optional): Confidence threshold. Defaults to 0.05.
        nms_iou (float, optional): NMS IoU threshold. Defaults to 0.5.
        letterbox_image (bool, optional): Whether to use letterboxing. Defaults to True.
        MINOVERLAP (float, optional): Minimum overlap for mAP calculation. Defaults to 0.5.
        eval_flag (bool, optional): Whether to perform evaluation. Defaults to True.
        period (int, optional): Evaluation period (epochs). Defaults to 1.
    """

    def __init__(self, net: torch.nn.Module, input_shape: Tuple[int, int], class_names: List[str], num_classes: int,
                 val_lines: List[str], log_dir: str, cuda: bool, map_out_path: str = ".temp_map_out", max_boxes: int = 100,
                 confidence: float = 0.05, nms_iou: float = 0.5, letterbox_image: bool = True, MINOVERLAP: float = 0.5,
                 eval_flag: bool = True, period: int = 1):

        self.net = net
        self.input_shape = input_shape
        self.class_names = class_names
        self.num_classes = num_classes
        self.val_lines = val_lines
        self.log_dir = log_dir
        self.cuda = cuda
        self.map_out_path = map_out_path
        self.max_boxes = max_boxes
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.letterbox_image = letterbox_image
        self.MINOVERLAP = MINOVERLAP
        self.eval_flag = eval_flag
        self.period = period

        self.bbox_util = DecodeBox(self.num_classes, (self.input_shape[0], self.input_shape[1]))

        self.maps = [0]
        self.epoches = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(0) + "\n")

    def get_map_txt(self, image_id: str, image: Image.Image, class_names: List[str], map_out_path: str):
        """Generates detection results in txt format for mAP calculation.

        Args:
            image_id (str): Image ID.
            image (PIL.Image): Image.
            class_names (List[str]): List of class names.
            map_out_path (str): Output path for mAP files.
        """
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w", encoding='utf-8')
        image_shape = np.array(np.shape(image)[0:2])

        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape,
                                                        image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                        nms_thres=self.nms_iou)

            if results[0] is None:
                return

            top_label = np.array(results[0][:, 5], dtype='int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]

        top_100 = np.argsort(top_conf)[::-1][:self.max_boxes]
        top_boxes = top_boxes[top_100]
        top_conf = top_conf[top_100]
        top_label = top_label[top_100]

        for i, c in enumerate(top_label):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        f.close()

    def on_epoch_end(self, epoch: int, model_eval: torch.nn.Module):
        """Performs evaluation at the end of each epoch (or every `period` epochs).

        Calculates and logs the mAP.

        Args:
            epoch (int): Current epoch number.
            model_eval (torch.nn.Module): Model to evaluate.
        """
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            os.makedirs(os.path.join(self.map_out_path, "ground-truth"), exist_ok=True)
            os.makedirs(os.path.join(self.map_out_path, "detection-results"), exist_ok=True)

            print("Get map.")
            for annotation_line in tqdm(self.val_lines, desc="Generating ground truth and detection results"):
                line = annotation_line.split()
                image_id = os.path.basename(line[0]).split('.')[0]
                image = Image.open(line[0])
                gt_boxes = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
                self.get_map_txt(image_id, image, self.class_names, self.map_out_path)

                with open(os.path.join(self.map_out_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
                    for box in gt_boxes:
                        left, top, right, bottom, obj = box
                        obj_name = self.class_names[obj]
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))

            print("Calculate Map.")
            try:
                temp_map = get_coco_map(class_names=self.class_names, path=self.map_out_path)[1]
            except Exception as e:
                print(f"Error calculating coco map, using voc map instead: {e}")
                temp_map = get_map(self.MINOVERLAP, False, path=self.map_out_path)
            self.maps.append(temp_map)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(temp_map) + "\n")

            plt.figure()
            plt.plot(self.epoches, self.maps, 'red', linewidth=2, label='train map')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Map %s' % str(self.MINOVERLAP))
            plt.title('A Map Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_map.png"))
            plt.cla()
            plt.close("all")

            print("Get map done.")
            shutil.rmtree(self.map_out_path)