import json
import os

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from utils.utils import cvtColor, preprocess_input, resize_image

#---------------------------------------------------------------------------#
#   map_mode specifies what the script should compute:
#   map_mode = 0: Full mAP calculation process (predict + evaluate).
#   map_mode = 1: Only generate prediction results.
#   map_mode = 2: Only evaluate the mAP.
#---------------------------------------------------------------------------#
map_mode = 0

#-------------------------------------------------------#
#   Paths to the validation dataset and annotations
#-------------------------------------------------------#
annotation_path = 'path_to_annotations.json'  # Path to the annotation file
dataset_img_path = 'path_to_images'           # Path to the image directory

#-------------------------------------------------------#
#   Output directory for saving results
#-------------------------------------------------------#
temp_save_path = 'map_out'

class mAP_FOCUS():
    """
    A class to evaluate FOCUS model performance using mAP.
    Inherits from the FOCUS class.
    """
    def detect_image(self, image_id, image, results, clsid2catid):
        """
        Detect objects in an image and store the results.

        Args:
            image_id (int): The ID of the image.
            image (PIL.Image): The input image.
            results (list): A list to store the detection results.
            clsid2catid (dict): A mapping from class IDs to category IDs.

        Returns:
            list: Updated results with detection information.
        """
        # Calculate input image height and width
        image_shape = np.array(np.shape(image)[0:2])

        # Convert image to RGB (required by the model)
        image = cvtColor(image)

        # Resize the image without distortion
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        # Add batch dimension
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            # Perform inference
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)

            # Apply non-maximum suppression (NMS)
            outputs = self.bbox_util.non_max_suppression(
                outputs, self.num_classes, self.input_shape, image_shape,
                self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou
            )

            if outputs[0] is None:
                return results

            # Extract detection results
            top_label = np.array(outputs[0][:, 5], dtype='int32')
            top_conf = outputs[0][:, 4]
            top_boxes = outputs[0][:, :4]

        # Format results for evaluation
        for i, c in enumerate(top_label):
            result = {}
            top, left, bottom, right = top_boxes[i]

            result["image_id"] = int(image_id)
            result["category_id"] = clsid2catid[c]
            result["bbox"] = [float(left), float(top), float(right - left), float(bottom - top)]
            result["score"] = float(top_conf[i])
            results.append(result)

        return results

def generate_predictions(model, cocoGt, dataset_img_path, temp_save_path):
    """
    Generate predictions for all images in the dataset.

    Args:
        model (mAP_FOCUS): The FOCUS model instance.
        cocoGt (COCO): The COCO object for ground truth annotations.
        dataset_img_path (str): Path to the image directory.
        temp_save_path (str): Path to save the prediction results.
    """
    ids = list(cocoGt.imgToAnns.keys())
    clsid2catid = cocoGt.getCatIds()

    results = []
    for image_id in tqdm(ids, desc="Generating Predictions"):
        image_path = os.path.join(dataset_img_path, cocoGt.loadImgs(image_id)[0]['file_name'])
        image = Image.open(image_path)
        results = model.detect_image(image_id, image, results, clsid2catid)

    # Save predictions to a JSON file
    with open(os.path.join(temp_save_path, 'eval_results.json'), "w") as f:
        json.dump(results, f)

def evaluate_map(cocoGt, temp_save_path):
    """
    Evaluate the mAP using COCO evaluation tools.

    Args:
        cocoGt (COCO): The COCO object for ground truth annotations.
        temp_save_path (str): Path to the directory containing prediction results.
    """
    # Load prediction results
    cocoDt = cocoGt.loadRes(os.path.join(temp_save_path, 'eval_results.json'))

    # Initialize COCO evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("mAP evaluation completed.")

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    if not os.path.exists(temp_save_path):
        os.makedirs(temp_save_path)

    # Load COCO ground truth annotations
    cocoGt = COCO(annotation_path)

    if map_mode == 0 or map_mode == 1:
        # Initialize FOCUS model for prediction
        model = mAP_FOCUS(confidence=0.001, nms_iou=0.65)

        # Generate predictions
        generate_predictions(model, cocoGt, dataset_img_path, temp_save_path)

    if map_mode == 0 or map_mode == 2:
        # Evaluate mAP
        evaluate_map(cocoGt, temp_save_path)