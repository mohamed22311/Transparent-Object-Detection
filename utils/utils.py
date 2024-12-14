import random
import os

import numpy as np
import torch
from PIL import Image
from typing import Tuple, List

def cvtColor(image: Image.Image) -> Image.Image:
    """Converts an image to RGB format if necessary."""
    if image.mode == "RGB":
        return image
    else:
        return image.convert('RGB')

def resize_image(image: Image.Image, size: Tuple[int, int], letterbox_image: bool = False) -> Image.Image:
    """Resizes an image to the specified size."""
    iw, ih = image.size
    w, h = size
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

def get_classes(classes_path: str) -> Tuple[List[str], int]:
    """Reads class names from a text file."""
    try:
        with open(classes_path, encoding='utf-8') as f:
            class_names = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"Classes file not found at: {classes_path}")
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Gets the current learning rate from the optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return None # Return None if no param_groups are found

def seed_everything(seed: int = 11) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id: int, rank: int, seed: int) -> None:
    """Worker initialization function for DataLoaders."""
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def preprocess_input(image: np.ndarray) -> np.ndarray:
    """Normalizes an image by dividing by 255.0."""
    return image / 255.0

def show_config(**kwargs):
    """Prints a formatted configuration dictionary."""
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def download_weights(phi: str, model_dir: str = "./model_data") -> None:
    """Downloads pretrained weights."""
    from torch.hub import load_state_dict_from_url

    download_urls = {
        "n": 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_n_backbone_weights.pth',
        "s": 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_s_backbone_weights.pth',
        "m": 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_m_backbone_weights.pth',
        "l": 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_l_backbone_weights.pth',
        "x": 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_x_backbone_weights.pth',
    }
    if phi not in download_urls:
        raise ValueError(f"Invalid phi value: {phi}. Supported values are: {list(download_urls.keys())}")
    url = download_urls[phi]

    os.makedirs(model_dir, exist_ok=True) # use exist_ok=True to avoid error if directory exists
    try:
      load_state_dict_from_url(url, model_dir=model_dir)
    except Exception as e:
        print(f"Error downloading weights: {e}")