import torch
from .model import FOCUS  # Import the FOCUS model

def load_yolov8_weights(model, yolov8_weights_path):
    """
    Load YOLOv8 pre-trained weights into the FOCUS model.

    Args:
        model (nn.Module): The FOCUS model.
        yolov8_weights_path (str): Path to YOLOv8 pre-trained weights.

    Returns:
        nn.Module: The FOCUS model with YOLOv8 weights loaded.
    """
    try:
        # Load YOLOv8 weights
        yolov8_weights = torch.load(yolov8_weights_path, map_location='cpu')

        # Initialize the FOCUS model
        model_dict = model.state_dict()

        # Filter out weights that don't match (e.g., YOLOv8 might have different layers)
        yolov8_dict = {k: v for k, v in yolov8_weights.items() if k in model_dict}

        # Check if any weights were loaded
        if len(yolov8_dict) == 0:
            raise ValueError("No matching weights found between YOLOv8 and FOCUS model.")

        # Update the FOCUS model with YOLOv8 weights
        model_dict.update(yolov8_dict)
        model.load_state_dict(model_dict)

        print("YOLOv8 weights loaded into FOCUS model.")
        return model
    except Exception as e:
        print(f"Error loading YOLOv8 weights: {e}")
        return None


# Path to YOLOv8 pre-trained weights
YOLOV8_WEIGHTS_PATH = './yolov8x.pt'

# Path to COCO dataset
COCO_DATASET_PATH = 'path_to_coco_dataset'

# Create different model sizes
focus_n = FOCUS(phi='n', classes_path='model_data/coco_classes.txt')
focus_s = FOCUS(phi='s', classes_path='model_data/coco_classes.txt')
focus_m = FOCUS(phi='m', classes_path='model_data/coco_classes.txt')
focus_l = FOCUS(phi='l', classes_path='model_data/coco_classes.txt')
focus_x = FOCUS(phi='x', classes_path='model_data/coco_classes.txt')

# Load YOLOv8 weights into each model
focus_n = load_yolov8_weights(focus_n.model, YOLOV8_WEIGHTS_PATH)
focus_s = load_yolov8_weights(focus_s.model, YOLOV8_WEIGHTS_PATH)
focus_m = load_yolov8_weights(focus_m.model, YOLOV8_WEIGHTS_PATH)
focus_l = load_yolov8_weights(focus_l.model, YOLOV8_WEIGHTS_PATH)
focus_x = load_yolov8_weights(focus_x.model, YOLOV8_WEIGHTS_PATH)

# Check if weights were loaded successfully
if focus_n is None or focus_s is None or focus_m is None or focus_l is None or focus_x is None:
    raise ValueError("Failed to load YOLOv8 weights into one or more models.")

# Fine-tune each model on your dataset
focus_n.fine_tune(dataset_path=COCO_DATASET_PATH, epochs=50, batch_size=16, lr=1e-4, save_dir='./runs/logs_n')
focus_s.fine_tune(dataset_path=COCO_DATASET_PATH, epochs=50, batch_size=16, lr=1e-4, save_dir='./runs/logs_s')
focus_m.fine_tune(dataset_path=COCO_DATASET_PATH, epochs=50, batch_size=16, lr=1e-4, save_dir='./runs/logs_m')
focus_l.fine_tune(dataset_path=COCO_DATASET_PATH, epochs=50, batch_size=16, lr=1e-4, save_dir='./runs/logs_l')
focus_x.fine_tune(dataset_path=COCO_DATASET_PATH, epochs=50, batch_size=16, lr=1e-4, save_dir='./runs/logs_x')

# Save the fine-tuned weights for each model
torch.save(focus_n.model.state_dict(), 'focus_n_pretrained_weights.pth')
torch.save(focus_s.model.state_dict(), 'focus_s_pretrained_weights.pth')
torch.save(focus_m.model.state_dict(), 'focus_m_pretrained_weights.pth')
torch.save(focus_l.model.state_dict(), 'focus_l_pretrained_weights.pth')
torch.save(focus_x.model.state_dict(), 'focus_x_pretrained_weights.pth')

print("Fine-tuned weights saved successfully.")