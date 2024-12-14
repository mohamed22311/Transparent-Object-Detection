import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.cuda.amp import GradScaler
from PIL import Image
from torchvision import transforms
import cv2
from tqdm import tqdm

from model import BaseModel, Loss, ModelEMA
from utils.dataloader import ModelDataset, dataset_collate
from utils.callbacks import LossHistory, EvalCallback
from utils.utils import get_classes, show_config, train_one_epoch


class FOCUS:
    def __init__(self, phi='s', model_path=None, classes_path='model_data/coco_classes.txt', input_shape=(640, 640), cuda=True):
        """
        Initialize the FOCUS model.
        Args:
            phi (str): Model size identifier ('n', 's', 'm', 'l', 'x').
            model_path (str): Path to pre-trained weights (optional).
            classes_path (str): Path to class names file.
            input_shape (tuple): Input image shape (height, width).
            cuda (bool): Use CUDA if available.
        """
        self.phi = phi
        self.model_path = model_path
        self.classes_path = classes_path
        self.input_shape = input_shape
        self.cuda = cuda and torch.cuda.is_available()

        # Load class names
        self.class_names, self.num_classes = get_classes(classes_path)

        # Initialize the model
        self.model = self._create_model()

        # Load pre-trained weights if provided
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"Loaded pre-trained weights from {model_path}")

        # Move model to GPU if available
        if self.cuda:
            self.model = self.model.cuda()

        # Define loss function
        self.loss_fn = Loss(self.model)

        # Define EMA (Exponential Moving Average) for model weights
        self.ema = ModelEMA(self.model) if self.cuda else None

        # Display configuration
        show_config(phi=self.phi, model_path=self.model_path, classes_path=self.classes_path, input_shape=self.input_shape, cuda=self.cuda)

    def _create_model(self):
        """
        Create the FOCUS model based on the specified size.
        """
        # Define base channels, depth, and depth multiplier based on phi
        if self.phi == 'n':
            base_channels, base_depth, deep_mul = 64, 1, 0.33
        elif self.phi == 's':
            base_channels, base_depth, deep_mul = 128, 2, 0.5
        elif self.phi == 'm':
            base_channels, base_depth, deep_mul = 256, 3, 0.67
        elif self.phi == 'l':
            base_channels, base_depth, deep_mul = 512, 4, 1.0
        elif self.phi == 'x':
            base_channels, base_depth, deep_mul = 1024, 5, 1.25
        else:
            raise ValueError(f"Invalid model size: {self.phi}. Choose from 'n', 's', 'm', 'l', 'x'.")

        # Create the model
        return BaseModel(num_classes=self.num_classes, base_channels=base_channels, base_depth=base_depth, deep_mul=deep_mul)

    def train(self, dataset_path, epochs=100, batch_size=16, lr=1e-3, save_dir='./logs'):
        """
        Train the model on a dataset.
        Args:
            dataset_path (str): Path to the dataset in COCO format.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            lr (float): Learning rate.
            save_dir (str): Directory to save logs and weights.
        """
        print("Training the model...")

        # Load dataset
        train_dataset = ModelDataset(
            annotation_lines=self._load_annotations(dataset_path, 'train'),
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            epoch_length=epochs,
            mosaic=True,
            mixup=True,
            mosaic_prob=0.5,
            mixup_prob=0.5,
            train=True
        )
        val_dataset = ModelDataset(
            annotation_lines=self._load_annotations(dataset_path, 'val'),
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            epoch_length=epochs,
            mosaic=False,
            mixup=False,
            mosaic_prob=0.0,
            mixup_prob=0.0,
            train=False
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset_collate)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset_collate)

        # Define optimizer and loss
        optimizer = Adam(self.model.parameters(), lr=lr)
        scaler = GradScaler()
        loss_history = LossHistory(log_dir=save_dir, model=self.model, input_shape=self.input_shape)
        eval_callback = EvalCallback(
            net=self.model,
            input_shape=self.input_shape,
            class_names=self.class_names,
            num_classes=self.num_classes,
            val_lines=self._load_annotations(dataset_path, 'val'),
            log_dir=save_dir,
            cuda=self.cuda
        )

        # Training loop
        for epoch in range(epochs):
            train_one_epoch(
                model_train=self.model,
                model=self.ema,
                ema=self.ema,
                model_loss=self.loss_fn,
                loss_history=loss_history,
                eval_callback=eval_callback,
                optimizer=optimizer,
                epoch=epoch,
                epoch_step=len(train_loader),
                epoch_step_val=len(val_loader),
                gen=train_loader,
                gen_val=val_loader,
                epochs=epochs,
                cuda=self.cuda,
                fp16=True,
                scaler=scaler,
                save_period=10,
                save_dir=save_dir
            )

    def fine_tune(self, dataset_path, epochs=50, batch_size=16, lr=1e-4, save_dir='./logs'):
        """
        Fine-tune the model on a dataset.
        Args:
            dataset_path (str): Path to the dataset in YOLO format.
            epochs (int): Number of fine-tuning epochs.
            batch_size (int): Batch size for fine-tuning.
            lr (float): Learning rate.
            save_dir (str): Directory to save logs and weights.
        """
        print("Fine-tuning the model...")
        self.train(dataset_path, epochs, batch_size, lr, save_dir)

    def predict(self, source):
        """
        Predict objects in an image or video.
        Args:
            source (str): Path to the image or video file.
        """
        if source.endswith(('.jpg', '.png', '.jpeg')):
            return self.predict_image(source)
        elif source.endswith(('.mp4', '.avi')):
            return self.predict_video(source)
        else:
            raise ValueError("Unsupported file format. Use .jpg, .png, .jpeg for images or .mp4, .avi for videos.")

    def predict_image(self, image_path):
        """
        Predict objects in a single image using the model.
        Args:
            image_path (str): Path to the input image.
        Returns:
            dict: A dictionary containing the predictions with keys:
                - 'boxes': List of bounding boxes (x1, y1, x2, y2).
                - 'labels': List of class labels.
                - 'scores': List of confidence scores.
        """
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image_tensor = transforms.ToTensor()(image).unsqueeze(0)  # Add batch dimension
        image_tensor = torch.nn.functional.interpolate(image_tensor, size=self.input_shape, mode='bilinear', align_corners=False)

        # Move to GPU if available
        if self.cuda:
            image_tensor = image_tensor.cuda()
            self.model = self.model.cuda()

        # Set the model to evaluation mode
        self.model.eval()

        # Perform inference
        with torch.no_grad():
            predictions = self.model(image_tensor)

        # Post-process the predictions
        boxes = predictions['boxes'].cpu().numpy()  # Bounding boxes (x1, y1, x2, y2)
        labels = predictions['labels'].cpu().numpy()  # Class labels
        scores = predictions['scores'].cpu().numpy()  # Confidence scores

        # Filter predictions with confidence above a threshold (e.g., 0.5)
        threshold = 0.5
        filtered_indices = scores > threshold
        filtered_boxes = boxes[filtered_indices]
        filtered_labels = labels[filtered_indices]
        filtered_scores = scores[filtered_indices]

        # Return the predictions as a dictionary
        return {
            'boxes': filtered_boxes,
            'labels': filtered_labels,
            'scores': filtered_scores
        }

    def predict_video(self, video_path, output_path=None):
        """
        Predict objects in a video using the model.
        Args:
            video_path (str): Path to the input video file.
            output_path (str): Path to save the output video with predictions (optional).
        Returns:
            None
        """
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Define the codec and create a VideoWriter object if output_path is provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Set the model to evaluation mode
        self.model.eval()

        frame_count = 0
        while True:
            # Read the next frame
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess the frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = transforms.ToTensor()(frame_rgb).unsqueeze(0)  # Add batch dimension
            frame_tensor = torch.nn.functional.interpolate(frame_tensor, size=self.input_shape, mode='bilinear', align_corners=False)

            # Move to GPU if available
            if self.cuda:
                frame_tensor = frame_tensor.cuda()
                self.model = self.model.cuda()

            # Perform inference
            with torch.no_grad():
                predictions = self.model(frame_tensor)

            # Post-process the predictions
            boxes = predictions['boxes'].cpu().numpy()  # Bounding boxes (x1, y1, x2, y2)
            labels = predictions['labels'].cpu().numpy()  # Class labels
            scores = predictions['scores'].cpu().numpy()  # Confidence scores

            # Filter predictions with confidence above a threshold (e.g., 0.5)
            threshold = 0.5
            filtered_indices = scores > threshold
            filtered_boxes = boxes[filtered_indices]
            filtered_labels = labels[filtered_indices]
            filtered_scores = scores[filtered_indices]

            # Draw bounding boxes on the frame
            for box, label, score in zip(filtered_boxes, filtered_labels, filtered_scores):
                x1, y1, x2, y2 = box.astype(int)
                class_name = str(label)  # Replace with actual class names if available
                confidence = score

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add label and confidence
                label_text = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Write the frame to the output video if output_path is provided
            if output_path:
                out.write(frame)

            # Display progress
            frame_count += 1
            print(f"Processed frame {frame_count}/{total_frames}", end="\r")

        # Release resources
        cap.release()
        if output_path:
            out.release()
        print("\nVideo processing complete.")

    def export_onnx(self, output_path):
        """
        Export the model to ONNX format.
        Args:
            output_path (str): Path to save the ONNX model.
        """
        print("Exporting the model to ONNX...")
        torch.onnx.export(self.model, torch.zeros(1, 3, *self.input_shape).cuda() if self.cuda else torch.zeros(1, 3, *self.input_shape), output_path)
        print(f"Model exported to {output_path}")

    def _load_annotations(self, dataset_path, split):
        """
        Load annotations from the dataset.
        Args:
            dataset_path (str): Path to the dataset.
            split (str): Dataset split ('train', 'val').
        Returns:
            list: List of annotation lines.
        """
        annotation_file = os.path.join(dataset_path, f'{split}.txt')
        with open(annotation_file, 'r') as f:
            return f.readlines()