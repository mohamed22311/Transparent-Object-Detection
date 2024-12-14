import json
import os
from collections import defaultdict

#-------------------------------------------------------#
#   Paths to your dataset and annotations
#-------------------------------------------------------#
train_datasets_path     = "path_to_train_images"
val_datasets_path       = "path_to_val_images"
train_annotation_path   = "path_to_train_annotations.json"
val_annotation_path     = "path_to_val_annotations.json"

#-------------------------------------------------------#
#   Output paths for the generated txt files
#-------------------------------------------------------#
train_output_path       = "train.txt"
val_output_path         = "val.txt"

def map_category_id(category_id):
    """
    Map the category ID to a class index.
    
    Args:
        category_id (int): The original category ID from the dataset.
    
    Returns:
        int: The mapped class index.
    """
    # Define your own category mapping logic here
    # Example: Map COCO-like category IDs to class indices
    if 1 <= category_id <= 11:
        return category_id - 1
    elif 13 <= category_id <= 25:
        return category_id - 2
    elif 27 <= category_id <= 28:
        return category_id - 3
    elif 31 <= category_id <= 44:
        return category_id - 5
    elif 46 <= category_id <= 65:
        return category_id - 6
    elif category_id == 67:
        return category_id - 7
    elif category_id == 70:
        return category_id - 9
    elif 72 <= category_id <= 82:
        return category_id - 10
    elif 84 <= category_id <= 90:
        return category_id - 11
    else:
        raise ValueError(f"Unknown category ID: {category_id}")

def process_annotations(annotation_path, image_dir, output_path):
    """
    Process the annotation JSON file and generate a txt file for training or validation.
    
    Args:
        annotation_path (str): Path to the annotation JSON file.
        image_dir (str): Path to the directory containing the images.
        output_path (str): Path to save the generated txt file.
    """
    name_box_id = defaultdict(list)
    
    # Load the annotation JSON file
    try:
        with open(annotation_path, encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Annotation file not found at {annotation_path}")
        return
    
    # Extract annotations
    annotations = data.get('annotations', [])
    for ant in annotations:
        image_id = ant['image_id']
        image_name = os.path.join(image_dir, f"{image_id:012d}.jpg")  # Assuming image format is JPG
        category_id = ant['category_id']
        
        # Map category ID to class index
        try:
            class_index = map_category_id(category_id)
        except ValueError as e:
            print(f"Error: {e}. Skipping annotation.")
            continue
        
        # Extract bounding box coordinates
        bbox = ant['bbox']
        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height
        
        # Store the bounding box and class index
        name_box_id[image_name].append([x_min, y_min, x_max, y_max, class_index])
    
    # Write the output txt file
    with open(output_path, 'w') as f:
        for image_name, box_infos in name_box_id.items():
            f.write(image_name)
            for info in box_infos:
                x_min, y_min, x_max, y_max, class_index = info
                box_info = f" {x_min},{y_min},{x_max},{y_max},{class_index}"
                f.write(box_info)
            f.write('\n')

if __name__ == "__main__":
    # Process training annotations
    process_annotations(train_annotation_path, train_datasets_path, train_output_path)
    
    # Process validation annotations
    process_annotations(val_annotation_path, val_datasets_path, val_output_path)