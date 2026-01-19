#!/usr/bin/env python3
"""
Convert YOLO format dataset to COCO format.
YOLO format: class_id cx cy w h (normalized)
COCO format: x_min y_min width height (absolute pixels)
"""

import os
import json
import argparse
import random
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def read_class_names(label_list_file):
    """Read class names from label_list.txt"""
    with open(label_list_file, 'r') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    return classes


def yolo_to_coco_bbox(yolo_bbox, img_width, img_height):
    """
    Convert YOLO bbox (normalized cxcywh) to COCO bbox (absolute xywh)
    
    Args:
        yolo_bbox: [cx, cy, w, h] normalized (0-1)
        img_width: image width in pixels
        img_height: image height in pixels
    
    Returns:
        [x, y, w, h] in absolute pixels
    """
    cx, cy, w, h = yolo_bbox
    
    # Convert normalized to absolute
    cx_abs = cx * img_width
    cy_abs = cy * img_height
    w_abs = w * img_width
    h_abs = h * img_height
    
    # Convert center to top-left
    x = cx_abs - w_abs / 2
    y = cy_abs - h_abs / 2
    
    # Ensure bbox is within image bounds
    x = max(0, min(x, img_width - 1))
    y = max(0, min(y, img_height - 1))
    w_abs = min(w_abs, img_width - x)
    h_abs = min(h_abs, img_height - y)
    
    return [x, y, w_abs, h_abs]


def parse_yolo_label(label_file):
    """Parse YOLO format label file"""
    annotations = []
    if not os.path.exists(label_file):
        return annotations
    
    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            
            annotations.append({
                'class_id': class_id,
                'bbox': [cx, cy, w, h]
            })
    
    return annotations


def convert_dataset(yolo_images_dir, yolo_labels_dir, label_list_file, output_dir, train_ratio=0.9, seed=42):
    """
    Convert YOLO format dataset to COCO format
    
    Args:
        yolo_images_dir: Directory containing YOLO format images
        yolo_labels_dir: Directory containing YOLO format labels
        label_list_file: Path to label_list.txt
        output_dir: Output directory for COCO format dataset
        train_ratio: Ratio of training data (default 0.9 for 90/10 split)
        seed: Random seed for train/val split
    """
    # Read class names
    classes = read_class_names(label_list_file)
    num_classes = len(classes)
    print(f"Found {num_classes} classes: {classes}")
    
    # Create output directories
    train_images_dir = os.path.join(output_dir, 'train', 'images')
    train_annotations_dir = os.path.join(output_dir, 'train', 'annotations')
    val_images_dir = os.path.join(output_dir, 'val', 'images')
    val_annotations_dir = os.path.join(output_dir, 'val', 'annotations')
    
    for dir_path in [train_images_dir, train_annotations_dir, val_images_dir, val_annotations_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Get all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        image_files.extend(Path(yolo_images_dir).glob(f'*{ext}'))
    
    image_files = sorted(image_files)
    print(f"Found {len(image_files)} images")
    
    # Shuffle and split
    random.seed(seed)
    random.shuffle(image_files)
    
    split_idx = int(len(image_files) * train_ratio)
    train_images = image_files[:split_idx]
    val_images = image_files[split_idx:]
    
    print(f"Train images: {len(train_images)}, Val images: {len(val_images)}")
    
    # Convert train set
    train_coco = {
        'images': [],
        'annotations': [],
        'categories': [{'id': i + 1, 'name': name} for i, name in enumerate(classes)]
    }
    
    ann_id = 1
    for img_path in tqdm(train_images, desc="Converting train set"):
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        # Get corresponding label file
        label_file = Path(yolo_labels_dir) / (img_path.stem + '.txt')
        yolo_annotations = parse_yolo_label(label_file)
        
        # Add image entry
        image_id = len(train_coco['images']) + 1
        train_coco['images'].append({
            'id': image_id,
            'file_name': img_path.name,
            'width': img_width,
            'height': img_height
        })
        
        # Add annotations
        for yolo_ann in yolo_annotations:
            class_id = yolo_ann['class_id']
            if class_id < 0 or class_id >= num_classes:
                print(f"Warning: Invalid class_id {class_id} in {label_file}, skipping")
                continue
            
            # Convert bbox
            coco_bbox = yolo_to_coco_bbox(yolo_ann['bbox'], img_width, img_height)
            
            # Calculate area
            area = coco_bbox[2] * coco_bbox[3]
            
            train_coco['annotations'].append({
                'id': ann_id,
                'image_id': image_id,
                'category_id': class_id + 1,  # COCO uses 1-indexed categories
                'bbox': coco_bbox,
                'area': area,
                'iscrowd': 0
            })
            ann_id += 1
        
        # Copy image to train directory
        shutil.copy2(img_path, os.path.join(train_images_dir, img_path.name))
    
    # Convert val set
    val_coco = {
        'images': [],
        'annotations': [],
        'categories': [{'id': i + 1, 'name': name} for i, name in enumerate(classes)]
    }
    
    ann_id = 1
    for img_path in tqdm(val_images, desc="Converting val set"):
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        # Get corresponding label file
        label_file = Path(yolo_labels_dir) / (img_path.stem + '.txt')
        yolo_annotations = parse_yolo_label(label_file)
        
        # Add image entry
        image_id = len(val_coco['images']) + 1
        val_coco['images'].append({
            'id': image_id,
            'file_name': img_path.name,
            'width': img_width,
            'height': img_height
        })
        
        # Add annotations
        for yolo_ann in yolo_annotations:
            class_id = yolo_ann['class_id']
            if class_id < 0 or class_id >= num_classes:
                print(f"Warning: Invalid class_id {class_id} in {label_file}, skipping")
                continue
            
            # Convert bbox
            coco_bbox = yolo_to_coco_bbox(yolo_ann['bbox'], img_width, img_height)
            
            # Calculate area
            area = coco_bbox[2] * coco_bbox[3]
            
            val_coco['annotations'].append({
                'id': ann_id,
                'image_id': image_id,
                'category_id': class_id + 1,  # COCO uses 1-indexed categories
                'bbox': coco_bbox,
                'area': area,
                'iscrowd': 0
            })
            ann_id += 1
        
        # Copy image to val directory
        shutil.copy2(img_path, os.path.join(val_images_dir, img_path.name))
    
    # Save COCO JSON files
    train_json_path = os.path.join(train_annotations_dir, 'instances_train.json')
    val_json_path = os.path.join(val_annotations_dir, 'instances_val.json')
    
    with open(train_json_path, 'w') as f:
        json.dump(train_coco, f, indent=2)
    
    with open(val_json_path, 'w') as f:
        json.dump(val_coco, f, indent=2)
    
    print(f"\nConversion complete!")
    print(f"Train annotations: {train_json_path}")
    print(f"Val annotations: {val_json_path}")
    print(f"Train images: {len(train_coco['images'])}")
    print(f"Val images: {len(val_coco['images'])}")
    print(f"Train annotations: {len(train_coco['annotations'])}")
    print(f"Val annotations: {len(val_coco['annotations'])}")


def main():
    parser = argparse.ArgumentParser(description='Convert YOLO format dataset to COCO format')
    parser.add_argument('--images-dir', type=str, required=True,
                        help='Directory containing YOLO format images')
    parser.add_argument('--labels-dir', type=str, required=True,
                        help='Directory containing YOLO format labels')
    parser.add_argument('--label-list', type=str, required=True,
                        help='Path to label_list.txt file')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for COCO format dataset')
    parser.add_argument('--train-ratio', type=float, default=0.9,
                        help='Ratio of training data (default: 0.9 for 90/10 split)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for train/val split (default: 42)')
    
    args = parser.parse_args()
    
    convert_dataset(
        args.images_dir,
        args.labels_dir,
        args.label_list,
        args.output_dir,
        args.train_ratio,
        args.seed
    )


if __name__ == '__main__':
    main()
