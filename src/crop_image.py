"""
Crop text regions from synthetic invoice images based on polygon annotations.

This script reads synthetic invoice data (images + JSON labels) and extracts
cropped text regions for training the recognition model (PARSeq).

Usage:
    python src/crop_image.py --input_dir data/train --output_dir data/train_crop
    python src/crop_image.py --input_dir data/val --output_dir data/val_crop
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def get_native_rotated_crop(image, polygon):
    """
    Extract a native/rotated crop from the image using a 4-point polygon.
    Preserves the original orientation - NO forced rotation to landscape.
    
    Args:
        image: Input image (numpy array)
        polygon: List of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    
    Returns:
        Cropped and deskewed image patch (preserves original text orientation)
    """
    # Convert polygon to numpy array
    pts = np.array(polygon, dtype=np.float32)
    
    # Order points: [top-left, top-right, bottom-right, bottom-left]
    # This ensures text reads left-to-right in the cropped image
    rect_pts = order_points(pts)
    
    # Calculate width and height from the ordered points
    # Width: distance between top-left and top-right
    width = int(np.linalg.norm(rect_pts[1] - rect_pts[0]))
    # Height: distance between top-left and bottom-left  
    height = int(np.linalg.norm(rect_pts[3] - rect_pts[0]))
    
    # Ensure minimum size
    if width < 2 or height < 2:
        return None
    
    # Define destination points for perspective transform
    # This creates a rectangle of the extracted size
    dst_pts = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype=np.float32)
    
    # Get perspective transform matrix
    M = cv2.getPerspectiveTransform(rect_pts, dst_pts)
    
    # Apply perspective transform to extract the region
    warped = cv2.warpPerspective(image, M, (width, height))
    
    return warped


def order_points(pts):
    """
    Order points in [top-left, top-right, bottom-right, bottom-left] order.
    This ensures the text is readable left-to-right in the cropped image.
    
    Args:
        pts: Numpy array of 4 points
    
    Returns:
        Ordered points
    """
    # Initialize ordered points
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # Sum of coordinates: top-left has smallest sum, bottom-right has largest
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    
    # Difference of coordinates: top-right has min diff, bottom-left has max diff
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    
    return rect


def get_bbox_crop(image, bbox):
    """
    Extract an axis-aligned crop using bounding box.
    
    Args:
        image: Input image (numpy array)
        bbox: [x1, y1, x2, y2]
    
    Returns:
        Cropped image patch
    """
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    # Ensure coordinates are within image bounds
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    
    # Check if valid crop
    if x2 <= x1 or y2 <= y1:
        return None
    
    crop = image[y1:y2, x1:x2]
    return crop


def process_invoice(json_path, output_dir, use_polygon=True):
    """
    Process a single invoice JSON and extract all text crops.
    
    Args:
        json_path: Path to JSON annotation file
        output_dir: Directory to save crops
        use_polygon: If True, use polygon for rotated crops. If False, use bbox.
    
    Returns:
        Number of crops extracted
    """
    # Read JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get image path
    image_path = data.get('image_path')
    if not image_path or not os.path.exists(image_path):
        print(f"Warning: Image not found for {json_path}")
        return 0
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image {image_path}")
        return 0
    
    # Get OCR annotations
    ocr_annotations = data.get('ocr_annotations', [])
    if not ocr_annotations:
        return 0
    
    # Create output subdirectories
    crops_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Extract base name from image path
    base_name = Path(image_path).stem
    
    # Process each annotation
    num_crops = 0
    for idx, ann in enumerate(ocr_annotations):
        text = ann.get('text', '').strip()
        polygon = ann.get('polygon', [])
        bbox = ann.get('bbox', [])
        
        # Skip empty text
        if not text:
            continue
        
        # Extract crop
        if use_polygon and len(polygon) == 4:
            crop = get_native_rotated_crop(image, polygon)
        elif len(bbox) == 4:
            crop = get_bbox_crop(image, bbox)
        else:
            continue
        
        if crop is None or crop.size == 0:
            continue
        
        # Save crop
        crop_filename = f"{base_name}_crop_{idx:04d}.jpg"
        crop_path = os.path.join(crops_dir, crop_filename)
        cv2.imwrite(crop_path, crop)
        
        # Save label (text ground truth)
        label_filename = f"{base_name}_crop_{idx:04d}.txt"
        label_path = os.path.join(labels_dir, label_filename)
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        num_crops += 1
    
    return num_crops


def main():
    parser = argparse.ArgumentParser(description='Crop text regions from synthetic invoices')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing invoice images and JSON files (e.g., data/train)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for cropped images (e.g., data/train_crop)')
    parser.add_argument('--use_polygon', action='store_true', default=True,
                        help='Use polygon for rotated crops (default: True)')
    parser.add_argument('--use_bbox', dest='use_polygon', action='store_false',
                        help='Use bounding box for axis-aligned crops')
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    use_polygon = args.use_polygon
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all JSON files
    json_files = list(Path(input_dir).glob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} invoice JSON files")
    print(f"Output directory: {output_dir}")
    print(f"Crop mode: {'Polygon (native rotated)' if use_polygon else 'Bounding Box (axis-aligned)'}")
    
    # Process all invoices
    total_crops = 0
    for json_path in tqdm(json_files, desc='Processing invoices'):
        num_crops = process_invoice(json_path, output_dir, use_polygon)
        total_crops += num_crops
    
    print(f"\nProcessing complete!")
    print(f"Total crops extracted: {total_crops}")
    print(f"Crops saved to: {os.path.join(output_dir, 'images')}")
    print(f"Labels saved to: {os.path.join(output_dir, 'labels')}")
    
    # Create a dataset info file
    info_file = os.path.join(output_dir, 'dataset_info.txt')
    with open(info_file, 'w') as f:
        f.write(f"Dataset: Text Crops for Recognition Training\n")
        f.write(f"Source: {input_dir}\n")
        f.write(f"Total crops: {total_crops}\n")
        f.write(f"Crop method: {'Polygon (native rotated)' if use_polygon else 'BBox'}\n")
    
    print(f"Dataset info saved to: {info_file}")


if __name__ == '__main__':
    main()