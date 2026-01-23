#!/usr/bin/env python3
"""
Visualization script to verify OCR polygon annotations match text in generated images.

Usage:
    python test_visualize.py data/test_output/invoice_000000_29a4b303.json
    python test_visualize.py data/test_output/  # Visualize all in directory
"""

import json
import sys
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random


def get_random_color():
    """Generate a random bright color for visibility."""
    return (
        random.randint(100, 255),
        random.randint(100, 255),
        random.randint(50, 150),
    )


def visualize_polygons(json_path: str, output_dir: str = None):
    """
    Draw polygon annotations on the image and save/display the result.
    
    Args:
        json_path: Path to the JSON annotation file
        output_dir: Directory to save visualized images (optional)
    """
    # Load annotation
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Load image
    img_path = data.get("image_path")
    if not os.path.isabs(img_path):
        # Make path relative to JSON file location
        img_path = os.path.join(os.path.dirname(json_path), os.path.basename(img_path))
    
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return None
    
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    # Try to get a font for labels
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        font = ImageFont.load_default()
    
    # Get annotations
    annotations = data.get("ocr_annotations", [])
    
    if not annotations:
        print(f"No annotations found in {json_path}")
        return img
    
    print(f"\nVisualizing: {os.path.basename(json_path)}")
    print(f"Image size: {data.get('image_size')}")
    print(f"Layout: {data.get('layout_type')}")
    print(f"Found {len(annotations)} text annotations:")
    print("-" * 50)
    
    # Draw each polygon
    for i, ann in enumerate(annotations):
        text = ann.get("text", "")
        polygon = ann.get("polygon", [])
        bbox = ann.get("bbox", [])
        
        color = get_random_color()
        
        # Prefer polygon format, fallback to bbox
        if polygon and len(polygon) == 4:
            # Draw polygon (4 corner points)
            pts = [(p[0], p[1]) for p in polygon]
            draw.polygon(pts, outline=color, width=2)
            # Use first point for label position
            x1, y1 = pts[0]
        elif len(bbox) == 4:
            # Fallback to old bbox format
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        else:
            print(f"  [{i}] Invalid annotation format")
            continue
        
        # Draw label background
        label = f"{i}: {text[:20]}..." if len(text) > 20 else f"{i}: {text}"
        label_bbox = font.getbbox(label)
        label_w = label_bbox[2] - label_bbox[0]
        label_h = label_bbox[3] - label_bbox[1]
        
        # Position label above polygon if possible
        label_y = max(0, y1 - label_h - 2)
        draw.rectangle([x1, label_y, x1 + label_w + 4, label_y + label_h + 2], fill=(0, 0, 0))
        draw.text((x1 + 2, label_y), label, fill=color, font=font)
        
        poly_str = str(polygon[:2]) + "..." if polygon else str(bbox)
        print(f"  [{i}] polygon={poly_str} text=\"{text[:40]}{'...' if len(text) > 40 else ''}\"")
    
    print("-" * 50)
    
    # Save or show
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"viz_{os.path.basename(img_path)}")
        img.save(out_path)
        print(f"Saved: {out_path}")
    
    return img


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_visualize.py <json_file_or_directory> [output_dir]")
        print("\nExamples:")
        print("  python test_visualize.py data/test_output/invoice_000000.json")
        print("  python test_visualize.py data/test_output/ data/visualized/")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/visualized"
    
    if os.path.isfile(input_path):
        # Single file
        img = visualize_polygons(input_path, output_dir)
        if img:
            print(f"\nVisualization complete!")
    
    elif os.path.isdir(input_path):
        # Directory - process all JSON files
        json_files = list(Path(input_path).glob("*.json"))
        print(f"Found {len(json_files)} JSON files in {input_path}")
        
        for json_file in json_files:
            visualize_polygons(str(json_file), output_dir)
        
        print(f"\nVisualized {len(json_files)} images to {output_dir}/")
    
    else:
        print(f"Error: {input_path} not found")
        sys.exit(1)


if __name__ == "__main__":
    main()
