import os
import json
import cv2
import numpy as np
import math
import torch
from torch.utils.data import Dataset
import pyclipper
from shapely.geometry import Polygon

class DBDataset(Dataset):
    def __init__(self, data_dir, image_size=1024, is_train=True):
        self.data_dir = data_dir
        self.image_size = image_size
        self.is_train = is_train
        self.image_paths = []
        self.gt_paths = []
        
        # Find all JSON files
        json_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.json')])
        for jf in json_files:
            self.gt_paths.append(os.path.join(data_dir, jf))
            # Assuming image has same basename; checking extension inside __getitem__ is slow, 
            # but usually key 'image_path' in json tells us.
            
    def __len__(self):
        return len(self.gt_paths)

    def __getitem__(self, index):
        json_path = self.gt_paths[index]
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        image_path = data.get('image_path')
        if not os.path.isabs(image_path):
            # Assume relative to project root, simple fix if needed
            if os.path.exists(image_path):
                pass
            elif os.path.exists(os.path.join(self.data_dir, os.path.basename(image_path))):
                image_path = os.path.join(self.data_dir, os.path.basename(image_path))
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found: {image_path}")
            
        original_h, original_w = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract polygons
        polygons = []
        ignore_tags = []
        for ann in data.get('ocr_annotations', []):
            poly = np.array(ann['polygon'])
            polygons.append(poly)
            ignore_tags.append(False) # Assume all are valid for now
            
        # Resize image and polygons
        image, polygons = self.resize_image(image, polygons, self.image_size)
        
        # 1. Generate Binary Map (GT)
        gt = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        mask = np.ones((self.image_size, self.image_size), dtype=np.float32)
        
        # 2. Generate Threshold Map components
        thresh_map = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        thresh_mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        
        for i in range(len(polygons)):
            poly = polygons[i]
            if ignore_tags[i]:
                cv2.fillPoly(mask, [poly.astype(np.int32)], 0)
                continue
                
            # Generate Shrink Map (Probability Map GT)
            try:
                # Shrink using pyclipper
                polygon_shape = Polygon(poly)
                area = polygon_shape.area
                perimeter = polygon_shape.length
                
                if perimeter == 0: continue
                
                # DBNet formula: r = 0.4 * area / perimeter
                distance = polygon_shape.area * 0.4 / polygon_shape.length
                offset = pyclipper.PyclipperOffset()
                offset.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                shrunk = offset.Execute(-distance)
                
                if len(shrunk) == 0:
                    cv2.fillPoly(mask, [poly.astype(np.int32)], 0)
                    continue
                         
                shrunk = np.array(shrunk[0])
                cv2.fillPoly(gt, [shrunk.astype(np.int32)], 1)
                
                # Generate Threshold Map (Border Map)
                self.draw_border_map(poly, thresh_map, mask=thresh_mask)
                
            except Exception as e:
                # Fallback
                continue

        # Convert to tensors
        # Image: (3, H, W), normalized
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - norm_mean) / norm_std
        
        # Maps: (1, H, W) -> DBNet outputs are same size as input if upsampled, 
        # BUT DBNet head target is typically downsampled by 4 if following paper strictly?
        # Wait, the DBHead implementation uses ConvTranspose2d with stride 2 twice (total x4).
        # So it outputs full resolution.
        # So targets should be full resolution (image_size).
        
        return {
            'image': image,
            'gt': torch.from_numpy(gt).unsqueeze(0),
            'mask': torch.from_numpy(mask).unsqueeze(0),
            'thresh_map': torch.from_numpy(thresh_map).unsqueeze(0),
            'thresh_mask': torch.from_numpy(thresh_mask).unsqueeze(0)
        }

    def resize_image(self, image, polygons, size):
        h, w = image.shape[:2]
        scale = size / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        image = cv2.resize(image, (new_w, new_h))
        
        # Pad to size x size
        padded_image = np.zeros((size, size, 3), dtype=image.dtype)
        padded_image[:new_h, :new_w] = image
        
        new_polygons = []
        for poly in polygons:
            poly = poly * scale
            new_polygons.append(poly)
            
        return padded_image, new_polygons

    def draw_border_map(self, polygon, canvas, mask):
        polygon = np.array(polygon)
        assert polygon.ndim == 2
        assert polygon.shape[1] == 2

        polygon_shape = Polygon(polygon)
        distance = polygon_shape.area * (1 - np.power(1 - 0.4, 2)) / polygon_shape.length
        distance = distance * 0.4 # empirical scale? 
        # Actually standard DBNet uses distance = area * 0.4 / perimeter for shrinking
        # For border map, it expands by same distance.
        subject = [tuple(l) for l in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        padded_polygon = np.array(padding.Execute(distance)[0])
        
        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)
        
        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1

        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin

        xs = np.broadcast_to(np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
        ys = np.broadcast_to(np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

        distance_map = np.zeros((polygon.shape[0], height, width), dtype=np.float32)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            absolute_distance = self._distance(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        
        distance_map = distance_map.min(axis=0)
        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
        
        if xmin_valid >= xmax_valid or ymin_valid >= ymax_valid:
            return

        canvas[ymin_valid:ymax_valid+1, xmin_valid:xmax_valid+1] = np.maximum(
            1 - distance_map[ymin_valid-ymin:ymax_valid-ymin+1, xmin_valid-xmin:xmax_valid-xmin+1],
            canvas[ymin_valid:ymax_valid+1, xmin_valid:xmax_valid+1]
        )

    def _distance(self, xs, ys, point_1, point_2):
        '''
        compute the distance from point to a line
        ys: coordinates in the first axis
        xs: coordinates in the second axis
        point_1, point_2: (x, y), the end of the line
        '''
        height, width = xs.shape[:2]
        square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
        square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
        square_distance = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

        cosin = (square_distance - square_distance_1 - square_distance_2) / (2 * np.sqrt(square_distance_1 * square_distance_2) + 1e-6)
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_distance_1 * square_distance_2 * square_sin / (square_distance + 1e-6))

        result[cosin < 0] = np.sqrt(np.fmin(square_distance_1, square_distance_2))[cosin < 0]
        # self.extend_line(point_1, point_2, result)
        return result
