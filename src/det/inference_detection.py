import os
import sys

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import cv2
import torch
import numpy as np
import pyclipper
from shapely.geometry import Polygon
from model.det.dbnet import DBNetPP
import matplotlib.pyplot as plt

class DBPostProcessor:
    def __init__(self, thresh=0.3, box_thresh=0.6, max_candidates=1000, unclip_ratio=1.5):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3

    def __call__(self, pred, is_output_polygon=False):
        '''
        pred: binary map (H, W)
        '''
        prob_map = pred
        h, w = prob_map.shape
        
        # Thresholding
        mask = prob_map > self.thresh
        
        # Find contours
        contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        scores = []
        
        for contour in contours:
            if len(contour) < 2:
                continue
                
            # Box score
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape(-1, 2)
            if len(points) < 4: 
                continue
            
            score = self.box_score_fast(prob_map, points)
            if score < self.box_thresh:
                continue
            
            # Unclip
            poly = self.unclip(points)
            if len(poly) == 0: 
                continue
                
            poly = np.array(poly).reshape(-1, 2)
            
            if is_output_polygon:
                boxes.append(poly)
            else:
                # Rotated rectangle
                rect = cv2.minAreaRect(poly)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                boxes.append(box)
            scores.append(score)
            
        return boxes, scores

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax+1, xmin:xmax+1], mask)[0]

    def unclip(self, box):
        poly = Polygon(box)
        area = poly.area
        length = poly.length
        distance = area * self.unclip_ratio / length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(distance)
        return expanded[0] if len(expanded) > 0 else []

def resize_image_inference(image, size=1024):
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    resized_image = cv2.resize(image, (new_w, new_h))
    
    # Pad to size x size
    padded_image = np.zeros((size, size, 3), dtype=image.dtype)
    padded_image[:new_h, :new_w] = resized_image
    
    return padded_image, scale, (h, w)

def run_inference():
    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = '../../best_model/det/best_model_detection.pth'
    test_dir = '../../data/test'
    
    print(f"Loading model from {model_path}...")
    model = DBNetPP(backbone='resnet50', pretrained=False) 
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict)
        print("Model loaded successfully.")
    else:
        print(f"Warning: Model not found at {model_path}. Running with random weights.")
        
    model = model.to(device)
    model.eval()
    
    post_processor = DBPostProcessor(unclip_ratio=1.5)
    
    # 2. Select Images
    all_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if len(all_files) == 0:
        print("No images found in data/test")
        return

    all_files.sort()
    selected_files = all_files[:5]
        
    print(f"Running inference on: {selected_files}")
    
    # 3. Inference Loop
    norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    for filename in selected_files:
        img_path = os.path.join(test_dir, filename)
        original_image = cv2.imread(img_path)
        if original_image is None: continue
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        input_image, scale, (orig_h, orig_w) = resize_image_inference(original_image, size=1024)
        
        tensor_img = input_image.astype(np.float32) / 255.0
        tensor_img = torch.from_numpy(tensor_img).permute(2, 0, 1).unsqueeze(0).to(device)
        tensor_img = (tensor_img - norm_mean) / norm_std
        
        # Forward
        with torch.no_grad():
            preds = model(tensor_img)
            
        prob_map = preds['binary'][0, 0].cpu().numpy()
        
        # Post-process
        boxes, scores = post_processor(prob_map, is_output_polygon=True)
        
        final_boxes = []
        for box in boxes:
            box = box / scale
            box[:, 0] = np.clip(box[:, 0], 0, orig_w - 1)
            box[:, 1] = np.clip(box[:, 1], 0, orig_h - 1)
            final_boxes.append(box.astype(np.int32))
            
        # Visualize
        vis_img = original_image.copy()
        cv2.polylines(vis_img, final_boxes, isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Show
        # Print results
        print(f"Processed {filename}: Found {len(final_boxes)} text regions.")

        plt.figure(figsize=(10, 10))
        plt.imshow(vis_img)
        plt.title(f"Result: {filename}")
        plt.axis('off')
        plt.show()
        
if __name__ == '__main__':
    run_inference()
