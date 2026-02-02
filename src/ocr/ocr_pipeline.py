"""
OCR Pipeline: Text Detection + Recognition
Combines DBNetPP (detection) and PARSeq (recognition) for end-to-end OCR.

Usage:
    %run src/ocr/ocr_pipeline.py
    or
    python src/ocr/ocr_pipeline.py --image data/test/your_image.jpg
"""

import os
import sys

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import cv2
import torch
import numpy as np
import pyclipper
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

from model.det.dbnet import DBNetPP
from model.rec.parseq import ParSeq
from model.rec.vocab import VOCAB


# ============================================================================
# Detection Post-Processor
# ============================================================================

class DBPostProcessor:
    """Post-processor for DBNet binary maps to extract text bounding boxes."""
    
    def __init__(self, thresh=0.3, box_thresh=0.6, max_candidates=1000, unclip_ratio=1.5):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3

    def __call__(self, pred, is_output_polygon=False):
        """
        pred: binary map (H, W)
        Returns: boxes, scores
        """
        prob_map = pred
        h, w = prob_map.shape
        
        mask = prob_map > self.thresh
        contours, _ = cv2.findContours(
            (mask * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        
        boxes = []
        scores = []
        
        for contour in contours:
            if len(contour) < 2:
                continue
                
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape(-1, 2)
            if len(points) < 4: 
                continue
            
            score = self.box_score_fast(prob_map, points)
            if score < self.box_thresh:
                continue
            
            poly = self.unclip(points)
            if len(poly) == 0: 
                continue
                
            poly = np.array(poly).reshape(-1, 2)
            
            if is_output_polygon:
                boxes.append(poly)
            else:
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
        if length == 0:
            return []
        distance = area * self.unclip_ratio / length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(distance)
        return expanded[0] if len(expanded) > 0 else []


# ============================================================================
# OCR Pipeline
# ============================================================================

class OCRPipeline:
    """
    End-to-end OCR pipeline combining detection and recognition.
    """
    
    def __init__(
        self,
        det_model_path: str,
        rec_model_path: str,
        device: str = None,
        det_input_size: int = 1024,
        rec_img_size: tuple = (32, 128),  # (H, W)
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.det_input_size = det_input_size
        self.rec_img_size = rec_img_size  # (H, W)
        
        # Load detection model
        print(f"Loading detection model from {det_model_path}...")
        self.det_model = DBNetPP(backbone='resnet50', pretrained=False)
        if os.path.exists(det_model_path):
            checkpoint = torch.load(det_model_path, map_location=self.device)
            self.det_model.load_state_dict(checkpoint['model_state_dict'])
            print("  Detection model loaded.")
        else:
            print(f"  Warning: Detection model not found at {det_model_path}")
        self.det_model = self.det_model.to(self.device)
        self.det_model.eval()
        
        # Load recognition model
        print(f"Loading recognition model from {rec_model_path}...")
        self.rec_model = ParSeq(
            img_size=self.rec_img_size,
            patch_size=(4, 8),
            embed_dim=384,
            enc_depth=12,
            dec_depth=1,
            num_heads=6,
            charset=VOCAB,
            max_len=64  # Must match training config
        )
        if os.path.exists(rec_model_path):
            checkpoint = torch.load(rec_model_path, map_location=self.device)
            self.rec_model.load_state_dict(checkpoint['model_state_dict'])
            print("  Recognition model loaded.")
        else:
            print(f"  Warning: Recognition model not found at {rec_model_path}")
        self.rec_model = self.rec_model.to(self.device)
        self.rec_model.eval()
        
        # Post-processor for detection
        self.post_processor = DBPostProcessor(unclip_ratio=1.5)
        
        # Normalization constants
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def resize_for_detection(self, image, size=1024):
        """Resize and pad image for detection model."""
        h, w = image.shape[:2]
        scale = size / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        resized_image = cv2.resize(image, (new_w, new_h))
        padded_image = np.zeros((size, size, 3), dtype=image.dtype)
        padded_image[:new_h, :new_w] = resized_image
        
        return padded_image, scale, (h, w)

    def order_points(self, pts):
        """
        Order points in [top-left, top-right, bottom-right, bottom-left] order.
        """
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Sum of coordinates
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        
        # Diff of coordinates
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        
        return rect

    def crop_text_region(self, image, box):
        """
        Crop and rectify a text region using perspective transform.
        """
        box = np.array(box, dtype=np.float32)
        
        # 1. Order points
        rect = self.order_points(box)
        (tl, tr, br, bl) = rect
        
        # 2. Compute width and height of the new image
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        # 3. Construct destination points
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
            
        # 4. Perspective warp
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        return warped

    def preprocess_for_recognition(self, crop):
        """
        Resize cropped text region to recognition model input size.
        Maintains aspect ratio and pads if necessary.
        """
        if crop is None or crop.size == 0:
            return None
            
        h, w = crop.shape[:2]
        target_h, target_w = self.rec_img_size  # (32, 128)
        
        # Scale to target height
        scale = target_h / h
        new_w = int(w * scale)
        
        # Limit width
        if new_w > target_w:
            new_w = target_w
            scale = new_w / w
            
        resized = cv2.resize(crop, (new_w, target_h))
        
        # Pad to target width
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        padded[:, :new_w] = resized
        
        return padded

    @torch.no_grad()
    def detect(self, image):
        """
        Run text detection on an image.
        
        Args:
            image: RGB numpy array (H, W, 3)
            
        Returns:
            boxes: List of polygon boxes (each is Nx2 array)
            scores: List of confidence scores
        """
        input_image, scale, (orig_h, orig_w) = self.resize_for_detection(
            image, size=self.det_input_size
        )
        
        # To tensor
        tensor_img = input_image.astype(np.float32) / 255.0
        tensor_img = torch.from_numpy(tensor_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        tensor_img = (tensor_img - self.norm_mean) / self.norm_std
        
        # Forward
        preds = self.det_model(tensor_img)
        prob_map = preds['binary'][0, 0].cpu().numpy()
        
        # Post-process
        boxes, scores = self.post_processor(prob_map, is_output_polygon=True)
        
        # Scale back to original image size
        final_boxes = []
        for box in boxes:
            box = box / scale
            box[:, 0] = np.clip(box[:, 0], 0, orig_w - 1)
            box[:, 1] = np.clip(box[:, 1], 0, orig_h - 1)
            final_boxes.append(box.astype(np.int32))
            
        return final_boxes, scores

    @torch.no_grad()
    def recognize(self, crops):
        """
        Run text recognition on a batch of cropped text images.
        
        Args:
            crops: List of RGB numpy arrays
            
        Returns:
            texts: List of recognized strings
            scores: List of confidence scores
        """
        self.rec_model.eval()
        texts = []
        scores = []
        
        if len(crops) == 0:
            return []
            
        # Preprocess all crops
        batch = []
        valid_indices = []
        for i, crop in enumerate(crops):
            processed = self.preprocess_for_recognition(crop)
            if processed is not None:
                batch.append(processed)
                valid_indices.append(i)
        
        if len(batch) == 0:
            return [''] * len(crops)
        
        # Stack and convert to tensor
        batch = np.stack(batch, axis=0)  # (B, H, W, 3)
        batch = batch.astype(np.float32) / 255.0
        batch = torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device)  # (B, 3, H, W)
        batch = (batch - self.norm_mean) / self.norm_std
        
        # Forward
        texts = self.rec_model.decode_greedy(batch)
        
        # Map back to original indices
        result = [''] * len(crops)
        for idx, text in zip(valid_indices, texts):
            result[idx] = text
            
        return result

    def __call__(self, image):
        """
        Run full OCR pipeline on an image.
        
        Args:
            image: RGB numpy array (H, W, 3)
            
        Returns:
            results: List of dicts with 'box', 'text', 'score', 'crop'
        """
        # Detection
        boxes, scores = self.detect(image)
        
        # Crop text regions
        crops = []
        for box in boxes:
            crop = self.crop_text_region(image, box)
            crops.append(crop)
        
        # Recognition
        texts = self.recognize(crops)
        
        # Combine results
        results = []
        for box, text, score, crop in zip(boxes, texts, scores, crops):
            results.append({
                'box': box,
                'text': text,
                'score': score,
                'crop': crop
            })
            
        return results

    def visualize(self, image, results, figsize=(15, 15)):
        """
        Visualize OCR results on the image.
        
        Args:
            image: RGB numpy array
            results: List of dicts from __call__
            figsize: Matplotlib figure size
        """
        vis_img = image.copy()
        
        for i, r in enumerate(results):
            box = r['box']
            
            # Draw polygon
            cv2.polylines(vis_img, [box], isClosed=True, color=(0, 255, 0), thickness=2)
            
            # Draw region ID 
            x, y = box[0]
            cv2.putText(
                vis_img, str(i + 1), (x, max(0, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA
            )
        
        plt.figure(figsize=figsize)
        plt.imshow(vis_img)
        plt.title(f"OCR Results: {len(results)} text regions detected")
        plt.axis('off')
        plt.show()
        
        return vis_img

    def visualize_crops(self, results, num_cols=4, figsize=(20, 20)):
        """
        Visualize cropped images alongside recognition results.
        
        Args:
            results: List of dicts from __call__
            num_cols: Number of columns in the grid
        """
        n = len(results)
        if n == 0:
            print("No results to visualize.")
            return

        num_rows = (n + num_cols - 1) // num_cols
        
        plt.figure(figsize=figsize)
        for i, res in enumerate(results):
            plt.subplot(num_rows, num_cols, i + 1)
            crop = res.get('crop')
            if crop is not None:
                plt.imshow(crop)
            
            text = res['text']
            score = res['score']
            plt.title(f"[{i+1}] {text}\nScore: {score:.2f}", fontsize=10)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run OCR on an image")
    parser.add_argument('--image', type=str, default=None, 
                        help='Path to input image. If not provided, uses first image from data/test/')
    parser.add_argument('--det-model', type=str, 
                        default='../../best_model/det/best_model_detection.pth',
                        help='Path to detection model')
    parser.add_argument('--rec-model', type=str,
                        default='../../best_model/rec/best_model_rec_acc.pth',
                        help='Path to recognition model')
    parser.add_argument('--crop', action='store_true',
                        help='Visualize cropped text regions instead of full image')
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = OCRPipeline(
        det_model_path=args.det_model,
        rec_model_path=args.rec_model
    )
    
    # Get image path
    if args.image:
        img_path = args.image
    else:
        test_dir = '../../data/test'
        all_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if len(all_files) == 0:
            print("No images found in data/test/")
            return
        all_files.sort()
        img_path = os.path.join(test_dir, all_files[0])
    
    print(f"\nRunning OCR on: {img_path}")
    
    # Load image
    image = cv2.imread(img_path)
    if image is None:
        print(f"Error: Could not load image from {img_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run OCR
    results = pipeline(image)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Detected {len(results)} text regions:")
    print(f"{'='*60}")
    for i, r in enumerate(results):
        print(f"  [{i+1}] Text: '{r['text']}' (score: {r['score']:.3f})")
    print(f"{'='*60}\n")
    
    # Visualize
    if args.crop:
        print("Visualizing cropped text regions...")
        pipeline.visualize_crops(results)
    else:
        print("Visualizing full image results...")
        pipeline.visualize(image, results)


if __name__ == '__main__':
    main()
