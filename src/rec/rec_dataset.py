import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from model.rec.tokenizer import Tokenizer
from model.rec.vocab import VOCAB

class RecDataset(Dataset):
    def __init__(self, data_dir, image_size=(128, 32), max_len=64, is_train=True):
        self.data_dir = data_dir
        self.image_size = image_size # (W, H)
        self.max_len = max_len
        self.is_train = is_train
        self.samples = []
        
        self.tokenizer = Tokenizer(VOCAB)
        
        self.img_dir = os.path.join(data_dir, 'images')
        self.lbl_dir = os.path.join(data_dir, 'labels')
        
        if not os.path.exists(self.img_dir) or not os.path.exists(self.lbl_dir):
            print(f"Warning: Image or Label directory not found in {data_dir}. Expected 'images' and 'labels'.")
            return
            
        # Load samples
        self._load_samples()

    def _load_samples(self):
        # We iterate over labels as the ground truth source
        label_files = sorted([f for f in os.listdir(self.lbl_dir) if f.endswith('.txt')])
        
        for lf in label_files:
            # Construct paths
            label_path = os.path.join(self.lbl_dir, lf)
            
            # Image path: assume same basename, check extension
            base_name = os.path.splitext(lf)[0]
            
            # Check common extensions
            found_img = False
            image_path = ""
            for ext in ['.jpg', '.png', '.jpeg']:
                img_p = os.path.join(self.img_dir, base_name + ext)
                if os.path.exists(img_p):
                    image_path = img_p
                    found_img = True
                    break
            
            if not found_img:
                continue
                
            # Read label
            with open(label_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                
            if self.is_train and len(text) == 0:
                continue
                
            self.samples.append({
                'image_path': image_path,
                'text': text
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_path = sample['image_path']
        text = sample['text']
        
        image = cv2.imread(image_path)
        if image is None:
            # Fallback or error
            # For robustness, we might want to return another sample or raise
            raise ValueError(f"Image not found: {image_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        # Input images are already crops, so we just resize/pad
        crop, width_ratio = self.resize_norm_img(image, self.image_size)
        
        # To Tensor
        # Image: (3, H, W)
        crop = crop.astype(np.float32) / 255.0
        crop = torch.from_numpy(crop).permute(2, 0, 1)
        
        # Normalize
        norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        crop = (crop - norm_mean) / norm_std
        
        # Label Encoding
        label_tensor = self.tokenizer.encode([text], max_length=self.max_len)[0]
        
        # Manually Pad to max_len + 2 (BOS + EOS + max_len)
        target_len = self.max_len + 2
        if len(label_tensor) < target_len:
            pad_len = target_len - len(label_tensor)
            # Use pad_id from tokenizer
            pads = torch.full((pad_len,), self.tokenizer.pad_id, dtype=torch.long)
            label_tensor = torch.cat([label_tensor, pads])
        elif len(label_tensor) > target_len:
            # Should be handled by tokenizer, but just in case
             label_tensor = label_tensor[:target_len]

        return {
            'image': crop,
            'label': label_tensor,
            'text': text,
            'width_ratio': width_ratio
        }

    def resize_norm_img(self, img, image_size):
        # image_size: (W, H)
        w_target, h_target = image_size
        h, w = img.shape[:2]
        
        ratio = w / float(h)
        
        # Resize logic: 
        # 1. Resize height to h_target
        # 2. Resize width scaling with ratio, but max w_target
        # 3. Pad if necessary
        
        new_h = h_target
        new_w = int(new_h * ratio)
        
        if new_w > w_target:
            new_w = w_target
            
        resized = cv2.resize(img, (new_w, new_h))
        
        padded = np.zeros((h_target, w_target, 3), dtype=img.dtype)
        padded[:new_h, :new_w, :] = resized
        
        return padded, new_w / w_target
