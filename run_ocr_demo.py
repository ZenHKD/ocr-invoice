import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import DBNetPP, ParSeq

from model.rec.vocab import VOCAB

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Initialize Models
print("Initializing DBNet++...")
det_model = DBNetPP(backbone='resnet50', pretrained=False).to(device)
det_params = sum(p.numel() for p in det_model.parameters() if p.requires_grad)
print(f"DBNet++ Parameters: {det_params / 1e6:.2f} M")
det_model.eval()

print("Initializing ParSeq...")
rec_model = ParSeq(embed_dim=384, num_heads=6, charset=VOCAB).to(device)
rec_params = sum(p.numel() for p in rec_model.parameters() if p.requires_grad)
print(f"ParSeq Parameters: {rec_params / 1e6:.2f} M")
rec_model.eval()

# 2. Dummy Input (or load real image)
from PIL import Image, ImageDraw, ImageFont

# Create a dummy image (H, W, 3)
dummy_img = np.ones((640, 640, 3), dtype=np.uint8) * 255

# Use PIL to render text because cv2.putText doesn't supoprt utf-8
pil_img = Image.fromarray(dummy_img)
draw = ImageDraw.Draw(pil_img)
try:
    # Try to load a font that supports Vietnamese (e.g., DejaVuSans or Arial)
    # Ubuntu/Linux typically has DejaVuSans
    font = ImageFont.truetype("DejaVuSans.ttf", 32)
except IOError:
    # Fallback if specific font not found, though this might still not support Vietnamese perfectly on all systems
    font = ImageFont.load_default()

draw.text((100, 300), "XIN CHÀO, HÚ HÚ HÚ", font=font, fill=(0, 0, 0))

# Convert back to numpy
dummy_img = np.array(pil_img)

# Preprocess for Detection
# Resize, Normalize, ToTensor
img_tensor = torch.from_numpy(dummy_img).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
# Normalization (ImageNet stats usually)
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
img_tensor = (img_tensor - mean) / std

# 3. Detection Inference
print("Running Detection...")
with torch.no_grad():
    det_out = det_model(img_tensor)
    # output: {'binary': ..., 'thresh': ..., 'thresh_binary': ...}
    prob_map = det_out['binary'][0, 0].cpu().numpy()

# Post-processing (Simplified)
# Thresholding
preds = prob_map > 0.3
# Find contours (using OpenCV on the binary map)
preds_uint8 = (preds * 255).astype(np.uint8)
# Resize back to original if needed (DBNet outputs 1/4 size usually, need to check head implementation)
# In our head implementation:
# self.bin_conv has ConvTranspose2d 2x then 2x -> 4x upsampling. So output is same size as input.

contours, _ = cv2.findContours(preds_uint8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

print(f"Found {len(contours)} contours.")

# 4. Recognition Inference
print("Running Recognition...")
crops = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w < 10 or h < 10: continue # Skip small noise

    crop = dummy_img[y:y+h, x:x+w]
    # Resize to (32, 128) expected by ParSeq
    crop_resized = cv2.resize(crop, (128, 32))
    crop_tensor = torch.from_numpy(crop_resized).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    crop_tensor = (crop_tensor - mean) / std # Same normalization

    with torch.no_grad():
        text = rec_model.decode_greedy(crop_tensor)
        print(f"Recognized Text: {text}")
        crops.append(crop_resized)

# Visualization
if len(crops) > 0:
    # Draw contours on the image
    vis_img = dummy_img.copy()
    cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 2)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(dummy_img)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Detection Heatmap")
    plt.imshow(prob_map)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Detected Polygons")
    plt.imshow(vis_img)
    plt.axis('off')

    plt.show()
    print("Inference finished successfully.")
else:
    print("No text detected.")
